#include "kernel_executor.hpp"

#include <cstring>

#if FAKEGPU_PTX_EXECUTION

#include "global_state.hpp"
#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/sass/exec.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fake_gpu::kernel {
namespace {

struct FreeMemory {
    std::size_t mapped_bytes = 0;
    void operator()(void* pointer) const {
        if (mapped_bytes) munmap(pointer, mapped_bytes);
        else std::free(pointer);
    }
};

struct Storage {
    std::unique_ptr<void, FreeMemory> storage;
    std::size_t size;
    bool constant;
    std::uint64_t address;
};

struct Global {
    void* pointer;
    std::size_t size;
};

struct Module {
    vgpu::ptx::Module code;
    std::shared_ptr<vgpu::sass::Module> cubin;
    vgpu::MemoryManager* memory = nullptr;
    vgpu::DeviceProfile profile;
    int device;
    std::vector<Storage> storage;
    std::unordered_map<std::string, Global> globals;
    vgpu::exec::SymbolTable symbols;
    std::unordered_map<std::string, void*> functions;

    ~Module() { if (cubin) vgpu::sass::unload(*cubin, *memory); }
};

struct Parameter {
    std::size_t ordinal;
    std::size_t offset;
    std::size_t size;
};

struct Function {
    Module* module;
    const vgpu::ptx::EntryFn* entry = nullptr;
    std::string name;
    std::vector<Parameter> parameters;
};

struct DeviceMemory {
    explicit DeviceMemory(std::uint64_t capacity, unsigned device) : memory(capacity, device) {
        memory.set_alloc_alignment(static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE)));
    }
    vgpu::MemoryManager memory;
    std::vector<std::uint64_t> host_mappings;
};

std::mutex execution_mutex;
std::unordered_map<int, std::unique_ptr<DeviceMemory>> device_memories;
std::unordered_map<void*, std::unique_ptr<Module>> modules;
std::unordered_map<void*, Function> functions;
// Handles are never dereferenced or recycled, so unloading a module cannot
// turn one of its stale handles into a function in a later module.
std::uintptr_t next_handle = 1;

void* new_handle() { return reinterpret_cast<void*>(next_handle++); }

int error_code(vgpu::Err code) {
    switch (code) {
        case vgpu::Err::InvalidValue:
        case vgpu::Err::LaunchConfig:
        case vgpu::Err::LaunchCluster: return 1;
        case vgpu::Err::OutOfMemory: return 2;
        case vgpu::Err::PtxParse: return 218;
        case vgpu::Err::NotFound: return 500;
        case vgpu::Err::InvalidPointer:
        case vgpu::Err::OutOfBounds:
        case vgpu::Err::MisalignedAccess:
        case vgpu::Err::UseAfterFree: return 700;
        case vgpu::Err::LaunchResources: return 701;
        case vgpu::Err::UnsupportedPtx:
        case vgpu::Err::Unsupported: return 801;
        default: return 719;
    }
}

template <class Operation>
int execute(Operation operation) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    try {
        return operation();
    } catch (const vgpu::Error& error) {
        std::fprintf(stderr, "[FakeGPU kernel] %s\n", error.what());
        return error_code(error.code());
    } catch (const std::bad_alloc&) {
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[FakeGPU kernel] %s\n", error.what());
        return 719;
    }
}

vgpu::DeviceProfile device_profile(const Device& device) {
    const auto& source = device.profile;
    vgpu::DeviceProfile profile;
    profile.id = source.id;
    profile.vendor = "nvidia";
    profile.architecture = to_string(source.architecture);
    profile.cc_major = source.compute_major;
    profile.cc_minor = source.compute_minor;
    profile.warp_size = 32;
    profile.vram_bytes = device.total_memory;
    auto& limits = profile.limits;
    limits.max_threads_per_block = 1024;
    limits.max_block_dim = {1024, 1024, 64};
    limits.max_grid_dim = {2147483647, 2147483647, 2147483647};
    limits.shared_mem_per_block = source.shared_mem_per_block;
    limits.shared_mem_per_block_optin = source.shared_mem_per_block_optin;
    limits.shared_mem_per_sm = source.shared_mem_per_sm;
    limits.registers_per_block = source.regs_per_block;
    limits.registers_per_sm = source.regs_per_multiprocessor;
    limits.max_threads_per_sm = source.max_threads_per_multiprocessor;
    limits.max_blocks_per_sm = source.max_blocks_per_multiprocessor;
    limits.multiprocessors = source.sm_count;
    return profile;
}

bool target_compatible(const std::string& target, unsigned actual) {
    const auto start = target.find("sm_");
    if (start == std::string::npos) return true;
    char* end = nullptr;
    const long requested = std::strtol(target.c_str() + start + 3, &end, 10);
    return requested <= actual && (*end != 'a' || requested == actual) &&
           (*end != 'f' || requested / 10 == actual / 10);
}

void validate_target(const Module& module) {
    if (!target_compatible(module.code.target, module.profile.cc_major * 10 + module.profile.cc_minor)) {
        throw vgpu::Error::make(vgpu::Err::PtxParse, "module targets ", module.code.target,
                               " but the selected profile has compute capability ",
                               module.profile.cc_major, ".", module.profile.cc_minor);
    }
}

void initialize_globals(Module& module) {
    for (const auto& declaration : module.code.globals) {
        void* pointer = nullptr;
        const auto alignment = std::max<std::size_t>(declaration.align, sizeof(void*));
        if (posix_memalign(&pointer, alignment, declaration.size) != 0) throw std::bad_alloc();
        module.storage.push_back({std::unique_ptr<void, FreeMemory>(pointer),
                                  declaration.size, declaration.is_const, 0});
        std::memset(pointer, 0, declaration.size);
        if (!declaration.init.empty()) {
            std::memcpy(pointer, declaration.init.data(), declaration.init.size());
        }
        module.symbols[declaration.name] = reinterpret_cast<std::uintptr_t>(pointer);
        module.globals.emplace(declaration.name, Global{pointer, declaration.size});
    }
    for (std::size_t index = 0; index < module.code.funcs.size(); ++index) {
        module.symbols[module.code.funcs[index]->name] = vgpu::kFuncVaBase + index * vgpu::kFuncVaStride;
    }
    for (const auto& entry : module.code.entries) {
        void* handle = new_handle();
        module.functions.emplace(entry.name, handle);
        module.symbols[entry.name] = vgpu::kKernelVaBase +
            reinterpret_cast<std::uintptr_t>(handle) * vgpu::kKernelVaStride;
    }
    for (const auto& declaration : module.code.globals) {
        auto* destination = static_cast<unsigned char*>(module.globals.at(declaration.name).pointer);
        for (const auto& initializer : declaration.init_symbols) {
            const auto symbol = module.symbols.find(initializer.name);
            if (symbol == module.symbols.end()) {
                throw vgpu::Error::make(vgpu::Err::UnsupportedPtx,
                                       "unresolved global initializer: ", initializer.name);
            }
            if (initializer.offset > declaration.size ||
                sizeof(std::uint64_t) > declaration.size - initializer.offset) {
                throw vgpu::Error(vgpu::Err::PtxParse, "global pointer initializer exceeds its storage");
            }
            std::memcpy(destination + initializer.offset, &symbol->second, sizeof(std::uint64_t));
        }
    }
}

void initialize_cubin_storage(Module& module) {
    // Map each section at its CUDA address so existing CPU libraries can use
    // globals and relocated pointers directly. Never replace an existing map.
    for (const auto& [name, address] : module.cubin->section_va) {
        const auto* section = module.cubin->cubin.section(name);
        void* requested = reinterpret_cast<void*>(address);
        void* pointer = mmap(requested, section->size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (pointer == MAP_FAILED) throw std::bad_alloc();
        // Older kernels may ignore MAP_FIXED_NOREPLACE and choose another VA.
        if (pointer != requested) {
            munmap(pointer, section->size);
            throw vgpu::Error(vgpu::Err::Unsupported, "cubin globals require fixed non-replacing memory maps");
        }
        module.storage.push_back({std::unique_ptr<void, FreeMemory>(pointer, FreeMemory{section->size}),
                                  section->size, name.rfind(".nv.constant", 0) == 0, address});
        module.memory->read(address, pointer, section->size);
        for (const auto& [symbol, va] : module.cubin->symbol_va) {
            if (va < address || va - address >= section->size) continue;
            module.globals.emplace(symbol, Global{
                reinterpret_cast<void*>(va), module.cubin->symbol_size.at(symbol)});
        }
    }
}

DeviceMemory& device_memory(const Module& module) {
    auto& memory = device_memories[module.device];
    if (!memory) memory = std::make_unique<DeviceMemory>(module.profile.vram_bytes, module.device);
    return *memory;
}

void clear_mappings(DeviceMemory& device) {
    for (const auto address : device.host_mappings) device.memory.unmap_host(address);
    device.host_mappings.clear();
}

int pack_arguments(const std::vector<Parameter>& parameters, void** arguments, void** extra,
                   std::vector<std::vector<std::uint8_t>>& packed) {
    if (arguments && extra) return 1;
    const unsigned char* buffer = nullptr;
    std::size_t buffer_size = 0;
    if (extra) {
        bool has_buffer = false, has_size = false;
        std::size_t index = 0;
        for (; index < 4 && extra[index]; index += 2) {
            const auto key = reinterpret_cast<std::uintptr_t>(extra[index]);
            if (key == 1 && !has_buffer) {
                buffer = static_cast<const unsigned char*>(extra[index + 1]);
                has_buffer = true;
            } else if (key == 2 && !has_size && extra[index + 1]) {
                buffer_size = *static_cast<const std::size_t*>(extra[index + 1]);
                has_size = true;
            } else {
                return 1;
            }
        }
        if (extra[index] || !has_buffer || !has_size || (!buffer && buffer_size)) return 1;
    }
    for (const auto& parameter : parameters) {
        const void* source = nullptr;
        if (extra) {
            if (parameter.offset > buffer_size || parameter.size > buffer_size - parameter.offset || !buffer) return 1;
            source = buffer + parameter.offset;
        } else {
            if (!arguments || !arguments[parameter.ordinal]) return 1;
            source = arguments[parameter.ordinal];
        }
        packed.emplace_back(parameter.size);
        std::memcpy(packed.back().data(), source, parameter.size);
    }
    return 0;
}

} // namespace

bool available() { return true; }

int ptx_compatible(const std::string& image, unsigned compute_capability) {
    return execute([&] {
        return target_compatible(vgpu::ptx::parse(image).target, compute_capability) ? 0 : 209;
    });
}

int cubin_compatible(const std::string& image, unsigned compute_capability) {
    try {
        const auto cubin = vgpu::sass::parse_cubin(
            reinterpret_cast<const std::uint8_t*>(image.data()), image.size());
        if (!vgpu::sass::runs_on(cubin.sm, cubin.arch_specific, compute_capability)) return 209;
        return cubin.sm < 75 ? 801 : 0;
    } catch (const vgpu::Error& error) {
        return error.code() == vgpu::Err::InvalidValue ? 200 : error_code(error.code());
    } catch (const std::bad_alloc&) {
        return 2;
    }
}

int load(const std::string& image, void** output) {
    if (!output) return 1;
    *output = nullptr;
    return execute([&] {
        auto& state = GlobalState::instance();
        state.initialize();
        auto module = std::make_unique<Module>();
        module->device = state.get_current_device();
        module->profile = device_profile(state.get_device(module->device));
        std::vector<Function> entries;
        if (vgpu::sass::is_cubin(image.data(), image.size())) {
            if (const int status = cubin_compatible(image, module->profile.cc_major * 10 + module->profile.cc_minor))
                return status;
            module->memory = &device_memory(*module).memory;
            module->cubin = vgpu::sass::load(reinterpret_cast<const std::uint8_t*>(image.data()),
                                            image.size(), *module->memory, module->profile);
            initialize_cubin_storage(*module);
            for (const auto& entry : module->cubin->cubin.kernels) {
                Function function{module.get(), nullptr, entry.name, {}};
                for (const auto& param : entry.params)
                    function.parameters.push_back({param.ordinal, param.offset, param.size});
                module->functions.emplace(entry.name, new_handle());
                entries.push_back(std::move(function));
            }
        } else {
            module->code = vgpu::ptx::parse(image);
            validate_target(*module);
            initialize_globals(*module);
            for (const auto& entry : module->code.entries) {
                Function function{module.get(), &entry, entry.name, {}};
                std::size_t offset = 0;
                for (const auto& param : entry.params) {
                    const auto alignment = param.align ? param.align : param.ty.bytes();
                    offset = (offset + alignment - 1) / alignment * alignment;
                    function.parameters.push_back({function.parameters.size(), offset, param.size});
                    offset += param.size;
                }
                entries.push_back(std::move(function));
            }
        }
        void* handle = new_handle();
        Module* stored = module.get();
        modules.emplace(handle, std::move(module));
        for (auto& entry : entries) {
            functions.emplace(stored->functions.at(entry.name), std::move(entry));
        }
        *output = handle;
        return 0;
    });
}

int function(void* handle, const char* name, void** output) {
    if (!name || !output) return 1;
    *output = nullptr;
    return execute([&] {
        const auto module = modules.find(handle);
        if (module == modules.end()) return 400;
        const auto found = module->second->functions.find(name);
        if (found == module->second->functions.end()) return 500;
        *output = found->second;
        return 0;
    });
}

int global(void* handle, const char* name, void** pointer, std::size_t* bytes) {
    if (!name || (!pointer && !bytes)) return 1;
    return execute([&] {
        const auto module = modules.find(handle);
        if (module == modules.end()) return 400;
        const auto found = module->second->globals.find(name);
        if (found == module->second->globals.end()) return 500;
        if (pointer) *pointer = found->second.pointer;
        if (bytes) *bytes = found->second.size;
        return 0;
    });
}

int unload(void* handle) {
    return execute([&] {
        const auto module = modules.find(handle);
        if (module == modules.end()) return 400;
        const auto memory = device_memories.find(module->second->device);
        if (memory != device_memories.end()) clear_mappings(*memory->second);
        for (const auto& [name, function] : module->second->functions) functions.erase(function);
        modules.erase(module);
        return 0;
    });
}

int launch(void* handle, const unsigned grid[3], const unsigned block[3],
           std::size_t shared_bytes, void** arguments, void** extra) {
    if (!grid || !block || shared_bytes > std::numeric_limits<std::uint32_t>::max()) return 1;
    return execute([&] {
        const auto found = functions.find(handle);
        if (found == functions.end()) return 400;
        const auto& function = found->second;
        const auto& module = *function.module;
        auto& state = GlobalState::instance();
        if (module.device != state.get_current_device()) return 400;
        std::vector<std::vector<std::uint8_t>> packed;
        if (const int status = pack_arguments(function.parameters, arguments, extra, packed)) return status;

        // Device-side malloc storage survives launches. Host mappings follow
        // the current allocations, including frees and module unloads.
        auto& device = device_memory(module);
        auto& memory = device.memory;
        clear_mappings(device);
        auto map_host = [&](std::uint64_t address, void* pointer, std::size_t size, bool constant = false) {
            memory.map_host(address, pointer, size, constant);
            device.host_mappings.push_back(address);
        };
        for (const auto& [pointer, size] : state.snapshot_allocations()) {
            map_host(reinterpret_cast<std::uintptr_t>(pointer), pointer, size);
        }
        for (const auto& [module_handle, loaded] : modules) {
            if (loaded->device != module.device) continue;
            for (const auto& storage : loaded->storage) {
                void* pointer = storage.storage.get();
                const auto address = storage.address ? storage.address : reinterpret_cast<std::uintptr_t>(pointer);
                map_host(address, pointer, storage.size, storage.constant);
            }
        }
        vgpu::exec::LaunchConfig configuration;
        std::copy_n(grid, 3, configuration.grid.begin());
        std::copy_n(block, 3, configuration.block.begin());
        configuration.shared_bytes = shared_bytes;
        configuration.device_ordinal = module.device;
        configuration.device_count = state.get_device_count();
        vgpu::exec::KernelTable kernels;
        if (!module.cubin) {
            for (const auto& [module_handle, loaded] : modules) {
                if (loaded->device != module.device || loaded->cubin) continue;
                for (const auto& entry : loaded->code.entries)
                    kernels.emplace(loaded->symbols.at(entry.name),
                                    vgpu::exec::KernelRef{&entry, &loaded->symbols});
            }
            configuration.kernels = &kernels;
        }
        if (module.cubin)
            vgpu::sass::launch(*module.cubin, function.name, configuration, packed, memory, module.profile);
        else
            vgpu::exec::launch(*function.entry, configuration, packed, memory, module.profile, &module.symbols);
        return 0;
    });
}

std::string function_name(void* handle) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    const auto found = functions.find(handle);
    return found == functions.end() ? std::string{} : found->second.name;
}

// CUDA copies may receive a global's address from an initializer or a kernel,
// not just cuModuleGetGlobal. These addresses retain the cubin's virtual VA.
void* host_pointer(const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    // SASS exposes module-relative constant/local/shared windows inside kernels.
    // A host copy has no module with which to resolve those addresses.
    if (address >= vgpu::sass::kConstWindow && address < 0x7000'0000'0000ull)
        throw vgpu::Error(vgpu::Err::Unsupported, "host copies of kernel-private addresses are unsupported");
    if (!vgpu::is_device_va(address)) return const_cast<void*>(pointer);
    for (const auto& [handle, module] : modules) {
        for (const auto& storage : module->storage) {
            if (!storage.address || address < storage.address) continue;
            const auto offset = address - storage.address;
            if (offset <= storage.size && bytes <= storage.size - offset)
                return static_cast<unsigned char*>(storage.storage.get()) + offset;
        }
    }
    throw vgpu::Error(vgpu::Err::InvalidPointer, "copy address is outside a live cubin data section");
}

int copy(void* destination, const void* source, std::size_t bytes) {
    return execute([&] {
        std::memcpy(host_pointer(destination, bytes), host_pointer(source, bytes), bytes);
        return 0;
    });
}

int fill(void* destination, unsigned value, std::size_t count, unsigned element_size) {
    return execute([&] {
        if (element_size != 1 && element_size != 2 && element_size != 4) return 1;
        if (count > std::numeric_limits<std::size_t>::max() / element_size) return 1;
        void* pointer = host_pointer(destination, count * element_size);
        if (element_size == 1) std::memset(pointer, value, count);
        else if (element_size == 2) std::fill_n(static_cast<std::uint16_t*>(pointer), count, value);
        else std::fill_n(static_cast<std::uint32_t*>(pointer), count, value);
        return 0;
    });
}

} // namespace fake_gpu::kernel

#else

namespace fake_gpu::kernel {

bool available() { return false; }
int ptx_compatible(const std::string&, unsigned) { return 801; }
int cubin_compatible(const std::string&, unsigned) { return 801; }
int load(const std::string&, void** output) {
    if (!output) return 1;
    *output = nullptr;
    return 801;
}
int function(void*, const char* name, void** output) {
    if (!name || !output) return 1;
    *output = nullptr;
    return 400;
}
int global(void*, const char* name, void** pointer, std::size_t* bytes) {
    return !name || (!pointer && !bytes) ? 1 : 400;
}
int unload(void*) { return 400; }
int launch(void*, const unsigned[3], const unsigned[3], std::size_t, void**, void**) { return 400; }
std::string function_name(void*) { return {}; }
int copy(void* destination, const void* source, std::size_t bytes) {
    std::memcpy(destination, source, bytes);
    return 0;
}
int fill(void* destination, unsigned value, std::size_t count, unsigned element_size) {
    auto* pointer = static_cast<unsigned char*>(destination);
    for (std::size_t i = 0; i < count; ++i)
        std::memcpy(pointer + i * element_size, &value, element_size);
    return 0;
}

} // namespace fake_gpu::kernel

#endif
