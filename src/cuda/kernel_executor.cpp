#include "kernel_executor.hpp"

#if FAKEGPU_PTX_EXECUTION

#include "global_state.hpp"
#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/ptx/parser.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace fake_gpu::kernel {
namespace {

struct FreeMemory {
    void operator()(void* pointer) const { std::free(pointer); }
};

struct Global {
    std::unique_ptr<void, FreeMemory> storage;
    std::size_t size;
    bool constant;
};

struct Module {
    vgpu::ptx::Module code;
    vgpu::DeviceProfile profile;
    int device;
    std::unordered_map<std::string, Global> globals;
    vgpu::exec::SymbolTable symbols;
    std::unordered_map<std::string, void*> functions;
};

struct Function {
    Module* module;
    const vgpu::ptx::EntryFn* entry;
};

struct DeviceMemory {
    explicit DeviceMemory(std::uint64_t capacity, unsigned device) : memory(capacity, device) {}
    vgpu::MemoryManager memory;
    std::vector<std::uint64_t> host_mappings;
};

std::mutex execution_mutex;
std::unordered_map<void*, std::unique_ptr<Module>> modules;
std::unordered_map<void*, Function> functions;
std::unordered_map<int, std::unique_ptr<DeviceMemory>> device_memories;
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
        std::fprintf(stderr, "[FakeGPU PTX] %s\n", error.what());
        return error_code(error.code());
    } catch (const std::bad_alloc&) {
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[FakeGPU PTX] %s\n", error.what());
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

void validate_target(const Module& module) {
    const auto& target = module.code.target;
    const auto start = target.find("sm_");
    if (start == std::string::npos) return;
    char* end = nullptr;
    const long requested = std::strtol(target.c_str() + start + 3, &end, 10);
    const int actual = module.profile.cc_major * 10 + module.profile.cc_minor;
    if (requested > actual || (*end == 'a' && requested != actual) ||
        (*end == 'f' && requested / 10 != actual / 10)) {
        throw vgpu::Error::make(vgpu::Err::PtxParse, "module targets ", target,
                               " but the selected profile has compute capability ",
                               module.profile.cc_major, ".", module.profile.cc_minor);
    }
}

void initialize_globals(Module& module) {
    for (const auto& declaration : module.code.globals) {
        void* pointer = nullptr;
        const auto alignment = std::max<std::size_t>(declaration.align, sizeof(void*));
        if (posix_memalign(&pointer, alignment, declaration.size) != 0) throw std::bad_alloc();
        Global global{std::unique_ptr<void, FreeMemory>(pointer), declaration.size, declaration.is_const};
        std::memset(pointer, 0, declaration.size);
        if (!declaration.init.empty()) {
            std::memcpy(pointer, declaration.init.data(), declaration.init.size());
        }
        module.symbols[declaration.name] = reinterpret_cast<std::uintptr_t>(pointer);
        module.globals.emplace(declaration.name, std::move(global));
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
        auto* destination = static_cast<unsigned char*>(module.globals.at(declaration.name).storage.get());
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

int pack_arguments(const vgpu::ptx::EntryFn& entry, void** arguments, void** extra,
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
    std::size_t offset = 0;
    for (std::size_t index = 0; index < entry.params.size(); ++index) {
        const auto& parameter = entry.params[index];
        const void* source = nullptr;
        if (extra) {
            const auto alignment = parameter.align ? parameter.align : parameter.ty.bytes();
            offset = (offset + alignment - 1) / alignment * alignment;
            if (offset > buffer_size || parameter.size > buffer_size - offset || !buffer) return 1;
            source = buffer + offset;
            offset += parameter.size;
        } else {
            if (!arguments || !arguments[index]) return 1;
            source = arguments[index];
        }
        packed.emplace_back(parameter.size);
        std::memcpy(packed.back().data(), source, parameter.size);
    }
    return 0;
}

} // namespace

bool available() { return true; }

int load(const std::string& ptx, void** output) {
    if (!output) return 1;
    *output = nullptr;
    return execute([&] {
        auto& state = GlobalState::instance();
        state.initialize();
        auto module = std::make_unique<Module>();
        module->device = state.get_current_device();
        module->profile = device_profile(state.get_device(module->device));
        module->code = vgpu::ptx::parse(ptx);
        validate_target(*module);
        initialize_globals(*module);
        void* handle = new_handle();
        Module* stored = module.get();
        modules.emplace(handle, std::move(module));
        for (const auto& entry : stored->code.entries) {
            functions.emplace(stored->functions.at(entry.name), Function{stored, &entry});
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
        if (pointer) *pointer = found->second.storage.get();
        if (bytes) *bytes = found->second.size;
        return 0;
    });
}

int unload(void* handle) {
    return execute([&] {
        const auto module = modules.find(handle);
        if (module == modules.end()) return 400;
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
        if (const int status = pack_arguments(*function.entry, arguments, extra, packed)) return status;

        // Device-side malloc storage survives launches. Host mappings follow
        // the current allocations, including frees and module unloads.
        auto& device_memory = device_memories[module.device];
        if (!device_memory) {
            device_memory = std::make_unique<DeviceMemory>(module.profile.vram_bytes, module.device);
        }
        auto& memory = device_memory->memory;
        for (const auto address : device_memory->host_mappings) memory.unmap_host(address);
        device_memory->host_mappings.clear();
        auto map_host = [&](void* pointer, std::size_t size, bool constant = false) {
            const auto address = reinterpret_cast<std::uintptr_t>(pointer);
            memory.map_host(address, pointer, size, constant);
            device_memory->host_mappings.push_back(address);
        };
        for (const auto& [pointer, size] : state.snapshot_allocations()) {
            map_host(pointer, size);
        }
        for (const auto& [module_handle, loaded] : modules) {
            if (loaded->device != module.device) continue;
            for (const auto& [name, global] : loaded->globals) {
                map_host(global.storage.get(), global.size, global.constant);
            }
        }
        vgpu::exec::LaunchConfig configuration;
        std::copy_n(grid, 3, configuration.grid.begin());
        std::copy_n(block, 3, configuration.block.begin());
        configuration.shared_bytes = shared_bytes;
        configuration.device_ordinal = module.device;
        configuration.device_count = state.get_device_count();
        vgpu::exec::launch(*function.entry, configuration, packed, memory, module.profile, &module.symbols);
        return 0;
    });
}

std::string function_name(void* handle) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    const auto found = functions.find(handle);
    return found == functions.end() ? std::string{} : found->second.entry->name;
}

} // namespace fake_gpu::kernel

#else

namespace fake_gpu::kernel {

bool available() { return false; }
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

} // namespace fake_gpu::kernel

#endif
