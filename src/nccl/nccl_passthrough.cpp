#include "nccl_passthrough.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <mutex>
#include <string>
#include <vector>

namespace fake_gpu::nccl {

namespace {

using ncclGetVersion_fn = ncclResult_t (*)(int*);
using ncclGetUniqueId_fn = ncclResult_t (*)(ncclUniqueId*);
using ncclCommInitRank_fn = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int);
using ncclCommDestroy_fn = ncclResult_t (*)(ncclComm_t);
using ncclCommAbort_fn = ncclResult_t (*)(ncclComm_t);
using ncclGetErrorString_fn = const char* (*)(ncclResult_t);
using ncclAllReduce_fn =
    ncclResult_t (*)(const void*, void*, std::size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
using ncclReduce_fn =
    ncclResult_t (*)(const void*, void*, std::size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t);
using ncclBroadcast_fn =
    ncclResult_t (*)(const void*, void*, std::size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
using ncclAllGather_fn =
    ncclResult_t (*)(const void*, void*, std::size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
using ncclAlltoAll_fn =
    ncclResult_t (*)(const void*, void*, std::size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
using ncclReduceScatter_fn =
    ncclResult_t (*)(const void*, void*, std::size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);

struct LoaderState {
    std::mutex mutex;
    bool initialized = false;
    void* handle = nullptr;
    std::string loaded_path;
    ncclGetVersion_fn get_version = nullptr;
    ncclGetUniqueId_fn get_unique_id = nullptr;
    ncclCommInitRank_fn comm_init_rank = nullptr;
    ncclCommDestroy_fn comm_destroy = nullptr;
    ncclCommAbort_fn comm_abort = nullptr;
    ncclGetErrorString_fn get_error_string = nullptr;
    ncclAllReduce_fn all_reduce = nullptr;
    ncclReduce_fn reduce = nullptr;
    ncclBroadcast_fn broadcast = nullptr;
    ncclAllGather_fn all_gather = nullptr;
    ncclAlltoAll_fn all_to_all = nullptr;
    ncclReduceScatter_fn reduce_scatter = nullptr;
};

LoaderState& loader_state() {
    static LoaderState state;
    return state;
}

std::string current_library_path() {
    Dl_info info {};
    if (::dladdr(reinterpret_cast<void*>(&RealNcclLoader::instance), &info) != 0 &&
        info.dli_fname != nullptr) {
        char resolved[PATH_MAX] = {};
        if (::realpath(info.dli_fname, resolved) != nullptr) {
            return resolved;
        }
        return info.dli_fname;
    }
    return {};
}

bool paths_equal(const std::string& lhs, const std::string& rhs) {
    return !lhs.empty() && !rhs.empty() && lhs == rhs;
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

std::vector<std::string> candidate_paths() {
    std::vector<std::string> paths;

    const std::string env_path = env_or_empty("FAKEGPU_REAL_NCCL_PATH");
    if (!env_path.empty()) {
        paths.push_back(env_path);
    }

    const char* conda_prefix = std::getenv("CONDA_PREFIX");
    if (conda_prefix && *conda_prefix) {
#ifdef __APPLE__
        paths.emplace_back(std::string(conda_prefix) + "/lib/libnccl.dylib");
        paths.emplace_back(std::string(conda_prefix) + "/lib/libnccl.2.dylib");
#else
        paths.emplace_back(std::string(conda_prefix) + "/lib/libnccl.so.2");
#endif
    }

#ifdef __APPLE__
    paths.emplace_back("/opt/homebrew/lib/libnccl.dylib");
    paths.emplace_back("/opt/homebrew/lib/libnccl.2.dylib");
    paths.emplace_back("/usr/local/lib/libnccl.dylib");
    paths.emplace_back("/usr/local/lib/libnccl.2.dylib");
#else
    paths.emplace_back("/usr/lib/x86_64-linux-gnu/libnccl.so.2");
    paths.emplace_back("/usr/lib64/libnccl.so.2");
    paths.emplace_back("/usr/local/cuda/lib64/libnccl.so.2");
    paths.emplace_back("/usr/local/nvidia/lib64/libnccl.so.2");
#endif
    return paths;
}

void* resolve_symbol(void* handle, const char* name, std::string& error) {
    ::dlerror();
    void* symbol = ::dlsym(handle, name);
    const char* dl_error = ::dlerror();
    if (dl_error != nullptr) {
        error = std::string("failed to resolve ") + name + ": " + dl_error;
        return nullptr;
    }
    return symbol;
}

template <typename Fn>
bool resolve_required(void* handle, const char* name, Fn& out, std::string& error) {
    out = reinterpret_cast<Fn>(resolve_symbol(handle, name, error));
    return out != nullptr;
}

}  // namespace

RealNcclLoader& RealNcclLoader::instance() {
    static RealNcclLoader loader;
    return loader;
}

bool RealNcclLoader::load_handle(std::string& error) {
    LoaderState& state = loader_state();
    const std::string self_path = current_library_path();

    for (const std::string& candidate : candidate_paths()) {
        char resolved[PATH_MAX] = {};
        std::string resolved_path = candidate;
        if (::realpath(candidate.c_str(), resolved) != nullptr) {
            resolved_path = resolved;
        }
        if (paths_equal(self_path, resolved_path)) {
            continue;
        }

        void* handle = ::dlopen(resolved_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            continue;
        }

        state.handle = handle;
        state.loaded_path = resolved_path;
        return true;
    }

    error = "failed to load a real NCCL library; set FAKEGPU_REAL_NCCL_PATH";
    return false;
}

bool RealNcclLoader::resolve_symbols(std::string& error) {
    LoaderState& state = loader_state();
    void* handle = state.handle;
    if (!resolve_required(handle, "ncclGetVersion", state.get_version, error) ||
        !resolve_required(handle, "ncclGetUniqueId", state.get_unique_id, error) ||
        !resolve_required(handle, "ncclCommInitRank", state.comm_init_rank, error) ||
        !resolve_required(handle, "ncclCommDestroy", state.comm_destroy, error)) {
        return false;
    }
    state.comm_abort =
        reinterpret_cast<ncclCommAbort_fn>(::dlsym(handle, "ncclCommAbort"));
    ::dlerror();
    if (!resolve_required(handle, "ncclGetErrorString", state.get_error_string, error) ||
        !resolve_required(handle, "ncclAllReduce", state.all_reduce, error) ||
        !resolve_required(handle, "ncclReduce", state.reduce, error) ||
        !resolve_required(handle, "ncclBroadcast", state.broadcast, error) ||
        !resolve_required(handle, "ncclAllGather", state.all_gather, error)) {
        return false;
    }
    ::dlerror();
    state.all_to_all =
        reinterpret_cast<ncclAlltoAll_fn>(::dlsym(handle, "ncclAlltoAll"));
    ::dlerror();
    return resolve_required(handle, "ncclReduceScatter", state.reduce_scatter, error);
}

bool RealNcclLoader::initialize(std::string& error) {
    LoaderState& state = loader_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.initialized) {
        if (!state.handle) {
            error = "real NCCL loader initialization failed earlier";
            return false;
        }
        return true;
    }

    state.initialized = true;
    if (!load_handle(error)) {
        return false;
    }
    if (!resolve_symbols(error)) {
        ::dlclose(state.handle);
        state.handle = nullptr;
        state.loaded_path.clear();
        return false;
    }
    return true;
}

bool RealNcclLoader::available() const {
    return loader_state().handle != nullptr;
}

const std::string& RealNcclLoader::loaded_path() const {
    return loader_state().loaded_path;
}

ncclResult_t RealNcclLoader::get_version(int* version, std::string& /*error*/) const {
    return loader_state().get_version(version);
}

ncclResult_t RealNcclLoader::get_unique_id(ncclUniqueId* unique_id, std::string& /*error*/) const {
    return loader_state().get_unique_id(unique_id);
}

ncclResult_t RealNcclLoader::comm_init_rank(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId comm_id,
    int rank,
    std::string& /*error*/) const {
    return loader_state().comm_init_rank(comm, nranks, comm_id, rank);
}

ncclResult_t RealNcclLoader::comm_destroy(ncclComm_t comm, std::string& /*error*/) const {
    return loader_state().comm_destroy(comm);
}

ncclResult_t RealNcclLoader::comm_abort(ncclComm_t comm, std::string& error) const {
    const LoaderState& state = loader_state();
    if (state.comm_abort) {
        return state.comm_abort(comm);
    }
    return comm_destroy(comm, error);
}

const char* RealNcclLoader::get_error_string(ncclResult_t result, std::string& /*error*/) const {
    return loader_state().get_error_string(result);
}

ncclResult_t RealNcclLoader::all_reduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& /*error*/) const {
    return loader_state().all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t RealNcclLoader::reduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& /*error*/) const {
    return loader_state().reduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t RealNcclLoader::broadcast(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& /*error*/) const {
    return loader_state().broadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t RealNcclLoader::all_gather(
    const void* sendbuff,
    void* recvbuff,
    std::size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& /*error*/) const {
    return loader_state().all_gather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t RealNcclLoader::all_to_all(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.all_to_all) {
        error = "real ncclAlltoAll is unavailable";
        return ncclSystemError;
    }
    return state.all_to_all(sendbuff, recvbuff, count, datatype, comm, stream);
}

ncclResult_t RealNcclLoader::reduce_scatter(
    const void* sendbuff,
    void* recvbuff,
    std::size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& /*error*/) const {
    return loader_state().reduce_scatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

}  // namespace fake_gpu::nccl
