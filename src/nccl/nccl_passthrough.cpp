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
        paths.emplace_back(std::string(conda_prefix) + "/lib/libnccl.so.2");
    }

    paths.emplace_back("/usr/lib/x86_64-linux-gnu/libnccl.so.2");
    paths.emplace_back("/usr/lib64/libnccl.so.2");
    paths.emplace_back("/usr/local/cuda/lib64/libnccl.so.2");
    paths.emplace_back("/usr/local/nvidia/lib64/libnccl.so.2");
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

}  // namespace

RealNcclLoader& RealNcclLoader::instance() {
    static RealNcclLoader loader;
    return loader;
}

bool RealNcclLoader::load_handle(std::string& error) {
    LoaderState& state = loader_state();
    const std::string self_path = current_library_path();

    for (const std::string& candidate : candidate_paths()) {
        if (candidate.empty()) {
            continue;
        }
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

    error = "failed to load a real libnccl.so.2; set FAKEGPU_REAL_NCCL_PATH";
    return false;
}

bool RealNcclLoader::resolve_symbols(std::string& error) {
    LoaderState& state = loader_state();
    state.get_version =
        reinterpret_cast<ncclGetVersion_fn>(resolve_symbol(state.handle, "ncclGetVersion", error));
    if (!state.get_version) {
        return false;
    }
    state.get_unique_id =
        reinterpret_cast<ncclGetUniqueId_fn>(resolve_symbol(state.handle, "ncclGetUniqueId", error));
    if (!state.get_unique_id) {
        return false;
    }
    state.comm_init_rank =
        reinterpret_cast<ncclCommInitRank_fn>(resolve_symbol(state.handle, "ncclCommInitRank", error));
    if (!state.comm_init_rank) {
        return false;
    }
    state.comm_destroy =
        reinterpret_cast<ncclCommDestroy_fn>(resolve_symbol(state.handle, "ncclCommDestroy", error));
    if (!state.comm_destroy) {
        return false;
    }
    state.comm_abort =
        reinterpret_cast<ncclCommAbort_fn>(::dlsym(state.handle, "ncclCommAbort"));
    ::dlerror();
    state.get_error_string =
        reinterpret_cast<ncclGetErrorString_fn>(resolve_symbol(state.handle, "ncclGetErrorString", error));
    if (!state.get_error_string) {
        return false;
    }
    state.all_reduce =
        reinterpret_cast<ncclAllReduce_fn>(resolve_symbol(state.handle, "ncclAllReduce", error));
    if (!state.all_reduce) {
        return false;
    }
    state.reduce =
        reinterpret_cast<ncclReduce_fn>(resolve_symbol(state.handle, "ncclReduce", error));
    if (!state.reduce) {
        return false;
    }
    state.broadcast =
        reinterpret_cast<ncclBroadcast_fn>(resolve_symbol(state.handle, "ncclBroadcast", error));
    if (!state.broadcast) {
        return false;
    }
    state.all_gather =
        reinterpret_cast<ncclAllGather_fn>(resolve_symbol(state.handle, "ncclAllGather", error));
    if (!state.all_gather) {
        return false;
    }
    ::dlerror();
    state.all_to_all =
        reinterpret_cast<ncclAlltoAll_fn>(::dlsym(state.handle, "ncclAlltoAll"));
    ::dlerror();
    state.reduce_scatter =
        reinterpret_cast<ncclReduceScatter_fn>(resolve_symbol(state.handle, "ncclReduceScatter", error));
    if (!state.reduce_scatter) {
        return false;
    }
    return true;
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

ncclResult_t RealNcclLoader::get_version(int* version, std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.get_version) {
        error = "real ncclGetVersion is unavailable";
        return ncclSystemError;
    }
    return state.get_version(version);
}

ncclResult_t RealNcclLoader::get_unique_id(ncclUniqueId* unique_id, std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.get_unique_id) {
        error = "real ncclGetUniqueId is unavailable";
        return ncclSystemError;
    }
    return state.get_unique_id(unique_id);
}

ncclResult_t RealNcclLoader::comm_init_rank(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId comm_id,
    int rank,
    std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.comm_init_rank) {
        error = "real ncclCommInitRank is unavailable";
        return ncclSystemError;
    }
    return state.comm_init_rank(comm, nranks, comm_id, rank);
}

ncclResult_t RealNcclLoader::comm_destroy(ncclComm_t comm, std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.comm_destroy) {
        error = "real ncclCommDestroy is unavailable";
        return ncclSystemError;
    }
    return state.comm_destroy(comm);
}

ncclResult_t RealNcclLoader::comm_abort(ncclComm_t comm, std::string& error) const {
    const LoaderState& state = loader_state();
    if (state.comm_abort) {
        return state.comm_abort(comm);
    }
    return comm_destroy(comm, error);
}

const char* RealNcclLoader::get_error_string(ncclResult_t result, std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.get_error_string) {
        error = "real ncclGetErrorString is unavailable";
        return nullptr;
    }
    return state.get_error_string(result);
}

ncclResult_t RealNcclLoader::all_reduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.all_reduce) {
        error = "real ncclAllReduce is unavailable";
        return ncclSystemError;
    }
    return state.all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
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
    std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.reduce) {
        error = "real ncclReduce is unavailable";
        return ncclSystemError;
    }
    return state.reduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t RealNcclLoader::broadcast(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.broadcast) {
        error = "real ncclBroadcast is unavailable";
        return ncclSystemError;
    }
    return state.broadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t RealNcclLoader::all_gather(
    const void* sendbuff,
    void* recvbuff,
    std::size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.all_gather) {
        error = "real ncclAllGather is unavailable";
        return ncclSystemError;
    }
    return state.all_gather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
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
    std::string& error) const {
    const LoaderState& state = loader_state();
    if (!state.reduce_scatter) {
        error = "real ncclReduceScatter is unavailable";
        return ncclSystemError;
    }
    return state.reduce_scatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

}  // namespace fake_gpu::nccl
