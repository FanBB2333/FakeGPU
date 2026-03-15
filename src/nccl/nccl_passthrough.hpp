#pragma once

#include "nccl_defs.hpp"

#include <string>

namespace fake_gpu::nccl {

class RealNcclLoader {
public:
    static RealNcclLoader& instance();

    bool initialize(std::string& error);
    bool available() const;
    const std::string& loaded_path() const;

    ncclResult_t get_version(int* version, std::string& error) const;
    ncclResult_t get_unique_id(ncclUniqueId* unique_id, std::string& error) const;
    ncclResult_t comm_init_rank(
        ncclComm_t* comm,
        int nranks,
        ncclUniqueId comm_id,
        int rank,
        std::string& error) const;
    ncclResult_t comm_destroy(ncclComm_t comm, std::string& error) const;
    ncclResult_t comm_abort(ncclComm_t comm, std::string& error) const;
    const char* get_error_string(ncclResult_t result, std::string& error) const;

    ncclResult_t all_reduce(
        const void* sendbuff,
        void* recvbuff,
        std::size_t count,
        ncclDataType_t datatype,
        ncclRedOp_t op,
        ncclComm_t comm,
        cudaStream_t stream,
        std::string& error) const;
    ncclResult_t reduce(
        const void* sendbuff,
        void* recvbuff,
        std::size_t count,
        ncclDataType_t datatype,
        ncclRedOp_t op,
        int root,
        ncclComm_t comm,
        cudaStream_t stream,
        std::string& error) const;
    ncclResult_t broadcast(
        const void* sendbuff,
        void* recvbuff,
        std::size_t count,
        ncclDataType_t datatype,
        int root,
        ncclComm_t comm,
        cudaStream_t stream,
        std::string& error) const;
    ncclResult_t all_gather(
        const void* sendbuff,
        void* recvbuff,
        std::size_t sendcount,
        ncclDataType_t datatype,
        ncclComm_t comm,
        cudaStream_t stream,
        std::string& error) const;
    ncclResult_t reduce_scatter(
        const void* sendbuff,
        void* recvbuff,
        std::size_t recvcount,
        ncclDataType_t datatype,
        ncclRedOp_t op,
        ncclComm_t comm,
        cudaStream_t stream,
        std::string& error) const;

private:
    RealNcclLoader() = default;

    bool resolve_symbols(std::string& error);
    bool load_handle(std::string& error);
};

}  // namespace fake_gpu::nccl
