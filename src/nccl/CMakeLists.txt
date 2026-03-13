add_library(fake_gpu_nccl OBJECT
    nccl_stubs.cpp
)

target_include_directories(fake_gpu_nccl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_nccl PRIVATE ../core ../distributed)
set_property(TARGET fake_gpu_nccl PROPERTY POSITION_INDEPENDENT_CODE ON)
