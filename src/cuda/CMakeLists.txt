add_library(fake_gpu_cuda OBJECT
    cuda_stubs.cpp
    cuda_driver_stubs.cpp
)

target_include_directories(fake_gpu_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_cuda PRIVATE ../core)
set_property(TARGET fake_gpu_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

# CUDA Runtime API object library
add_library(fake_gpu_cudart OBJECT
    cudart_stubs.cpp
)

target_include_directories(fake_gpu_cudart PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_cudart PRIVATE ../core)
set_property(TARGET fake_gpu_cudart PROPERTY POSITION_INDEPENDENT_CODE ON)
