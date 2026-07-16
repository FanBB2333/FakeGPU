add_library(fake_gpu_cuda_runtime OBJECT
    cuda_stubs.cpp
)

target_include_directories(fake_gpu_cuda_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_cuda_runtime PRIVATE ../core)
set_property(TARGET fake_gpu_cuda_runtime PROPERTY POSITION_INDEPENDENT_CODE ON)

# Keep CUDA Driver and Runtime symbols in separate object libraries.  In
# passthrough/hybrid mode libcuda is preloaded while the real libcudart stays in
# use.  Exporting cudaMalloc/cudaLaunchKernel from libcuda would otherwise
# interpose the real Runtime API and turn real kernel launches into simulation
# stubs.
add_library(fake_gpu_cuda_driver OBJECT
    cuda_driver_stubs.cpp
)

target_include_directories(fake_gpu_cuda_driver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_cuda_driver PRIVATE ../core)
set_property(TARGET fake_gpu_cuda_driver PROPERTY POSITION_INDEPENDENT_CODE ON)

# CUDA Runtime API object library
add_library(fake_gpu_cudart OBJECT
    cudart_stubs.cpp
)

target_include_directories(fake_gpu_cudart PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_cudart PRIVATE ../core)
set_property(TARGET fake_gpu_cudart PROPERTY POSITION_INDEPENDENT_CODE ON)
