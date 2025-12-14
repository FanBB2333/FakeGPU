cmake_minimum_required(VERSION 3.14)
project(FakeGPU VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source directories
add_subdirectory(src/core)
add_subdirectory(src/nvml)
add_subdirectory(src/cuda)
add_subdirectory(src/monitor)

# Main Shared Library for NVML
# We link the OBJECT libraries from subdirectories to create one single shared library
add_library(fake_gpu SHARED
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_nvml>
    $<TARGET_OBJECTS:fake_gpu_cuda>
    $<TARGET_OBJECTS:fake_gpu_monitor>
)

# Link dl library for dlopen/dlsym interception
target_link_libraries(fake_gpu PRIVATE ${CMAKE_DL_LIBS})

# CUDA Driver Library (libcuda.so.1)
# This library intercepts CUDA Driver API calls
add_library(fake_cuda SHARED
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_cuda>
    $<TARGET_OBJECTS:fake_gpu_monitor>
)
target_link_libraries(fake_cuda PRIVATE ${CMAKE_DL_LIBS})

# Platform specific flags
if(APPLE)
    # macOS specific linker flags for interposition if needed
    target_link_options(fake_gpu PRIVATE "-undefined" "dynamic_lookup")
    set_target_properties(fake_gpu PROPERTIES SUFFIX ".dylib")
    target_link_options(fake_cuda PRIVATE "-undefined" "dynamic_lookup")
    set_target_properties(fake_cuda PROPERTIES SUFFIX ".dylib")
else()
    # Set OUTPUT_NAME to libnvidia-ml.so.1 so CMake sets the correct SONAME
    set_target_properties(fake_gpu PROPERTIES
        OUTPUT_NAME "nvidia-ml"
        VERSION "1.0.0"
        SOVERSION "1"
    )
    # Create symlink for backward compatibility with libfake_gpu.so name
    add_custom_command(TARGET fake_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libnvidia-ml.so.1
            ${CMAKE_BINARY_DIR}/libfake_gpu.so
        COMMENT "Creating libfake_gpu.so symlink for backward compatibility"
    )

    # Set OUTPUT_NAME to libcuda.so.1 for CUDA Driver API interception
    set_target_properties(fake_cuda PROPERTIES
        OUTPUT_NAME "cuda"
        VERSION "1.0.0"
        SOVERSION "1"
    )
endif()
