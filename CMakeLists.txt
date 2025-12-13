cmake_minimum_required(VERSION 3.14)
project(FakeGPU VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source directories
add_subdirectory(src/core)
add_subdirectory(src/nvml)
add_subdirectory(src/cuda)
add_subdirectory(src/monitor)

# Main Shared Library
# We link the OBJECT libraries from subdirectories to create one single shared library
add_library(fake_gpu SHARED
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_nvml>
    $<TARGET_OBJECTS:fake_gpu_cuda>
    $<TARGET_OBJECTS:fake_gpu_monitor>
)

# Link dl library for dlopen/dlsym interception
target_link_libraries(fake_gpu PRIVATE ${CMAKE_DL_LIBS})

# Platform specific flags
if(APPLE)
    # macOS specific linker flags for interposition if needed
    target_link_options(fake_gpu PRIVATE "-undefined" "dynamic_lookup")
    set_target_properties(fake_gpu PROPERTIES SUFFIX ".dylib")
else()
    set_target_properties(fake_gpu PROPERTIES SUFFIX ".so")
    # Create symlink for libnvidia-ml.so.1 so PyTorch can find it via LD_LIBRARY_PATH
    add_custom_command(TARGET fake_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE_NAME:fake_gpu>
            ${CMAKE_BINARY_DIR}/libnvidia-ml.so.1
        COMMENT "Creating libnvidia-ml.so.1 symlink"
    )
endif()
