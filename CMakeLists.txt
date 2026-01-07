cmake_minimum_required(VERSION 3.14)

# Keep the project/package version in one place: fakegpu/_version.py
set(FAKEGPU_VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/fakegpu/_version.py")
if(NOT EXISTS "${FAKEGPU_VERSION_FILE}")
    message(FATAL_ERROR "Version file not found: ${FAKEGPU_VERSION_FILE}")
endif()
file(READ "${FAKEGPU_VERSION_FILE}" FAKEGPU_VERSION_FILE_CONTENTS)
string(REGEX MATCH "__version__ *=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" FAKEGPU_VERSION_MATCH "${FAKEGPU_VERSION_FILE_CONTENTS}")
if(NOT FAKEGPU_VERSION_MATCH)
    message(FATAL_ERROR "Could not parse __version__ from: ${FAKEGPU_VERSION_FILE}")
endif()
set(FAKEGPU_VERSION "${CMAKE_MATCH_1}")

project(FakeGPU VERSION "${FAKEGPU_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE AND (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR
        "Building FakeGPU with GCC on macOS is not supported due to Apple SDK header incompatibilities.\n"
        "Use AppleClang (default), or reconfigure in a clean build directory with:\n"
        "  cmake -S . -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++\n"
    )
endif()

# Option to enable/disable debug logging
option(ENABLE_FAKEGPU_LOGGING "Enable debug logging output" OFF)

if(ENABLE_FAKEGPU_LOGGING)
    add_compile_definitions(ENABLE_FAKEGPU_LOGGING)
    message(STATUS "FakeGPU logging: ENABLED")
else()
    message(STATUS "FakeGPU logging: DISABLED")
endif()

# Option to enable CPU-based compute for selected operators (e.g. cuBLAS GEMM)
option(ENABLE_FAKEGPU_CPU_SIMULATION "Execute supported FakeGPU operators on CPU for improved correctness" OFF)

if(ENABLE_FAKEGPU_CPU_SIMULATION)
    message(STATUS "FakeGPU CPU simulation: ENABLED")
else()
    message(STATUS "FakeGPU CPU simulation: DISABLED")
endif()

# Embed YAML GPU profiles at configure time so defaults are available without runtime files
set(FAKEGPU_PROFILE_DIR "${CMAKE_SOURCE_DIR}/profiles")
file(GLOB FAKEGPU_PROFILE_FILES "${FAKEGPU_PROFILE_DIR}/*.yaml")
set(FAKEGPU_GENERATED_PROFILE_HEADER "${CMAKE_BINARY_DIR}/generated_profiles.hpp")

if(FAKEGPU_PROFILE_FILES)
    file(WRITE ${FAKEGPU_GENERATED_PROFILE_HEADER}
"// Auto-generated from profiles/*.yaml - do not edit by hand\n"
"#pragma once\n"
"#include <string>\n"
"#include <vector>\n"
"namespace fake_gpu {\n"
"struct ProfileYamlBlob {\n"
"    const char* filename;\n"
"    const char* yaml;\n"
"};\n"
"inline const std::vector<ProfileYamlBlob>& builtin_profile_yamls() {\n"
"    static const std::vector<ProfileYamlBlob> kProfiles = {\n"
)
    set(_profile_idx 0)
    foreach(_profile_path ${FAKEGPU_PROFILE_FILES})
        file(READ ${_profile_path} _profile_content)
        get_filename_component(_profile_name ${_profile_path} NAME)
        file(APPEND ${FAKEGPU_GENERATED_PROFILE_HEADER}
"        {\"${_profile_name}\", R\"fakegpu_${_profile_idx}(\n${_profile_content})fakegpu_${_profile_idx}\"},\n"
)
        math(EXPR _profile_idx "${_profile_idx} + 1")
    endforeach()
    file(APPEND ${FAKEGPU_GENERATED_PROFILE_HEADER}
"    };\n"
"    return kProfiles;\n"
"}\n"
"} // namespace fake_gpu\n"
)
else()
    file(WRITE ${FAKEGPU_GENERATED_PROFILE_HEADER}
"// Auto-generated: no YAML profiles were found\n"
"#pragma once\n"
"#include <string>\n"
"#include <vector>\n"
"namespace fake_gpu {\n"
"struct ProfileYamlBlob {\n"
"    const char* filename;\n"
"    const char* yaml;\n"
"};\n"
"inline const std::vector<ProfileYamlBlob>& builtin_profile_yamls() {\n"
"    static const std::vector<ProfileYamlBlob> kProfiles;\n"
"    return kProfiles;\n"
"}\n"
"} // namespace fake_gpu\n"
)
endif()

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

# CUDA Runtime Library (libcudart.so.12)
# This library intercepts CUDA Runtime API calls and uses fake_cuda Driver API internally
# We only include Driver API stubs, not Runtime API stubs from cuda_stubs.cpp
add_library(fake_cudart SHARED
    src/cuda/cudart_stubs.cpp
    src/cuda/cuda_driver_stubs.cpp
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_monitor>
)
target_link_libraries(fake_cudart PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(fake_cudart PRIVATE src/cuda src/core)

# cuBLAS Library (libcublas.so.12)
# This library intercepts cuBLAS calls and returns random values for profiling/monitoring
add_library(fake_cublas SHARED
    src/cublas/cublas_stubs.cpp
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_monitor>
)
target_link_libraries(fake_cublas PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(fake_cublas PRIVATE src/cublas src/core)
target_compile_definitions(fake_cublas PRIVATE FAKEGPU_CPU_SIMULATION=$<BOOL:${ENABLE_FAKEGPU_CPU_SIMULATION}>)
target_link_libraries(fake_cublas PRIVATE m)

# Platform specific flags
if(APPLE)
    # macOS: build dylibs with CUDA/NVML-compatible names so DYLD_* search works.
    target_link_options(fake_gpu PRIVATE "-undefined" "dynamic_lookup")
    target_link_options(fake_cuda PRIVATE "-undefined" "dynamic_lookup")

    set_target_properties(fake_gpu PROPERTIES
        OUTPUT_NAME "nvidia-ml"
        VERSION "${FAKEGPU_VERSION}"
        SOVERSION "1"
    )

    set_target_properties(fake_cuda PROPERTIES
        OUTPUT_NAME "cuda"
        VERSION "${FAKEGPU_VERSION}"
        SOVERSION "1"
    )

    set_target_properties(fake_cudart PROPERTIES
        OUTPUT_NAME "cudart"
        VERSION "${FAKEGPU_VERSION}"
        SOVERSION "12"
    )

    set_target_properties(fake_cublas PROPERTIES
        OUTPUT_NAME "cublas"
        VERSION "${FAKEGPU_VERSION}"
        SOVERSION "12"
    )
else()
    # Set OUTPUT_NAME to libnvidia-ml.so.1 so CMake sets the correct SONAME
    set_target_properties(fake_gpu PROPERTIES
        OUTPUT_NAME "nvidia-ml"
        VERSION "${FAKEGPU_VERSION}"
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
        VERSION "${FAKEGPU_VERSION}"
        SOVERSION "1"
    )

    # Set OUTPUT_NAME to libcudart.so.12 for CUDA Runtime API interception
    set_target_properties(fake_cudart PROPERTIES
        OUTPUT_NAME "cudart"
        VERSION "${FAKEGPU_VERSION}"
        SOVERSION "12"
    )

    # Set OUTPUT_NAME to libcublas.so.12 for cuBLAS API interception
    set_target_properties(fake_cublas PROPERTIES
        OUTPUT_NAME "cublas"
        VERSION "${FAKEGPU_VERSION}"
        SOVERSION "12"
    )
endif()
