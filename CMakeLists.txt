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

# macOS: always build with AppleClang.
# Homebrew/other GCC toolchains are not compatible with Apple SDK headers.
if(APPLE)
    if(EXISTS "/usr/bin/clang" AND EXISTS "/usr/bin/clang++")
        set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE FILEPATH "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE FILEPATH "C++ compiler" FORCE)
    else()
        find_program(_FAKEGPU_CLANG clang)
        find_program(_FAKEGPU_CLANGXX clang++)
        if(NOT _FAKEGPU_CLANG OR NOT _FAKEGPU_CLANGXX)
            message(FATAL_ERROR "clang/clang++ not found. Install Xcode Command Line Tools to build FakeGPU on macOS.")
        endif()
        set(CMAKE_C_COMPILER "${_FAKEGPU_CLANG}" CACHE FILEPATH "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "${_FAKEGPU_CLANGXX}" CACHE FILEPATH "C++ compiler" FORCE)
    endif()
endif()

project(FakeGPU VERSION "${FAKEGPU_VERSION}")

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
    # On older glibc targets such as manylinux2014, shm_open/shm_unlink live in librt.
    set(FAKEGPU_POSIX_SHM_LIB rt)
else()
    set(FAKEGPU_POSIX_SHM_LIB "")
endif()

function(fakegpu_link_common_system_libs target_name)
    if(CMAKE_DL_LIBS)
        target_link_libraries(${target_name} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    target_link_libraries(${target_name} PRIVATE Threads::Threads)
    if(FAKEGPU_POSIX_SHM_LIB)
        target_link_libraries(${target_name} PRIVATE ${FAKEGPU_POSIX_SHM_LIB})
    endif()
endfunction()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable/disable debug logging
option(ENABLE_FAKEGPU_LOGGING "Enable debug logging output" OFF)

if(ENABLE_FAKEGPU_LOGGING)
    add_compile_definitions(ENABLE_FAKEGPU_LOGGING)
    message(STATUS "FakeGPU logging: ENABLED")
else()
    message(STATUS "FakeGPU logging: DISABLED")
endif()

# Option to enable CPU-based compute for selected operators (e.g. cuBLAS GEMM)
option(ENABLE_FAKEGPU_CPU_SIMULATION "Execute supported FakeGPU operators on CPU for improved correctness" ON)

if(ENABLE_FAKEGPU_CPU_SIMULATION)
    message(STATUS "FakeGPU CPU simulation: ENABLED")
else()
    message(STATUS "FakeGPU CPU simulation: DISABLED")
endif()

# Embed YAML GPU profiles at configure time so defaults are available without runtime files
set(FAKEGPU_PROFILE_DIR "${CMAKE_SOURCE_DIR}/profiles")
file(
    GLOB_RECURSE FAKEGPU_PROFILE_FILES
    CONFIGURE_DEPENDS
    "${FAKEGPU_PROFILE_DIR}/*.yaml"
)
list(SORT FAKEGPU_PROFILE_FILES)
set(FAKEGPU_GENERATED_PROFILE_HEADER "${CMAKE_BINARY_DIR}/generated_profiles.hpp")

file(WRITE ${FAKEGPU_GENERATED_PROFILE_HEADER}
"// Auto-generated from profiles/**/*.yaml - do not edit by hand\n"
"#pragma once\n"
"#include <string>\n"
"#include <vector>\n"
"namespace fake_gpu {\n"
"struct ProfileYamlBlob {\n"
"    const char* filename;\n"
"    // Path below profiles/, <architecture>/<segment>/<id>.yaml.\n"
"    const char* relative_path;\n"
"    const char* yaml;\n"
"};\n"
"inline const std::vector<ProfileYamlBlob>& builtin_profile_yamls() {\n"
"    static const std::vector<ProfileYamlBlob> kProfiles = {\n"
)
set(_profile_idx 0)
foreach(_profile_path ${FAKEGPU_PROFILE_FILES})
    file(READ ${_profile_path} _profile_content)
    get_filename_component(_profile_name ${_profile_path} NAME)
    file(RELATIVE_PATH _profile_relative_path ${FAKEGPU_PROFILE_DIR} ${_profile_path})
    file(APPEND ${FAKEGPU_GENERATED_PROFILE_HEADER}
"        {\"${_profile_name}\", \"${_profile_relative_path}\", R\"fakegpu_${_profile_idx}(\n${_profile_content})fakegpu_${_profile_idx}\"},\n"
)
    math(EXPR _profile_idx "${_profile_idx} + 1")
endforeach()
file(APPEND ${FAKEGPU_GENERATED_PROFILE_HEADER}
"    };\n"
"    return kProfiles;\n"
"}\n"
"} // namespace fake_gpu\n"
)

# Source directories
add_subdirectory(src/core)
add_subdirectory(src/nvml)
add_subdirectory(src/cuda)
add_subdirectory(src/monitor)
add_subdirectory(src/distributed)
add_subdirectory(src/nccl)
include(cmake/PtxBackend.cmake)

# Main Shared Library for NVML
# We link the OBJECT libraries from subdirectories to create one single shared library
add_library(fake_gpu SHARED
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_nvml>
    $<TARGET_OBJECTS:fake_gpu_monitor>
    $<TARGET_OBJECTS:fake_gpu_distributed>
)

fakegpu_link_common_system_libs(fake_gpu)

# CUDA Driver Library (libcuda.so.1)
# This library intercepts CUDA Driver API calls
add_library(fake_cuda SHARED
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_cuda_driver>
    $<TARGET_OBJECTS:fake_gpu_monitor>
    $<TARGET_OBJECTS:fake_gpu_distributed>
)
fakegpu_link_common_system_libs(fake_cuda)
target_link_libraries(fake_cuda PRIVATE fake_gpu_kernel)

# CUDA Runtime Library (libcudart.so.12)
# This library intercepts CUDA Runtime API calls and uses fake_cuda Driver API internally
# Keep the Runtime implementation here, separate from libcuda's exports.
add_library(fake_cudart SHARED
    $<TARGET_OBJECTS:fake_gpu_cudart>
    $<TARGET_OBJECTS:fake_gpu_cuda_driver>
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_monitor>
    $<TARGET_OBJECTS:fake_gpu_distributed>
)
fakegpu_link_common_system_libs(fake_cudart)
target_link_libraries(fake_cudart PRIVATE fake_gpu_kernel)

# cuBLAS Library (libcublas.so.12)
# This library intercepts cuBLAS/cuBLASLt calls; supported ops can run on CPU for correctness
# (see ENABLE_FAKEGPU_CPU_SIMULATION).
add_library(fake_cublas SHARED
    src/cublas/cublas_stubs.cpp
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_monitor>
    $<TARGET_OBJECTS:fake_gpu_distributed>
)
fakegpu_link_common_system_libs(fake_cublas)
target_include_directories(fake_cublas PRIVATE src/cublas src/core)
target_compile_definitions(fake_cublas PRIVATE FAKEGPU_CPU_SIMULATION=$<BOOL:${ENABLE_FAKEGPU_CPU_SIMULATION}>)
target_link_libraries(fake_cublas PRIVATE m)

add_library(fake_nccl SHARED
    $<TARGET_OBJECTS:fake_gpu_nccl>
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_distributed>
)
fakegpu_link_common_system_libs(fake_nccl)

# Library names and SONAMEs that CUDA, NVML and NCCL users look up.
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
set_target_properties(fake_nccl PROPERTIES
    OUTPUT_NAME "nccl"
    VERSION "${FAKEGPU_VERSION}"
    SOVERSION "2"
)

# Platform specific flags
if(APPLE)
    # macOS: build dylibs with CUDA/NVML-compatible names so DYLD_* search works.
    foreach(_fakegpu_library fake_gpu fake_cuda fake_cudart fake_cublas fake_nccl)
        target_link_options(${_fakegpu_library} PRIVATE "-undefined" "dynamic_lookup")
    endforeach()
    add_custom_command(TARGET fake_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE_NAME:fake_gpu>
            ${CMAKE_BINARY_DIR}/libfake_gpu.dylib
        COMMENT "Creating libfake_gpu.dylib symlink for backward compatibility"
    )
    # nvidia-ml-py (pynvml) loads CDLL("libnvidia-ml.so.1") on every
    # non-Windows platform; dyld finds that leaf name on DYLD_LIBRARY_PATH.
    add_custom_command(TARGET fake_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_SONAME_FILE_NAME:fake_gpu>
            ${CMAKE_BINARY_DIR}/libnvidia-ml.so.1
        COMMENT "Creating libnvidia-ml.so.1 symlink for pynvml"
    )
else()
    # Create symlink for backward compatibility with libfake_gpu.so name
    add_custom_command(TARGET fake_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libnvidia-ml.so.1
            ${CMAKE_BINARY_DIR}/libfake_gpu.so
        COMMENT "Creating libfake_gpu.so symlink for backward compatibility"
    )
endif()

add_executable(fakegpu_coordinator
    src/distributed/coordinator_main.cpp
    $<TARGET_OBJECTS:fake_gpu_core>
    $<TARGET_OBJECTS:fake_gpu_distributed>
)
target_include_directories(fakegpu_coordinator PRIVATE src/distributed)
fakegpu_link_common_system_libs(fakegpu_coordinator)
set_target_properties(fakegpu_coordinator PROPERTIES OUTPUT_NAME "fakegpu-coordinator")
