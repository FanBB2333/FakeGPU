# Only the PTX parser, interpreter and memory model are linked. The upstream
# CUDA shims, device catalog and machine-management runtime are not built.
set(_fakegpu_ptx_default OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_fakegpu_ptx_default ON)
endif()
option(ENABLE_FAKEGPU_PTX_EXECUTION "Execute custom PTX kernels on CPU (Linux)" ${_fakegpu_ptx_default})
if(ENABLE_FAKEGPU_PTX_EXECUTION AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "ENABLE_FAKEGPU_PTX_EXECUTION currently requires Linux")
endif()

add_library(fake_gpu_kernel STATIC "${PROJECT_SOURCE_DIR}/src/cuda/kernel_executor.cpp")
target_include_directories(fake_gpu_kernel PRIVATE "${PROJECT_SOURCE_DIR}/src/core")
set_target_properties(fake_gpu_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ENABLE_FAKEGPU_PTX_EXECUTION AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The interpreter uses C++20 constexpr std::variant (libstdc++ 12 or newer).
    include(CheckCXXSourceCompiles)
    set(CMAKE_CXX_STANDARD 20)
    check_cxx_source_compiles("
        #include <variant>
        #if __cpp_lib_variant < 202106L
        #error constexpr std::variant is required
        #endif
        int main() { return 0; }
    " FAKEGPU_HAS_CONSTEXPR_VARIANT)
    set(CMAKE_CXX_STANDARD 17)
    if(NOT FAKEGPU_HAS_CONSTEXPR_VARIANT)
        message(FATAL_ERROR "PTX execution needs a C++20 standard library with constexpr std::variant (for example GCC 12+). Select a newer CMAKE_CXX_COMPILER or set ENABLE_FAKEGPU_PTX_EXECUTION=OFF.")
    endif()
    include(FetchContent)
    FetchContent_Declare(pantheonsim
        URL https://codeload.github.com/pantheongpu/pantheonsim/tar.gz/42919cd4a95b012b7e106094df059f41fa9c71de
        URL_HASH SHA256=7595f5570e650a54577a86fa0df0093a49d1380caf03a3d96bf540de83871f67
    )
    FetchContent_GetProperties(pantheonsim)
    if(NOT pantheonsim_POPULATED)
        FetchContent_Populate(pantheonsim)
    endif()
    set(_ptx_sources
        src/core/error.cpp
        src/core/memory.cpp
        src/core/memory_backing.cpp
        src/core/faults.cpp
        src/ptx/ast.cpp
        src/ptx/lexer.cpp
        src/ptx/parser.cpp
        src/ptx/regalloc.cpp
        src/ptx/contract.cpp
        src/ptx/sync_masks.cpp
        src/simd/simd.cpp
        src/exec/scheduler.cpp
        src/exec/interpreter.cpp
        src/exec/devrt.cpp
        src/exec/worker_pool.cpp
    )
    list(TRANSFORM _ptx_sources PREPEND "${pantheonsim_SOURCE_DIR}/")
    add_library(fake_gpu_ptx STATIC ${_ptx_sources})
    target_include_directories(fake_gpu_ptx SYSTEM PUBLIC "${pantheonsim_SOURCE_DIR}/include")
    target_link_libraries(fake_gpu_ptx PUBLIC Threads::Threads)
    set_target_properties(fake_gpu_ptx PROPERTIES
        CXX_STANDARD 20
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    set_property(TARGET fake_gpu_kernel PROPERTY CXX_STANDARD 20)
    target_compile_definitions(fake_gpu_kernel PRIVATE FAKEGPU_PTX_EXECUTION=1)
    target_link_libraries(fake_gpu_kernel PRIVATE fake_gpu_ptx)
    message(STATUS "FakeGPU PTX execution: ENABLED (PantheonSim)")
else()
    target_compile_definitions(fake_gpu_kernel PRIVATE FAKEGPU_PTX_EXECUTION=0)
    message(STATUS "FakeGPU PTX execution: DISABLED")
endif()
