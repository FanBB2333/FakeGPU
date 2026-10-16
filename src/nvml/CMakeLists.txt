add_library(fake_gpu_nvml OBJECT
    nvml_stubs.cpp
    state_bridge.cpp
)

target_include_directories(fake_gpu_nvml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_nvml PRIVATE ../core)
set_property(TARGET fake_gpu_nvml PROPERTY POSITION_INDEPENDENT_CODE ON)
