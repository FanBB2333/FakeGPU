add_library(fake_gpu_core OBJECT
    global_state.cpp
    device.cpp
    gpu_profile.cpp
    # dl_intercept.cpp  # Disabled for now - too complex and causes recursion issues
)

target_include_directories(fake_gpu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR})
# Needs generated Position Independent Code for shared lib linkage
set_property(TARGET fake_gpu_core PROPERTY POSITION_INDEPENDENT_CODE ON)
# Define _GNU_SOURCE for RTLD_NEXT and other GNU extensions
target_compile_definitions(fake_gpu_core PRIVATE _GNU_SOURCE)
