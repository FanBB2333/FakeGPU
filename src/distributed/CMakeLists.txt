add_library(fake_gpu_distributed OBJECT
    cluster_config.cpp
    communicator.cpp
    cluster_coordinator.cpp
    collective_allreduce.cpp
    collective_broadcast.cpp
    transport_local.cpp
    staging_buffer.cpp
)

target_include_directories(fake_gpu_distributed PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

set_property(TARGET fake_gpu_distributed PROPERTY POSITION_INDEPENDENT_CODE ON)
