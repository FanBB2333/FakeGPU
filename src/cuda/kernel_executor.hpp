#pragma once

#include <cstddef>
#include <string>

namespace fake_gpu::kernel {

// CPU execution of PTX/cubin modules. Return values use CUDA Driver error codes.
bool available();
int ptx_compatible(const std::string& image, unsigned compute_capability);
int cubin_compatible(const std::string& image, unsigned compute_capability);
int load(const std::string& image, void** module);
int function(void* module, const char* name, void** function);
int global(void* module, const char* name, void** pointer, std::size_t* bytes);
int unload(void* module);
int launch(void* function, const unsigned grid[3], const unsigned block[3],
           std::size_t shared_bytes, void** arguments, void** extra);
std::string function_name(void* function);
// Resolve cubin data-section addresses used by device pointer initializers.
int copy(void* destination, const void* source, std::size_t bytes);
int fill(void* destination, unsigned value, std::size_t count, unsigned element_size);

} // namespace fake_gpu::kernel
