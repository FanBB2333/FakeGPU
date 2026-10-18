#pragma once

#include <cstddef>
#include <string>

namespace fake_gpu::kernel {

// CPU execution of PTX modules. Return values use CUDA Driver error codes.
bool available();
int load(const std::string& ptx, void** module);
int function(void* module, const char* name, void** function);
int global(void* module, const char* name, void** pointer, std::size_t* bytes);
int unload(void* module);
int launch(void* function, const unsigned grid[3], const unsigned block[3],
           std::size_t shared_bytes, void** arguments, void** extra);
std::string function_name(void* function);

} // namespace fake_gpu::kernel
