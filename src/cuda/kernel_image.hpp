#pragma once

#include <cstddef>
#include <string>

namespace fake_gpu::kernel {

// A zero size follows cuModuleLoadData's caller-owned image contract.
int extract_ptx(const void* image, std::string& ptx, std::size_t size = 0);

} // namespace fake_gpu::kernel
