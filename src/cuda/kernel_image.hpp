#pragma once

#include <cstddef>
#include <string>

namespace fake_gpu::kernel {

// A zero size follows cuModuleLoadData's caller-owned image contract.
int extract_image(const void* image, std::string& code, unsigned compute_capability,
                  std::size_t size = 0);

} // namespace fake_gpu::kernel
