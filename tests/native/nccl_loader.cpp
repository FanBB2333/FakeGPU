// Build the same source as an incomplete library and as the loader probe.
#ifdef FAKEGPU_TEST_INCOMPLETE_NCCL
extern "C" int ncclGetVersion(int* version) {
    *version = 1;
    return 0;
}
#else
#include "../../src/nccl/nccl_passthrough.hpp"

#include <cstdio>
#include <string>

int main() {
    auto& loader = fake_gpu::nccl::RealNcclLoader::instance();
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string error;
        if (loader.initialize(error) || loader.available() || error.empty()) {
            std::fprintf(stderr, "incomplete NCCL library accepted on attempt %d\n", attempt);
            return 1;
        }
    }
    std::puts("NCCL loader preserves failed initialization");
}
#endif
