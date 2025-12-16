#pragma once

// Logging control macros for FakeGPU
// Use CMAKE option ENABLE_FAKEGPU_LOGGING to control debug output

#ifdef ENABLE_FAKEGPU_LOGGING
    #define FGPU_LOG(...) printf(__VA_ARGS__)
#else
    #define FGPU_LOG(...) do {} while(0)
#endif
