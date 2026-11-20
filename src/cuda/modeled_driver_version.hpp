#pragma once

// One modeled driver release for every simulated surface: libcuda's
// cuDriverGetVersion, the runtime's cudaDriverGetVersion and NVML's driver,
// CUDA driver and NVML version queries. A real driver reports the same CUDA
// version through cuDriverGetVersion and nvmlSystemGetCudaDriverVersion, and
// NVML's version string is "<CUDA major>.<driver version>".
namespace fake_gpu {

inline constexpr int kModeledCudaDriverVersion = 12080;  // CUDA 12.8 driver API
inline constexpr char kModeledDriverVersion[] = "570.195.03";
inline constexpr char kModeledNvmlVersion[] = "12.570.195.03";

}  // namespace fake_gpu
