#pragma once

namespace fake_gpu {

class GlobalState;

void* create_native_smi_publisher(GlobalState& state);
void destroy_native_smi_publisher(void* publisher);

}  // namespace fake_gpu
