#!/usr/bin/env python3
"""Display GPU information using nvitop API."""

from nvitop import Device
import pynvml

def main():
    # Initialize NVML to get device count
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()

    print("=" * 80)
    print("GPU Information (via nvitop)")
    print("=" * 80)

    # Create devices one at a time instead of using Device.all()
    for i in range(device_count):
        device = Device(i)
        print(f"\nGPU {device.index}: {device.name()}")
        print("-" * 80)

        # Basic info
        print(f"  Bus ID:              {device.bus_id() or 'N/A'}")
        print(f"  UUID:                {device.uuid() or 'N/A'}")

        # Memory
        mem_total = device.memory_total()
        mem_used = device.memory_used()
        mem_free = device.memory_free()
        mem_util = device.memory_utilization()
        print(f"\n  Memory Total:        {device.memory_total_human()}")
        print(f"  Memory Used:         {device.memory_used_human()} ({mem_util}%)")
        print(f"  Memory Free:         {device.memory_free_human()}")

        # Utilization
        gpu_util = device.gpu_utilization()
        print(f"\n  GPU Utilization:     {gpu_util}%")
        print(f"  Memory Utilization:  {mem_util}%")

        # Temperature
        try:
            temp = device.temperature()
            print(f"\n  Temperature:         {temp}°C")
        except:
            print(f"\n  Temperature:         N/A")

        # Power
        try:
            power_usage = device.power_usage()
            power_limit = device.power_limit()
            power_percent = (power_usage / power_limit * 100) if power_limit > 0 else 0
            print(f"  Power Usage:         {power_usage / 1000:.2f}W / {power_limit / 1000:.2f}W ({power_percent:.1f}%)")
        except:
            print(f"  Power Usage:         N/A")

        # Performance state
        try:
            pstate = device.performance_state()
            print(f"  Performance State:   P{pstate}")
        except:
            print(f"  Performance State:   N/A")

        # Fan speed
        try:
            fan_speed = device.fan_speed()
            print(f"  Fan Speed:           {fan_speed}%")
        except:
            print(f"  Fan Speed:           N/A")

        # Clock speeds - use pynvml directly since nvitop's clock() has issues
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(device.index)
            gpu_clock = nvml.nvmlDeviceGetClockInfo(handle, 0)  # Graphics clock
            mem_clock = nvml.nvmlDeviceGetClockInfo(handle, 2)  # Memory clock
            nvml.nvmlShutdown()
            print(f"\n  GPU Clock:           {gpu_clock} MHz")
            print(f"  Memory Clock:        {mem_clock} MHz")
        except Exception as e:
            print(f"\n  Clock Speeds:        N/A")

        # Encoder/Decoder
        try:
            encoder_util = device.encoder_utilization()
            decoder_util = device.decoder_utilization()
            print(f"\n  Encoder Utilization: {encoder_util}%")
            print(f"  Decoder Utilization: {decoder_util}%")
        except:
            print(f"\n  Encoder/Decoder:     N/A")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
