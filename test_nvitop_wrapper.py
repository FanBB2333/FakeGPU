#!/usr/bin/env python3
"""
nvitop wrapper - 使用Python API安全地使用nvitop
避免TUI模式的段错误问题
"""

import sys
import time
from datetime import datetime

def clear_screen():
    print('\033[2J\033[H', end='')

def main():
    try:
        from nvitop import Device
        
        print("nvitop (FakeGPU 兼容模式)")
        print("=" * 80)
        print()
        
        # 持续监控模式
        update_interval = 1  # 秒
        
        while True:
            clear_screen()
            
            print(f"nvitop - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()
            
            devices = Device.all()
            
            # 表头
            print(f"{'GPU':>3} | {'Name':30} | {'Mem-Usage':>12} | {'Util':>5} | {'Temp':>5} | {'Power':>7}")
            print("-" * 80)
            
            for i, dev in enumerate(devices):
                try:
                    mem_info = dev.memory_info()
                    mem_used = mem_info.used / 1024**3
                    mem_total = mem_info.total / 1024**3
                    mem_str = f"{mem_used:5.1f}/{mem_total:5.1f}GB"
                    
                    try:
                        util = f"{dev.gpu_utilization():3d}%"
                    except:
                        util = "N/A"
                    
                    try:
                        temp = f"{dev.temperature():3d}°C"
                    except:
                        temp = "N/A"
                    
                    try:
                        power = f"{dev.power_usage()/1000:5.1f}W"
                    except:
                        power = "N/A"
                    
                    print(f"{i:3d} | {dev.name():30} | {mem_str:>12} | {util:>5} | {temp:>5} | {power:>7}")
                    
                except Exception as e:
                    print(f"{i:3d} | Error: {e}")
            
            print()
            print("=" * 80)
            print("按 Ctrl+C 退出 | 刷新间隔: {}秒".format(update_interval))
            print()
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\n程序已退出")
        sys.exit(0)
    except ImportError:
        print("错误: 未安装 nvitop")
        print("安装命令: pip install nvitop")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
