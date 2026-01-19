"""
GPU Temperature Monitoring Utility using pynvml
Provides real-time GPU temperature reading for thermal auto-scaling.
"""

import torch

# Try to import pynvml, gracefully handle if not available
PYNVML_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    print("[LTX2Efficient] Warning: pynvml not installed. Run 'pip install pynvml' for thermal auto-scaling.")
except Exception as e:
    print(f"[LTX2Efficient] Warning: pynvml initialization failed: {e}")

class GPUMonitor:
    """Monitor GPU temperature and provide throttling recommendations."""
    
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.handle = None
        self.available = PYNVML_AVAILABLE
        
        if self.available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                print(f"[LTX2Efficient] GPU Monitor initialized for: {name}")
            except Exception as e:
                print(f"[LTX2Efficient] Failed to get GPU handle: {e}")
                self.available = False
    
    def get_temperature(self) -> int:
        """Get current GPU temperature in Celsius. Returns -1 if unavailable."""
        if not self.available or self.handle is None:
            return -1
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except Exception as e:
            return -1
    
    def get_power_usage(self) -> float:
        """Get current GPU power usage in Watts. Returns -1 if unavailable."""
        if not self.available or self.handle is None:
            return -1.0
        
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
            return power_mw / 1000.0  # Convert to Watts
        except Exception as e:
            return -1.0
    
    def get_utilization(self) -> int:
        """Get current GPU utilization percentage. Returns -1 if unavailable."""
        if not self.available or self.handle is None:
            return -1
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu
        except Exception as e:
            return -1
    
    def calculate_throttle_delay(self, current_temp: int, target_temp: int, 
                                  min_delay: int = 0, max_delay: int = 300) -> int:
        """
        Calculate dynamic throttle delay based on temperature difference.
        
        Args:
            current_temp: Current GPU temperature in Celsius
            target_temp: Target temperature in Celsius
            min_delay: Minimum delay in milliseconds
            max_delay: Maximum delay in milliseconds
            
        Returns:
            Recommended delay in milliseconds
        """
        if current_temp < 0:
            # Can't read temperature, use conservative default
            return 50
        
        if current_temp <= target_temp:
            # Under target, minimal throttling
            return min_delay
        
        # Calculate proportional delay based on how much we exceed target
        temp_excess = current_temp - target_temp
        
        # Scale: every 5°C over target adds more delay
        # 5°C over -> 50ms, 10°C over -> 100ms, 15°C over -> 200ms, etc.
        delay = min_delay + int((temp_excess / 5) * 50)
        
        # Clamp to max
        delay = min(delay, max_delay)
        
        return delay
    
    def should_emergency_pause(self, current_temp: int, critical_temp: int = 83) -> bool:
        """Check if GPU is at critical temperature and needs emergency pause."""
        if current_temp < 0:
            return False
        return current_temp >= critical_temp


# Global instance (lazy initialized)
_gpu_monitor = None

def get_gpu_monitor(device_index: int = 0) -> GPUMonitor:
    """Get or create the global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(device_index)
    return _gpu_monitor
