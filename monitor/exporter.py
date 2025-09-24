from prometheus_client import Gauge, start_http_server
import psutil, pynvml, redis, time

start_http_server(8004)
gpu_mem = Gauge("gpu_mem_used_mb", "GPU memory MB")
gpu_mem_frac = Gauge("gpu_mem_frac", "GPU memory fraction allocated by PyTorch")
active_color = Gauge("active_color", "Active deployment color (blue=0, green=1)")
cpu_pct = Gauge("cpu_pct", "CPU util %")
r = redis.Redis(decode_responses=True)
while True:
    cpu_pct.set(psutil.cpu_percent(interval=1))
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_mem.set(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e6)
    mem_frac = float(r.get("gpu:mem_frac") or "0.8")
    gpu_mem_frac.set(mem_frac)
    color_val = float(r.get("metrics:active_color") or "0")
    active_color.set(color_val)
    time.sleep(5)
