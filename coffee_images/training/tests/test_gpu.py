import torch

# Check for XPU device (Intel GPU/accelerator)
xpu_available = torch.xpu.is_available() if hasattr(torch, "xpu") else False
print("XPU available:", xpu_available)

if xpu_available:
    print("XPU device count:", torch.xpu.device_count())
    print("XPU device name:", torch.xpu.get_device_name(0))
    # Try a simple tensor operation on XPU
    x = torch.ones(3, 3, device="xpu")
    y = x * 2
    print("Tensor on XPU:", y)
else:
    print("No XPU device found.")