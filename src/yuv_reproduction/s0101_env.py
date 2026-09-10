# 检查Python和CUDA运行环境

import platform
import sys
import torch

def collect_runtime_environment() -> dict[str, object]:  # 检查环境和CUDA
    python_version = platform.python_version()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch未检测到CUDA")

    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    cuda_probe = torch.tensor([1.0], device="cuda") + 1.0
    if cuda_probe.item() != 2.0:
        raise RuntimeError("CUDA张量计算失败")

    return {
        "python_executable": sys.executable,
        "python_version": python_version,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": True,
        "gpu_name": gpu_name,
        "gpu_compute_capability": list(compute_capability),
        "cuda_tensor_probe": cuda_probe.item(),
        "model_loaded": False,
    }
