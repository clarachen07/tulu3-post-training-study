import platform
import sys

def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")

    try:
        import torch
    except Exception as e:
        print(f"ERROR: failed to import torch: {e}")
        return 1

    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except Exception as e:
        print(f"WARNING: failed to import transformers: {e}")

    try:
        import accelerate
        print(f"accelerate version: {accelerate.__version__}")
    except Exception as e:
        print(f"WARNING: failed to import accelerate: {e}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not cuda_available:
        print("ERROR: torch cannot see CUDA. Stop here and fix the GPU environment.")
        return 1

    device = torch.device("cuda:0")
    print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
    print(f"GPU total memory: {total_gb:.2f} GB")
    print(f"GPU allocated memory: {allocated_gb:.4f} GB")
    print(f"GPU reserved memory: {reserved_gb:.4f} GB")

    x = torch.randn(512, 512, device=device)
    y = torch.randn(512, 512, device=device)
    z = x @ y
    torch.cuda.synchronize()
    print(f"Matmul result shape: {tuple(z.shape)}")

    del x, y, z
    torch.cuda.empty_cache()

    print("GPU environment check passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
