import torch
import os

def check_cuda_info():
    """Check CUDA availability and display GPU information."""
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device capabilities: {torch.cuda.get_device_capability()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0)} bytes")
        print(f"Memory cached: {torch.cuda.memory_reserved(0)} bytes")
    print(f"[DEBUG] torch={torch.__version__} cuda={torch.version.cuda} "
      f"is_available={torch.cuda.is_available()} "
      f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")


if __name__ == "__main__":
    check_cuda_info()
