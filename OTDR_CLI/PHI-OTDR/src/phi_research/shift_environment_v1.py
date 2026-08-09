"""Record the exact PHI shift-study runtime and a mandatory CUDA smoke test."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from .shift_protocol_v1 import finalize_payload, process_memory_snapshot


CORE_PACKAGES = ("numpy", "scipy", "scikit-learn", "matplotlib", "torch", "pytest")


def _command(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run(output_path: Path) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for the PHI neural gate; no CPU fallback is allowed")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    left = torch.arange(1024 * 1024, device="cuda", dtype=torch.float32).reshape(1024, 1024)
    result = left @ left.T
    checksum = float(result[0, 0].item())
    torch.cuda.synchronize()
    smoke_seconds = time.perf_counter() - started
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    packages = {}
    for package in CORE_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    nvidia_smi = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    pip_check = _command([sys.executable, "-m", "pip", "check"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "PHI-OTDR shift-v1 environment and CUDA smoke",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
        "cuda": {
            "available": True,
            "device_index": device,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "free_memory_bytes_after_smoke": int(free),
            "reported_total_memory_bytes": int(total),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "smoke_seconds": smoke_seconds,
            "smoke_checksum": checksum,
            "cpu_fallback": False,
        },
        "nvidia_smi": nvidia_smi,
        "pip_check": pip_check,
        "process_memory": process_memory_snapshot(),
        "dependency_interpretation": (
            "The PHI shift-v1 code used only the pinned numerical core. A nonzero pip-check "
            "result records conflicts in the broader reusable environment and is not hidden."
        ),
    }
    return finalize_payload(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.output)
    print(json.dumps({"payload_sha256": payload["payload_sha256"], "cuda": payload["cuda"]}, indent=2))


if __name__ == "__main__":
    main()
