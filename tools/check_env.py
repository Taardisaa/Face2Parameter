"""Environment smoke check for Face2Parameter.

Imports every core dependency, prints versions, asserts CUDA is available, and
(optionally, if online) probes the DINOv2 torch.hub load without failing the run.

Run inside the project venv:
    .venv/Scripts/python.exe tools/check_env.py
"""

import importlib
import sys

# (import name, pip/display name)
CORE_DEPS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("tensorboard", "tensorboard"),
    ("tqdm", "tqdm"),
    ("msgpack", "msgpack"),
    ("skimage", "scikit-image"),
    ("onnxruntime", "onnxruntime"),
    ("requests", "requests"),
]


def _version(mod) -> str:
    return getattr(mod, "__version__", "unknown")


def check_core() -> bool:
    ok = True
    print("=== core dependencies ===")
    for import_name, display in CORE_DEPS:
        try:
            mod = importlib.import_module(import_name)
            print(f"  [ok]   {display:16s} {_version(mod)}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            ok = False
            print(f"  [FAIL] {display:16s} {exc}")
    return ok


def check_cuda() -> bool:
    print("=== cuda ===")
    import torch

    available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available() = {available}")
    if available:
        print(f"  device = {torch.cuda.get_device_name(0)}")
    return available


def probe_dinov2() -> None:
    """Best-effort: confirm the real backbone path is reachable. Never fatal."""
    print("=== dinov2 (optional, needs internet on first run) ===")
    try:
        import torch

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [ok]   dinov2_vits14 loaded ({n:.1f}M params)")
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"  [skip] could not load DINOv2 (fine for skeleton): {exc}")


def main() -> int:
    core_ok = check_core()
    cuda_ok = check_cuda()
    if "--with-dinov2" in sys.argv:
        probe_dinov2()

    print("=== result ===")
    if core_ok and cuda_ok:
        print("  PASS: environment is ready.")
        return 0
    if not core_ok:
        print("  FAIL: one or more core dependencies failed to import.")
    if not cuda_ok:
        print("  WARN: CUDA not available (CPU-only fallback).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
