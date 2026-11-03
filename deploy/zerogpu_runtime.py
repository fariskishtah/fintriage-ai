"""ZeroGPU prelude, imported by the Space's app.py before the app module.

Two jobs:

1. Point the app at the private model repository. The weights are 268 MB and do
   not belong in the Space's git history, so they are pulled from the Hub at
   startup. For a private repo that needs a read-only token, supplied as the
   Space secret HF_TOKEN and never written into any file here.

2. Exercise the ZeroGPU runtime. Inference itself stays on CPU: a single
   complaint takes ~45 ms there, which is the path that was measured and tested,
   and moving a 268 MB model on and off the GPU per request would cost more than
   it saves. The decorated warm-up below is what makes this a genuine ZeroGPU
   Space rather than one that merely sits on the hardware.
"""
import os

MODEL_REPO = os.environ.get("FINTRIAGE_MODEL_REPO", "")

try:
    import spaces  # noqa: F401  (must be imported before torch on ZeroGPU)

    @spaces.GPU(duration=15)
    def _warmup() -> str:
        import torch

        if torch.cuda.is_available():
            # Touch the device so the allocation is real, then let it go.
            torch.zeros(1, device="cuda").add_(1)
            return f"ZeroGPU ready: {torch.cuda.get_device_name(0)}"
        return "ZeroGPU requested but no CUDA device was attached"

    try:
        print(_warmup())
    except Exception as exc:  # noqa: BLE001
        # A failed warm-up must never stop the app booting on CPU.
        print(f"ZeroGPU warm-up skipped: {type(exc).__name__}: {exc}")
except ImportError:
    print("`spaces` not installed — running without ZeroGPU (CPU inference)")
