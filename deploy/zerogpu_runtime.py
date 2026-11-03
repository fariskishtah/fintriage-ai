"""ZeroGPU prelude, imported by the Space's app.py before the app module.

Why this Space is on ZeroGPU at all: a private Gradio Space on `cpu-basic`
returns HTTP 402 and needs a PRO subscription. `zero-a10g` is the only flavour
this free account can host privately, so ZeroGPU is the hosting tier, not a
performance decision.

Inference runs on CPU. One complaint takes ~45 ms there, that is the path that
was measured and tested, and moving a 268 MB model on and off the GPU per
request would cost more than it saves.

The GPU entry point below is defined but deliberately **not called at startup**.
Every `@spaces.GPU` call reserves a slice of a shared A10G and draws down the
account's ZeroGPU quota; spending that on a warm-up for a workload that never
uses the GPU would be waste. It is kept so the capability is one call away if a
heavier model is ever deployed here.
"""
import os

try:
    import spaces

    @spaces.GPU(duration=15)
    def gpu_available() -> str:
        """Not called during boot. Invoke only if GPU work is ever added."""
        import torch

        if torch.cuda.is_available():
            return f"ZeroGPU attached: {torch.cuda.get_device_name(0)}"
        return "ZeroGPU requested but no CUDA device was attached"

    print("spaces runtime present; inference runs on CPU, no GPU reserved at boot")
except ImportError:
    print("`spaces` not installed — running on CPU")

print("model dir:", os.environ.get("FINTRIAGE_MODEL_DIR", "(from hub)"))
