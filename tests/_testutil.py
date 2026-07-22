"""Shared test utilities for cross-backend (CPU/CUDA/TPU) portability.

Tests historically hard-coded ``device='cpu'``. Under ``JAX_PLATFORMS=cuda`` the
CPU backend is disabled and ``jax.devices('cpu')`` raises, breaking the whole
suite. These helpers resolve a device selector that is ``'cpu'`` when the CPU
backend is available (preserving the original CPU test behavior on CPU-only and
default multi-backend installs) and ``None`` (-> JAX default backend) otherwise,
so the same suite runs clean on a CUDA-only environment without altering any
test's logic or numerical tolerances.
"""
from __future__ import annotations

import jax


def cpu_backend_available() -> bool:
    try:
        return len(jax.devices("cpu")) > 0
    except RuntimeError:
        return False


def default_test_device() -> str | None:
    """Device selector for ``load_model``: ``'cpu'`` if available, else ``None``.

    ``None`` makes ``load_model`` use ``jax.devices()`` (the default backend),
    which is CUDA under ``JAX_PLATFORMS=cuda`` and TPU under ``JAX_PLATFORMS=tpu``.
    Subprocess-based tests (test_prewarm) force ``JAX_PLATFORMS=cpu`` directly
    rather than using this helper, since the CPU backend is always present in a
    fresh jaxlib process and that avoids GPU memory contention with the parent.
    """
    return "cpu" if cpu_backend_available() else None
