"""O2(a) extended OOM boundary probe — push past the non-fatal BFC warnings to
find the batch_size that raises a *fatal* RESOURCE_EXHAUSTED exception.
"""
from __future__ import annotations

import gc
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, ".")

from nisqa_jax.checkpoint import load_model


def _try_run(model, bs, steps):
    feat = model.config.feature
    rng = np.random.default_rng(0)
    x = rng.normal(size=(bs, steps, 1, feat.n_mels, feat.seg_length)).astype(np.float32)
    n_wins = np.full((bs,), steps, dtype=np.int32)
    try:
        out = model.predict_segments(x, n_wins)
        jax.block_until_ready(jnp.asarray(out))
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def sweep(label, weight_path, steps, sizes):
    print(f"\n=== {label}  ({weight_path}, steps={steps}) ===", flush=True)
    model = load_model(weight_path, device="gpu", precision="float32")
    _try_run(model, 1, min(steps, 16))  # warmup compile
    max_ok = 0
    for bs in sizes:
        gc.collect()
        ok, msg = _try_run(model, bs, steps)
        print(f"  bs={bs:<4} -> {'OK ' if ok else 'FAIL'}  {msg}", flush=True)
        if ok:
            max_ok = bs
        else:
            break
    print(f"  >>> max bs without fatal OOM at steps={steps}: {max_ok}", flush=True)


if __name__ == "__main__":
    print(f"jax {jax.__version__}  devices={jax.devices()}", flush=True)
    sweep("self_att (mos_only)", "weights/nisqa_mos_only.npz", 1300, [32, 48, 64, 96, 128])
    sweep("self_att (dim, 5-out)", "weights/nisqa.npz", 1300, [16, 24, 32, 48, 64])
    sweep("tts (BiLSTM)", "weights/nisqa_tts.npz", 6000, [8, 12, 16, 24, 32])
