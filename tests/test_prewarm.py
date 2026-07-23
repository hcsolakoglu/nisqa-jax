"""H2: bucket prewarm tests.

Verifies ``prewarm`` populates the persistent compilation cache so a
subsequent process's first ``predict_segments`` for the prewarmed shape is a
cache hit (no compile stall). Uses subprocesses because the in-process JIT
cache would mask the persistent-cache benefit.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "nisqa_jax" / "weights"))
MOS_ONLY_NPZ = WEIGHTS_ROOT / "nisqa_mos_only.npz"

sys.path.insert(0, str(ROOT))

from nisqa_jax.checkpoint import load_model, prewarm  # noqa: E402
from nisqa_jax.predict import default_length_bucket  # noqa: E402
from _testutil import default_test_device  # noqa: E402


def _skip_if_weights_missing() -> None:
    if not MOS_ONLY_NPZ.exists():
        pytest.skip(f"weights artifact unavailable: {MOS_ONLY_NPZ}")


def _synthetic(model, *, bs: int, steps: int):
    feat = model.config.feature
    rng = np.random.default_rng(0)
    x = rng.normal(size=(bs, steps, 1, feat.n_mels, feat.seg_length)).astype(np.float32)
    n_wins = np.full((bs,), steps, dtype=np.int32)
    return x, n_wins


# ---------------------------------------------------------------------------
# prewarm runs, output discarded, cache dir populated
# ---------------------------------------------------------------------------

def test_prewarm_runs_and_populates_cache(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    # Run in a subprocess: the persistent cache dir is configured via a
    # process-global idempotent flag, so in the shared test process an earlier
    # test may have already pinned the cache to a different (now-deleted) dir.
    # A fresh process guarantees this cache_dir is the first/only one configured.
    cache_dir = tmp_path / "cache"
    # The prewarm subprocess always runs on CPU: the CPU backend is always
    # present in jaxlib (a fresh process can force JAX_PLATFORMS=cpu even when
    # the parent is CUDA-only), and these tests validate persistent-cache
    # mechanics (backend-agnostic) via a fresh process. Running the subprocess
    # on GPU would contend with the parent process's GPU memory on
    # memory-constrained cards (e.g. 8 GB RTX 3070: cuDNN handle init fails
    # with CUDNN_STATUS_INTERNAL_ERROR when the parent holds most of VRAM).
    script = textwrap.dedent(
        f"""
        import sys, json
        sys.path.insert(0, {str(ROOT)!r})
        from pathlib import Path
        from nisqa_jax.checkpoint import load_model, prewarm
        from nisqa_jax.predict import default_length_bucket
        model = load_model({str(MOS_ONLY_NPZ)!r}, device="cpu", cache_dir={str(cache_dir)!r})
        bl = default_length_bucket(model.config)
        prewarm(model, [2], [bl], cache_dir={str(cache_dir)!r})
        cc = Path({str(cache_dir)!r}) / "jax_compilation_cache"
        n_files = sum(1 for f in cc.rglob("*") if f.is_file())
        print(json.dumps({{"exists": cc.exists(), "n_files": n_files}}))
        """
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, timeout=180, check=True,
    )
    import json
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["exists"], "persistent cache directory not created after prewarm"
    assert result["n_files"] > 0, "persistent cache empty after prewarm"


def test_prewarm_rejects_invalid_sizes(tmp_path: Path) -> None:
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY_NPZ, device=default_test_device(), cache_dir=tmp_path / "c")
    bl = default_length_bucket(model.config)
    with pytest.raises(ValueError, match="batch_sizes must be >= 1"):
        prewarm(model, [0], [bl])
    with pytest.raises(ValueError, match="bucket_lengths must be >= 1"):
        prewarm(model, [1], [0])


def test_prewarm_then_predict_matches_cold(tmp_path: Path) -> None:
    """Prewarm must not change numerical output vs a cold predict."""
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY_NPZ, device=default_test_device(), cache_dir=tmp_path / "c")
    bl = default_length_bucket(model.config)
    x, n_wins = _synthetic(model, bs=2, steps=bl)
    cold = model.predict_segments(x, n_wins)  # compiles
    prewarm(model, [2], [bl])  # already compiled in-process; no-op effect
    warm = model.predict_segments(x, n_wins)
    np.testing.assert_allclose(warm, cold, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Cross-process: prewarm eliminates compile stall (timing)
# ---------------------------------------------------------------------------

_TIMING_SCRIPT = textwrap.dedent(
    """
    import os, sys, time, json
    sys.path.insert(0, {root!r})
    import numpy as np
    from nisqa_jax.checkpoint import load_model, prewarm
    from nisqa_jax.predict import default_length_bucket

    mode, cache_dir, artifact, bs, bl = sys.argv[1:6]
    bs, bl = int(bs), int(bl)
    model = load_model(artifact, device="cpu", cache_dir=cache_dir)
    if mode == "prewarm":
        prewarm(model, [bs], [bl], cache_dir=cache_dir)
        print(json.dumps({{"prewarmed": True}}))
    else:
        feat = model.config.feature
        x = np.zeros((bs, bl, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
        n_wins = np.full((bs,), bl, dtype=np.int32)
        t0 = time.perf_counter()
        model.predict_segments(x, n_wins)
        elapsed = time.perf_counter() - t0
        print(json.dumps({{"elapsed": elapsed}}))
    """
)


def _run_timing_subprocess(mode: str, cache_dir: Path, bs: int, bl: int) -> dict:
    # Subprocess runs on CPU (see test_prewarm_runs_and_populates_cache): the
    # CPU backend is always available in a fresh process and this avoids GPU
    # memory contention with the parent process on memory-constrained cards.
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    # Force a fresh process; disable in-process cache reuse noise.
    script = _TIMING_SCRIPT.format(root=str(ROOT))
    out = subprocess.run(
        [sys.executable, "-c", script, mode, str(cache_dir), str(MOS_ONLY_NPZ), str(bs), str(bl)],
        capture_output=True, text=True, env=env, timeout=180, check=True,
    )
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_prewarm_eliminates_compile_stall(tmp_path: Path) -> None:
    """A prewarmed shape's first predict in a fresh process is a cache hit.

    Compares two fresh processes sharing one cache_dir:
      - cold: no prewarm -> first predict compiles (slow).
      - warm: a prewarm subprocess ran first -> first predict is a cache hit (fast).
    The warm predict must be substantially faster than the cold compile.
    """
    _skip_if_weights_missing()
    model = load_model(MOS_ONLY_NPZ, device=default_test_device())  # just to get the bucket size
    bl = default_length_bucket(model.config)
    bs = 2

    # Cold: fresh cache, predict compiles.
    cold_cache = tmp_path / "cold_cache"
    cold = _run_timing_subprocess("predict", cold_cache, bs, bl)

    # Warm: prewarm in one process, then predict in another (same cache).
    warm_cache = tmp_path / "warm_cache"
    _run_timing_subprocess("prewarm", warm_cache, bs, bl)
    warm = _run_timing_subprocess("predict", warm_cache, bs, bl)

    t_cold = cold["elapsed"]
    t_warm = warm["elapsed"]
    # A cache hit is typically 5-50x faster than a cold compile. Use a generous
    # 2x threshold to stay robust to CPU scheduling noise on shared CI.
    assert t_warm < t_cold * 0.5, (
        f"prewarm did not eliminate compile stall: cold={t_cold:.3f}s warm={t_warm:.3f}s"
    )
