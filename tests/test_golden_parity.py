"""CI self-contained golden-vector parity gate.

Replays deterministic synthetic inputs (recorded in each ``*.golden.json``
sidecar) through the installed JAX port and compares the outputs and key
staged intermediates against the committed golden ``.npz`` fixtures with a
strict max-abs tolerance. No PyTorch install or external source checkout is
required -- the golden vectors were generated from the trusted converted
artifacts and validated against the PyTorch reference at generation time
(see ``jax_vs_pytorch_max_abs`` in each fixture's JSON sidecar, and
``scripts/generate_golden_fixtures.py``).

This gate catches port regressions (algorithm changes, weight-conversion
drift, masking bugs) on every CI run across the supported Python/JAX matrix
without the flakiness or external-dependency cost of a live PyTorch parity
test. The optional live PyTorch parity tests in ``test_jax_port.py`` remain
for environments that have the source checkpoints + torch installed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(os.environ.get("NISQA_JAX_GOLDEN_DIR", ROOT / "tests" / "golden"))
WEIGHTS_ROOT = Path(os.environ.get("NISQA_JAX_WEIGHTS_DIR", ROOT / "nisqa_jax" / "weights"))

sys.path.insert(0, str(ROOT))

from nisqa_jax.checkpoint import load_model  # noqa: E402
from nisqa_jax.weights import WEIGHTS_DIR  # noqa: E402

# Strict max-abs tolerance for JAX-vs-golden. The golden vectors were produced
# by this same JAX port on the same deterministic inputs, so on the same
# JAX/numpy version the match is exact (0.0). A small tolerance absorbs
# cross-version floating-point reordering (e.g. jaxlib 0.4.30 vs 0.4.38 may
# differ at the ULP level in reductions) without being loose enough to mask a
# real regression. The PyTorch-provenance max-abs (recorded per fixture) is
# ~1e-6, so 5e-5 is the same safety margin used by the live parity suite and
# is far tighter than any plausible port regression.
GOLDEN_MAX_ABS_TOL = 5e-5


def _golden_fixtures() -> list[tuple[Path, Path]]:
    """Return [(artifact_npz, golden_json), ...] for every committed fixture."""
    if not GOLDEN_DIR.is_dir():
        return []
    out = []
    for gj in sorted(GOLDEN_DIR.glob("*.golden.json")):
        meta = json.loads(gj.read_text())
        artifact = WEIGHTS_ROOT / meta["artifact"]
        out.append((artifact, gj))
    return out


def _make_inputs(
    seed: int, batch: int, steps: int, n_wins: int, n_mels: int, seg_length: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, steps, 1, n_mels, seg_length)).astype(np.float32)
    nw = np.full((batch,), n_wins, dtype=np.int32)
    for i in range(batch):
        x[i, n_wins:] = 0.0
    return x, nw


_FIXTURES = _golden_fixtures()


def _skip_if_no_fixtures() -> None:
    if not _FIXTURES:
        pytest.skip(f"no golden fixtures found in {GOLDEN_DIR}")


@pytest.mark.parametrize("artifact,golden_json", _FIXTURES, ids=[p[0].stem for p in _FIXTURES])
def test_jax_outputs_match_golden(artifact: Path, golden_json: Path) -> None:
    """Final model outputs must match the committed golden vectors within 5e-5."""
    if not artifact.exists():
        pytest.skip(f"artifact unavailable: {artifact}")
    meta = json.loads(golden_json.read_text())
    golden_npz = golden_json.with_suffix(".npz")  # *.golden.npz
    if not golden_npz.exists():
        pytest.skip(f"golden npz unavailable: {golden_npz}")
    # Guard against a stale fixture whose artifact checksum changed (re-convert
    # or re-generate). This makes the gate fail loudly on weight drift rather
    # than silently comparing against the wrong golden.
    from nisqa_jax.checkpoint import _sha256

    if _sha256(artifact) != meta["artifact_sha256"]:
        pytest.fail(
            f"artifact {artifact.name} sha256 changed since golden fixture was generated; "
            "re-run scripts/generate_golden_fixtures.py"
        )

    model = load_model(artifact, device="cpu", precision="float32")
    feat = model.config.feature
    with np.load(golden_npz) as loaded:
        for idx, entry in enumerate(meta["inputs"]):
            x, nw = _make_inputs(
                entry["seed"], entry["batch"], entry["steps"], entry["n_wins"], feat.n_mels, feat.seg_length
            )
            out = model.predict_segments(x, nw)
            expected = loaded[f"out_{idx}"]
            assert out.shape == expected.shape, f"input {idx}: output shape {out.shape} != golden {expected.shape}"
            max_abs = float(np.max(np.abs(out - expected)))
            assert max_abs <= GOLDEN_MAX_ABS_TOL, (
                f"input {idx} (seed={entry['seed']}, bs={entry['batch']}, steps={entry['steps']}): "
                f"max|jax-golden| = {max_abs:.3e} > {GOLDEN_MAX_ABS_TOL:.0e}"
            )


@pytest.mark.parametrize("artifact,golden_json", _FIXTURES, ids=[p[0].stem for p in _FIXTURES])
def test_jax_staged_intermediates_match_golden(artifact: Path, golden_json: Path) -> None:
    """Key staged intermediates (cnn, time_dependency) must match golden within 5e-5."""
    if not artifact.exists():
        pytest.skip(f"artifact unavailable: {artifact}")
    meta = json.loads(golden_json.read_text())
    golden_npz = golden_json.with_suffix(".npz")
    if not golden_npz.exists():
        pytest.skip(f"golden npz unavailable: {golden_npz}")
    from nisqa_jax.checkpoint import _sha256

    if _sha256(artifact) != meta["artifact_sha256"]:
        pytest.fail(f"artifact {artifact.name} sha256 drift; regenerate golden fixtures")

    model = load_model(artifact, device="cpu", precision="float32")
    feat = model.config.feature
    with np.load(golden_npz) as loaded:
        for idx, entry in enumerate(meta["inputs"]):
            x, nw = _make_inputs(
                entry["seed"], entry["batch"], entry["steps"], entry["n_wins"], feat.n_mels, feat.seg_length
            )
            stages = model.predict_stages(x, nw)
            for stage_key in ("cnn", "td"):
                actual = stages["cnn" if stage_key == "cnn" else "time_dependency"]
                expected = loaded[f"{stage_key}_{idx}"]
                assert actual.shape == expected.shape, (
                    f"input {idx} stage {stage_key}: shape {actual.shape} != golden {expected.shape}"
                )
                max_abs = float(np.max(np.abs(actual - expected)))
                assert max_abs <= GOLDEN_MAX_ABS_TOL, (
                    f"input {idx} stage {stage_key}: max|jax-golden| = {max_abs:.3e} > {GOLDEN_MAX_ABS_TOL:.0e}"
                )


def test_golden_manifest_consistency() -> None:
    """The top-level GOLDEN_MANIFEST.json must list every committed fixture consistently."""
    _skip_if_no_fixtures()
    manifest_path = GOLDEN_DIR / "GOLDEN_MANIFEST.json"
    if not manifest_path.exists():
        pytest.fail("GOLDEN_MANIFEST.json missing; re-run scripts/generate_golden_fixtures.py")
    manifest = json.loads(manifest_path.read_text())
    from nisqa_jax.checkpoint import _sha256

    for stem, entry in manifest.items():
        golden_npz = GOLDEN_DIR / entry["golden_npz"]
        golden_json = GOLDEN_DIR / entry["golden_json"]
        assert golden_npz.exists(), f"manifest lists missing {golden_npz}"
        assert golden_json.exists(), f"manifest lists missing {golden_json}"
        # Manifest checksum must match the on-disk golden npz.
        assert _sha256(golden_npz) == entry["golden_sha256"], (
            f"{stem}: golden npz checksum drift (manifest vs disk)"
        )
        # Artifact checksum must match the shipped artifact.
        artifact = WEIGHTS_ROOT / f"{stem}.npz"
        assert artifact.exists(), f"artifact {artifact} missing"
        assert _sha256(artifact) == entry["artifact_sha256"], (
            f"{stem}: shipped artifact checksum drift; regenerate golden fixtures"
        )
        # Provenance: every fixture must record a PyTorch-reference max-abs diff
        # (the generation script captures this when --pytorch-ref is given).
        assert entry.get("jax_vs_pytorch_max_abs") is not None, (
            f"{stem}: golden fixture lacks PyTorch provenance; regenerate with --pytorch-ref"
        )
        assert entry["jax_vs_pytorch_max_abs"] <= 5e-5, (
            f"{stem}: golden fixture's jax_vs_pytorch_max_abs "
            f"{entry['jax_vs_pytorch_max_abs']:.3e} exceeds 5e-5 parity bound"
        )


def test_golden_fixtures_cover_all_shipped_artifacts() -> None:
    """Every shipped .npz artifact must have a corresponding golden fixture."""
    _skip_if_no_fixtures()
    manifest = json.loads((GOLDEN_DIR / "GOLDEN_MANIFEST.json").read_text())
    shipped = {p.stem for p in WEIGHTS_DIR.glob("*.npz")}
    covered = set(manifest)
    missing = shipped - covered
    assert not missing, f"shipped artifacts without golden fixtures: {sorted(missing)}"
