"""CI self-contained golden-vector parity gate with PyTorch-reference trust chain.

Replays deterministic synthetic inputs (recorded in each ``*.golden.json``
sidecar) through the installed JAX port and compares the outputs and key
staged intermediates against the committed golden ``.npz`` fixtures with a
strict max-abs tolerance. No PyTorch install or external source checkout is
required to *run* this gate -- the golden vectors were generated from the
trusted converted artifacts and validated against the PyTorch reference at
generation time.

In addition to the JAX-vs-golden regression check, every fixture carries a
``*_ptref.npz`` containing the **real PyTorch reference outputs** for the same
inputs. This gate verifies, for every fixture:

  1. The ``ptref`` file exists (a missing ptref is a hard FAIL, not a skip --
     the trust chain is the primary correctness proof).
  2. The ``ptref`` file's raw SHA-256 matches the value recorded in
     ``GOLDEN_MANIFEST.json`` (and the sidecar) -- catches post-commit
     tampering/bit-rot of the PyTorch reference vectors.
  3. The ``ptref`` file has complete ``out_{i}`` keys with the expected
     shapes (one per recorded input).
  4. The **current JAX final outputs** compare directly against the PyTorch
     reference outputs within ``5e-5`` -- the live JAX-vs-PyTorch parity
     bound, justified by the ~1e-6 generation-time max-abs and the same
     margin used by the live parity suite.
  5. The recorded ``jax_vs_pytorch_max_abs`` provenance value is recomputed
     from the committed golden + ptref vectors and validated to match (the
     golden vectors are the JAX outputs captured at generation time, so this
     confirms the provenance field is truthful, not fabricated).

The JAX-vs-golden intermediate (cnn, time_dependency) regression is kept as a
separate test so a port regression in a staged intermediate is reported
independently of the final-output PyTorch parity.

The optional live PyTorch parity tests in ``test_jax_port.py`` remain for
environments that have the source checkpoints + torch installed.
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

from _testutil import default_test_device  # noqa: E402
from nisqa_jax.checkpoint import _sha256, load_model  # noqa: E402
from nisqa_jax.weights import WEIGHTS_DIR  # noqa: E402

# Strict max-abs tolerance for JAX-vs-golden and JAX-vs-PyTorch-reference. The
# golden vectors were produced by this same JAX port on the same deterministic
# inputs, so on the same JAX/numpy version the JAX-vs-golden match is exact
# (0.0). A small tolerance absorbs cross-version floating-point reordering
# (e.g. jaxlib 0.4.30 vs 0.4.38 may differ at the ULP level in reductions)
# without being loose enough to mask a real regression. The PyTorch-reference
# max-abs (recorded per fixture) is ~1e-6, so 5e-5 is the same safety margin
# used by the live parity suite and is far tighter than any plausible port
# regression.
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


def _require_artifact(artifact: Path) -> None:
    """Artifact must exist -- a missing shipped artifact is a release-blocking error."""
    if not artifact.exists():
        pytest.fail(f"shipped artifact unavailable: {artifact}")


def _require_golden_npz(golden_json: Path) -> Path:
    golden_npz = golden_json.with_suffix(".npz")  # *.golden.npz
    if not golden_npz.exists():
        pytest.fail(f"golden npz unavailable: {golden_npz}")
    return golden_npz


def _require_ptref(stem: str, meta: dict, golden_json: Path) -> Path:
    """The PyTorch reference npz must exist and be listed -- missing ptref FAILs.

    The ptref files are the trust chain: they are real PyTorch outputs that
    prove the JAX port was validated against the reference. A missing ptref
    means the trust chain is broken, which is a release-blocking error, not a
    skip, in the primary CI gate.
    """
    ptref_name = meta.get("ptref_npz") or f"{stem}.ptref.npz"
    ptref_path = golden_json.parent / ptref_name
    if not ptref_path.exists():
        pytest.fail(
            f"PyTorch reference npz missing: {ptref_path}. The golden trust chain "
            "requires a committed ptref; regenerate with --pytorch-ref."
        )
    return ptref_path


@pytest.mark.parametrize("artifact,golden_json", _FIXTURES, ids=[p[0].stem for p in _FIXTURES])
def test_jax_outputs_match_golden(artifact: Path, golden_json: Path) -> None:
    """Final model outputs must match the committed golden vectors within 5e-5."""
    _require_artifact(artifact)
    meta = json.loads(golden_json.read_text())
    golden_npz = _require_golden_npz(golden_json)
    # Guard against a stale fixture whose artifact checksum changed (re-convert
    # or re-generate). This makes the gate fail loudly on weight drift rather
    # than silently comparing against the wrong golden.
    if _sha256(artifact) != meta["artifact_sha256"]:
        pytest.fail(
            f"artifact {artifact.name} sha256 changed since golden fixture was generated; "
            "re-run scripts/generate_golden_fixtures.py"
        )

    model = load_model(artifact, device=default_test_device(), precision="float32")
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
    """Key staged intermediates (cnn, time_dependency) must match golden within 5e-5.

    Kept separate from the final-output PyTorch parity test so a staged
    intermediate regression is reported independently.
    """
    _require_artifact(artifact)
    meta = json.loads(golden_json.read_text())
    golden_npz = _require_golden_npz(golden_json)
    if _sha256(artifact) != meta["artifact_sha256"]:
        pytest.fail(f"artifact {artifact.name} sha256 drift; regenerate golden fixtures")

    model = load_model(artifact, device=default_test_device(), precision="float32")
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


@pytest.mark.parametrize("artifact,golden_json", _FIXTURES, ids=[p[0].stem for p in _FIXTURES])
def test_ptref_trust_chain(artifact: Path, golden_json: Path) -> None:
    """PyTorch reference trust chain: ptref exists, hash matches, keys/shapes complete.

    This is the primary correctness proof: the committed ``*_ptref.npz`` files
    are real PyTorch outputs. A missing or tampered ptref is a hard failure
    (not a skip) because the trust chain is what makes the self-contained
    golden gate meaningful -- without it the golden vectors are just the JAX
    port replayed against itself.
    """
    _require_artifact(artifact)
    stem = artifact.stem
    meta = json.loads(golden_json.read_text())
    ptref_path = _require_ptref(stem, meta, golden_json)

    # 1. Raw SHA-256 of the ptref file must match the manifest + sidecar.
    ptref_sha = _sha256(ptref_path)
    manifest = json.loads((GOLDEN_DIR / "GOLDEN_MANIFEST.json").read_text())
    manifest_entry = manifest.get(stem, {})
    expected_manifest_sha = manifest_entry.get("ptref_sha256")
    expected_sidecar_sha = meta.get("ptref_sha256")
    assert expected_manifest_sha is not None, (
        f"{stem}: GOLDEN_MANIFEST.json missing ptref_sha256; regenerate golden fixtures"
    )
    assert expected_sidecar_sha is not None, (
        f"{golden_json.name}: sidecar missing ptref_sha256; regenerate golden fixtures"
    )
    assert ptref_sha == expected_manifest_sha, (
        f"{stem}: ptref raw sha256 {ptref_sha[:12]}… != manifest {expected_manifest_sha[:12]}… "
        "(ptref file tampered/bit-rotted since commit)"
    )
    assert ptref_sha == expected_sidecar_sha, (
        f"{golden_json.name}: ptref raw sha256 {ptref_sha[:12]}… != sidecar {expected_sidecar_sha[:12]}…"
    )

    # 2. Key/shape completeness: one out_{i} per recorded input, matching the
    #    golden output shapes (PyTorch and JAX produce the same output shapes).
    golden_npz = _require_golden_npz(golden_json)
    n_inputs = len(meta["inputs"])
    with np.load(ptref_path) as ptref, np.load(golden_npz) as golden:
        ptref_keys = sorted(ptref.files)
        expected_keys = [f"out_{i}" for i in range(n_inputs)]
        assert ptref_keys == expected_keys, (
            f"{stem}: ptref keys {ptref_keys} != expected {expected_keys} "
            f"(n_inputs={n_inputs})"
        )
        for i in range(n_inputs):
            pt_arr = ptref[f"out_{i}"]
            g_arr = golden[f"out_{i}"]
            assert pt_arr.shape == g_arr.shape, (
                f"{stem} out_{i}: ptref shape {pt_arr.shape} != golden shape {g_arr.shape}"
            )
            assert pt_arr.dtype.kind == "f", (
                f"{stem} out_{i}: ptref dtype {pt_arr.dtype} is not floating-point"
            )


@pytest.mark.parametrize("artifact,golden_json", _FIXTURES, ids=[p[0].stem for p in _FIXTURES])
def test_jax_outputs_match_pytorch_reference(artifact: Path, golden_json: Path) -> None:
    """Current JAX final outputs must match the PyTorch reference outputs within 5e-5.

    This is the live JAX-vs-PyTorch parity check, run self-contained (no torch
    install needed -- the PyTorch outputs are committed in the ptref npz). The
    5e-5 tolerance is the same margin used by the live parity suite; the
    generation-time max-abs is ~1e-6.
    """
    _require_artifact(artifact)
    stem = artifact.stem
    meta = json.loads(golden_json.read_text())
    ptref_path = _require_ptref(stem, meta, golden_json)
    if _sha256(artifact) != meta["artifact_sha256"]:
        pytest.fail(
            f"artifact {artifact.name} sha256 changed since golden fixture was generated; "
            "re-run scripts/generate_golden_fixtures.py"
        )

    model = load_model(artifact, device=default_test_device(), precision="float32")
    feat = model.config.feature
    with np.load(ptref_path) as ptref:
        for idx, entry in enumerate(meta["inputs"]):
            x, nw = _make_inputs(
                entry["seed"], entry["batch"], entry["steps"], entry["n_wins"], feat.n_mels, feat.seg_length
            )
            out = model.predict_segments(x, nw)
            pt_out = ptref[f"out_{idx}"]
            assert out.shape == pt_out.shape, (
                f"input {idx}: jax shape {out.shape} != pytorch ref shape {pt_out.shape}"
            )
            max_abs = float(np.max(np.abs(out - pt_out)))
            assert max_abs <= GOLDEN_MAX_ABS_TOL, (
                f"input {idx} (seed={entry['seed']}, bs={entry['batch']}, steps={entry['steps']}): "
                f"max|jax-pytorch| = {max_abs:.3e} > {GOLDEN_MAX_ABS_TOL:.0e}"
            )


@pytest.mark.parametrize("artifact,golden_json", _FIXTURES, ids=[p[0].stem for p in _FIXTURES])
def test_recorded_jax_vs_pytorch_max_abs_is_truthful(artifact: Path, golden_json: Path) -> None:
    """Recompute the recorded ``jax_vs_pytorch_max_abs`` from golden + ptref and validate.

    The golden npz holds the JAX outputs captured at generation time; the ptref
    holds the PyTorch outputs. Their max-abs diff must equal the
    ``jax_vs_pytorch_max_abs`` value recorded in the manifest + sidecar (within
    a tiny float tolerance for cross-numpy serialisation). This confirms the
    provenance field is truthful, not fabricated, and that the committed golden
    + ptref pair is internally consistent.
    """
    _require_artifact(artifact)
    stem = artifact.stem
    meta = json.loads(golden_json.read_text())
    ptref_path = _require_ptref(stem, meta, golden_json)
    golden_npz = _require_golden_npz(golden_json)

    recorded = meta.get("jax_vs_pytorch_max_abs")
    assert recorded is not None, (
        f"{golden_json.name}: missing jax_vs_pytorch_max_abs provenance; regenerate with --pytorch-ref"
    )
    recomputed = 0.0
    with np.load(golden_npz) as golden, np.load(ptref_path) as ptref:
        for i in range(len(meta["inputs"])):
            diff = float(np.max(np.abs(golden[f"out_{i}"] - ptref[f"out_{i}"])))
            recomputed = max(recomputed, diff)
    # Cross-numpy serialisation can introduce ULP noise; allow a 1e-9 relative
    # slack so the provenance value is validated as truthful without being
    # brittle to float reordering.
    assert abs(recomputed - recorded) <= 1e-9 + 1e-6 * abs(recorded), (
        f"{stem}: recorded jax_vs_pytorch_max_abs={recorded:.3e} but recomputed={recomputed:.3e} "
        "from committed golden+ptref (provenance field is inconsistent with the committed vectors)"
    )
    # The provenance value itself must satisfy the parity bound.
    assert recorded <= GOLDEN_MAX_ABS_TOL, (
        f"{stem}: recorded jax_vs_pytorch_max_abs={recorded:.3e} exceeds {GOLDEN_MAX_ABS_TOL:.0e} parity bound"
    )


def test_golden_manifest_consistency() -> None:
    """The top-level GOLDEN_MANIFEST.json must list every committed fixture consistently."""
    _skip_if_no_fixtures()
    manifest_path = GOLDEN_DIR / "GOLDEN_MANIFEST.json"
    if not manifest_path.exists():
        pytest.fail("GOLDEN_MANIFEST.json missing; re-run scripts/generate_golden_fixtures.py")
    manifest = json.loads(manifest_path.read_text())

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
        # Trust chain: every fixture must record a ptref filename + raw sha256,
        # and the ptref file must exist on disk with a matching hash.
        ptref_name = entry.get("ptref_npz")
        ptref_sha = entry.get("ptref_sha256")
        assert ptref_name is not None, f"{stem}: manifest missing ptref_npz; regenerate with --pytorch-ref"
        assert ptref_sha is not None, f"{stem}: manifest missing ptref_sha256; regenerate with --pytorch-ref"
        ptref_path = GOLDEN_DIR / ptref_name
        assert ptref_path.exists(), (
            f"{stem}: manifest lists ptref {ptref_name} but file is missing on disk"
        )
        assert _sha256(ptref_path) == ptref_sha, (
            f"{stem}: ptref {ptref_name} raw sha256 drift (manifest vs disk)"
        )


def test_golden_fixtures_cover_all_shipped_artifacts() -> None:
    """Every shipped .npz artifact must have a corresponding golden fixture."""
    _skip_if_no_fixtures()
    manifest = json.loads((GOLDEN_DIR / "GOLDEN_MANIFEST.json").read_text())
    shipped = {p.stem for p in WEIGHTS_DIR.glob("*.npz")}
    covered = set(manifest)
    missing = shipped - covered
    assert not missing, f"shipped artifacts without golden fixtures: {sorted(missing)}"
