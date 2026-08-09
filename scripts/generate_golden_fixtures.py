#!/usr/bin/env python
"""Generate deterministic golden-vector fixtures for CI parity gating.

For each of the 3 shipped checkpoints, runs the JAX port on a set of
deterministic synthetic inputs (multiple batch/shape/n_wins combinations) and
records the final outputs plus key staged intermediates (cnn, time_dependency)
into a single ``.npz`` fixture, with a JSON sidecar describing the inputs and
the JAX/jaxlib/numpy versions used to produce them.

The fixtures make CI self-contained: the parity gate replays the same inputs
through the installed JAX port and compares against the committed golden
outputs with a strict max-abs tolerance -- no PyTorch install or external
source checkout is required in CI.

Optionally (when ``--pytorch-ref /tmp/nisqa_ref`` is given and torch is
importable), this script also runs the PyTorch reference on the same inputs and
records the PyTorch outputs in a separate ``*_ptref.npz`` so the golden
fixtures carry provenance that the JAX port was validated against PyTorch at
generation time. The CI gate itself only uses the JAX golden outputs.

Usage::

    # JAX-only golden generation (CI-relevant; no torch needed):
    python scripts/generate_golden_fixtures.py --out tests/golden

    # Full provenance: also capture PyTorch reference outputs:
    python scripts/generate_golden_fixtures.py --out tests/golden \
        --pytorch-ref /tmp/nisqa_ref
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _artifacts() -> list[Path]:
    from nisqa_jax.weights import WEIGHTS_DIR

    return sorted(WEIGHTS_DIR.glob("*.npz"))


# Deterministic input grid per checkpoint: (seed, batch_size, steps, n_wins_value).
# Multiple shapes exercise different compiled specializations and the
# masked-padding path (n_wins < steps). Seeds are fixed so the generated
# golden vectors are byte-reproducible across machines.
_INPUT_GRID: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 16, 16),
    (1, 1, 24, 20),
    (2, 2, 32, 32),
    (3, 2, 48, 40),
    (4, 3, 64, 60),
)


def _make_inputs(
    seed: int, batch: int, steps: int, n_wins: int, n_mels: int, seg_length: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, steps, 1, n_mels, seg_length)).astype(np.float32)
    # Mask the tail beyond n_wins with zeros so the golden output is well-defined
    # regardless of padding-invariance assumptions (the port masks internally too,
    # but zeroing keeps the fixture self-consistent if masking logic changes).
    nw = np.full((batch,), n_wins, dtype=np.int32)
    for i in range(batch):
        x[i, n_wins:] = 0.0
    return x, nw


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_ptref(
    artifact: Path, inputs: list[tuple[np.ndarray, np.ndarray]], ref_root: Path
) -> dict[str, np.ndarray] | None:
    """Run the PyTorch reference on the same inputs; return outputs or None."""
    import importlib
    import sys

    ref_weights = ref_root / "weights"
    tar = ref_weights / artifact.with_suffix(".tar").name
    if not tar.exists():
        print(f"  (skip ptref: {tar} not found)")
        return None
    sys.path.insert(0, str(ref_root))
    try:
        torch = importlib.import_module("torch")
        nl = importlib.import_module("nisqa.NISQA_lib")
    except Exception as exc:  # pragma: no cover - optional provenance path
        print(f"  (skip ptref: {exc})")
        return None

    from nisqa_jax.checkpoint import _load_torch_checkpoint

    ck = _load_torch_checkpoint(torch, tar)
    args = ck["args"]
    cls = {"NISQA": nl.NISQA, "NISQA_DIM": nl.NISQA_DIM}[args["model"]]
    # Reuse the port's arg-mapping to build the model constructor kwargs.
    model_args = _pt_model_args(args)
    model = cls(**model_args)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for x, nw in inputs:
            out = model(torch.from_numpy(x), torch.from_numpy(nw)).numpy()
            outs.append(np.asarray(out, dtype=np.float32))
    return {f"out_{i}": o for i, o in enumerate(outs)}


def _pt_model_args(args: dict) -> dict:
    # Mirror tests/test_jax_port.py:_model_args so the PyTorch reference model
    # is constructed with the exact same hyperparameters as training.
    return {
        "ms_seg_length": args["ms_seg_length"],
        "ms_n_mels": args["ms_n_mels"],
        "cnn_model": args["cnn_model"],
        "cnn_c_out_1": args["cnn_c_out_1"],
        "cnn_c_out_2": args["cnn_c_out_2"],
        "cnn_c_out_3": args["cnn_c_out_3"],
        "cnn_kernel_size": args["cnn_kernel_size"],
        "cnn_dropout": args["cnn_dropout"],
        "cnn_pool_1": args["cnn_pool_1"],
        "cnn_pool_2": args["cnn_pool_2"],
        "cnn_pool_3": args["cnn_pool_3"],
        "cnn_fc_out_h": args["cnn_fc_out_h"],
        "td": args["td"],
        "td_sa_d_model": args["td_sa_d_model"],
        "td_sa_nhead": args["td_sa_nhead"],
        "td_sa_pos_enc": args["td_sa_pos_enc"],
        "td_sa_num_layers": args["td_sa_num_layers"],
        "td_sa_h": args["td_sa_h"],
        "td_sa_dropout": args["td_sa_dropout"],
        "td_lstm_h": args["td_lstm_h"],
        "td_lstm_num_layers": args["td_lstm_num_layers"],
        "td_lstm_dropout": args["td_lstm_dropout"],
        "td_lstm_bidirectional": args["td_lstm_bidirectional"],
        "td_2": args["td_2"],
        "td_2_sa_d_model": args["td_2_sa_d_model"],
        "td_2_sa_nhead": args["td_2_sa_nhead"],
        "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
        "td_2_sa_num_layers": args["td_2_sa_num_layers"],
        "td_2_sa_h": args["td_2_sa_h"],
        "td_2_sa_dropout": args["td_2_sa_dropout"],
        "td_2_lstm_h": args["td_2_lstm_h"],
        "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
        "td_2_lstm_dropout": args["td_2_lstm_dropout"],
        "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
        "pool": args["pool"],
        "pool_att_h": args["pool_att_h"],
        "pool_att_dropout": args["pool_att_dropout"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "tests" / "golden", help="output directory for golden fixtures"
    )
    parser.add_argument(
        "--pytorch-ref", type=Path, default=None, help="optional PyTorch reference root for provenance capture"
    )
    parser.add_argument("--device", default="cpu", help="JAX device selector (default: cpu for portability)")
    args = parser.parse_args(argv)

    import jax
    import numpy as np  # noqa: F811

    from nisqa_jax.checkpoint import load_model

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import jaxlib  # noqa: F401

        jaxlib_version = jaxlib.__version__ if hasattr(jaxlib, "__version__") else "unknown"
    except Exception:
        jaxlib_version = "unknown"

    version_info = {
        "jax": jax.__version__,
        "jaxlib": jaxlib_version,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_grid": [
            {"seed": s, "batch": b, "steps": st, "n_wins": nw} for (s, b, st, nw) in _INPUT_GRID
        ],
    }

    artifacts = _artifacts()
    if not artifacts:
        print("error: no .npz artifacts found", flush=True)
        return 1

    manifest: dict[str, dict] = {}
    for artifact in artifacts:
        stem = artifact.stem
        print(f"generating golden for {stem} ...", flush=True)
        model = load_model(artifact, device=args.device, precision="float32")
        feat = model.config.feature
        inputs: list[tuple[np.ndarray, np.ndarray]] = []
        golden: dict[str, np.ndarray] = {}
        input_meta: list[dict] = []
        for idx, (seed, batch, steps, n_wins) in enumerate(_INPUT_GRID):
            x, nw = _make_inputs(seed, batch, steps, n_wins, feat.n_mels, feat.seg_length)
            inputs.append((x, nw))
            input_meta.append({
                "seed": seed, "batch": batch, "steps": steps, "n_wins": n_wins, "x_shape": list(x.shape)
            })
            out = model.predict_segments(x, nw)
            golden[f"out_{idx}"] = np.asarray(out, dtype=np.float32)
            # Key staged intermediates (cnn, time_dependency) for deeper parity.
            stages = model.predict_stages(x, nw)
            golden[f"cnn_{idx}"] = np.asarray(stages["cnn"], dtype=np.float32)
            golden[f"td_{idx}"] = np.asarray(stages["time_dependency"], dtype=np.float32)

        npz_path = out_dir / f"{stem}.golden.npz"
        np.savez(npz_path, **golden)
        json_path = out_dir / f"{stem}.golden.json"
        meta = {
            "artifact": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "model_name": model.config.model_name,
            "output_names": list(model.config.output_names),
            "inputs": input_meta,
            "n_inputs": len(inputs),
            "generator_version": 1,
            **version_info,
        }

        # Optional PyTorch reference provenance.
        if args.pytorch_ref is not None:
            ptref = _capture_ptref(artifact, inputs, args.pytorch_ref)
            if ptref is not None:
                ptref_path = out_dir / f"{stem}.ptref.npz"
                np.savez(ptref_path, **ptref)
                meta["ptref_npz"] = ptref_path.name
                meta["ptref_sha256"] = _sha256(ptref_path)
                # Record max abs diff JAX-vs-PyTorch as provenance evidence.
                max_diff = 0.0
                for i in range(len(inputs)):
                    diff = float(np.max(np.abs(golden[f"out_{i}"] - ptref[f"out_{i}"])))
                    max_diff = max(max_diff, diff)
                meta["jax_vs_pytorch_max_abs"] = max_diff
                print(f"  ptref captured: max|jax-pt| = {max_diff:.3e}")

        json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        manifest[stem] = {
            "golden_npz": npz_path.name,
            "golden_json": json_path.name,
            "golden_sha256": _sha256(npz_path),
            "artifact_sha256": meta["artifact_sha256"],
            "n_inputs": meta["n_inputs"],
            "ptref_npz": meta.get("ptref_npz"),
            "ptref_sha256": meta.get("ptref_sha256"),
            "jax_vs_pytorch_max_abs": meta.get("jax_vs_pytorch_max_abs"),
        }
        print(f"  wrote {npz_path.name} + {json_path.name}")

    # Top-level manifest listing every fixture + checksums.
    (out_dir / "GOLDEN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {len(artifacts)} golden fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
