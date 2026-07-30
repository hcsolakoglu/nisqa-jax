from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .checkpoint import _load_torch_checkpoint, load_model


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _model_args(args: dict) -> dict:
    keys = [
        "ms_seg_length",
        "ms_n_mels",
        "cnn_model",
        "cnn_c_out_1",
        "cnn_c_out_2",
        "cnn_c_out_3",
        "cnn_kernel_size",
        "cnn_dropout",
        "cnn_pool_1",
        "cnn_pool_2",
        "cnn_pool_3",
        "cnn_fc_out_h",
        "td",
        "td_sa_d_model",
        "td_sa_nhead",
        "td_sa_pos_enc",
        "td_sa_num_layers",
        "td_sa_h",
        "td_sa_dropout",
        "td_lstm_h",
        "td_lstm_num_layers",
        "td_lstm_dropout",
        "td_lstm_bidirectional",
        "td_2",
        "td_2_sa_d_model",
        "td_2_sa_nhead",
        "td_2_sa_pos_enc",
        "td_2_sa_num_layers",
        "td_2_sa_h",
        "td_2_sa_dropout",
        "td_2_lstm_h",
        "td_2_lstm_num_layers",
        "td_2_lstm_dropout",
        "td_2_lstm_bidirectional",
        "pool",
        "pool_att_h",
        "pool_att_dropout",
    ]
    return {key: args[key] for key in keys}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_torch_checkpoint(jax_checkpoint: Path, explicit: str | None) -> Path | None:
    """Resolve only caller-provided references, never conversion-host paths."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if jax_checkpoint.suffix == ".tar":
        return jax_checkpoint
    return None


def _verify_torch_checkpoint(checkpoint: Path, expected_sha256: str) -> str:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"PyTorch comparison checkpoint not found: {checkpoint}. "
            "Pass --torch_checkpoint with the matching original NISQA .tar file."
        )
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"PyTorch checkpoint SHA-256 mismatch for {checkpoint}: "
            f"expected {expected_sha256}, got {actual_sha256}. "
            "Use the exact source checkpoint recorded by the converted artifact."
        )
    return actual_sha256


def _comparison_unavailable_reason(
    *,
    requested: bool,
    disabled: bool,
    precision: str,
    torch_installed: bool,
    torch_cuda_available: bool,
    jax_platform: str,
) -> str | None:
    if not requested:
        return (
            "disabled by --no_torch"
            if disabled
            else "not requested; pass --torch_checkpoint with the matching original .tar"
        )
    if precision != "float32":
        return (
            "PyTorch comparison supports only --precision float32 so both frameworks use the same dtype; "
            "use --no_torch for a JAX-only bf16 benchmark"
        )
    if not torch_installed:
        return "PyTorch is not installed; install the conversion extra with pip install -e '.[convert]'"
    if not torch_cuda_available:
        return "PyTorch CUDA is unavailable; install a CUDA-enabled PyTorch build and verify torch.cuda.is_available()"
    if jax_platform != "gpu":
        return f"JAX model is on {jax_platform!r}, not GPU; install CUDA JAX and pass --device gpu"
    return None


def _torch_model(checkpoint: Path, device: Any, source_root: Path | None = None):
    source_root = source_root or checkpoint.parents[1]
    source_module = source_root / "nisqa" / "NISQA_lib.py"
    if not source_module.is_file():
        raise FileNotFoundError(
            f"PyTorch NISQA source not found at {source_module}. "
            "Pass --torch_source_root pointing to a gabrielmittag/NISQA checkout."
        )
    sys.path.insert(0, str(source_root))
    from nisqa import NISQA_lib as NL

    torch = _torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for reference comparison")
    ck = _load_torch_checkpoint(torch, checkpoint)
    cls = {"NISQA": NL.NISQA, "NISQA_DIM": NL.NISQA_DIM}[ck["args"]["model"]]
    model = cls(**_model_args(ck["args"]))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.to(device).eval(), ck["args"]


def _time_cuda(torch, fn, steps: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _time_jax(fn, steps: int) -> float:
    start = time.perf_counter()
    for _ in range(steps):
        fn().block_until_ready()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=["float32", "bf16"], default="float32")
    parser.add_argument("--min_speedup", type=float)
    parser.add_argument("--transfer_guard", choices=["allow", "log", "disallow"], default="allow")
    parser.add_argument("--no_torch", action="store_true")
    parser.add_argument(
        "--torch_checkpoint",
        help="original PyTorch .tar to compare; its SHA-256 must match the JAX artifact metadata",
    )
    parser.add_argument(
        "--torch_source_root",
        help="gabrielmittag/NISQA checkout containing nisqa/NISQA_lib.py "
        "(defaults to the checkpoint's parent repository)",
    )
    parser.add_argument("--cache_dir")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.seq_len < 1:
        raise ValueError("--seq_len must be >= 1")
    if args.no_torch and args.torch_checkpoint is not None:
        raise ValueError("--no_torch and --torch_checkpoint are mutually exclusive")
    if args.transfer_guard != "allow":
        jax.config.update("jax_transfer_guard", args.transfer_guard)

    checkpoint = Path(args.pretrained_model).resolve()
    if args.torch_source_root and args.no_torch:
        raise ValueError("--torch_source_root cannot be used with --no_torch")
    if args.torch_source_root and args.torch_checkpoint is None and checkpoint.suffix != ".tar":
        raise ValueError("--torch_source_root requires --torch_checkpoint for a converted JAX artifact")
    if args.min_speedup is not None and args.no_torch:
        raise ValueError("--min_speedup cannot be used with --no_torch")
    if args.min_speedup is not None and args.torch_checkpoint is None and checkpoint.suffix != ".tar":
        raise ValueError("--min_speedup requires --torch_checkpoint for a converted JAX artifact")
    jax_device_selector = args.device
    if jax_device_selector is None and any(device.platform == "gpu" for device in jax.devices()):
        jax_device_selector = "gpu"
    jax_model = load_model(checkpoint, device=jax_device_selector, cache_dir=args.cache_dir, precision=args.precision)

    torch_seconds = None
    torch_transfer_seconds = None
    torch_first_forward_seconds = None
    torch_input_dtype = None
    torch_parameter_dtype = None
    torch_checkpoint_sha256 = None
    speedup = None
    torch_checkpoint = _resolve_torch_checkpoint(checkpoint, args.torch_checkpoint)
    comparison_requested = not args.no_torch and torch_checkpoint is not None
    torch = _torch() if comparison_requested else None
    if comparison_requested:
        assert torch_checkpoint is not None
        torch_checkpoint_sha256 = _verify_torch_checkpoint(
            torch_checkpoint,
            jax_model.config.source_sha256,
        )
    comparison_unavailable_reason = _comparison_unavailable_reason(
        requested=comparison_requested,
        disabled=args.no_torch,
        precision=args.precision,
        torch_installed=torch is not None,
        torch_cuda_available=bool(torch is not None and torch.cuda.is_available()),
        jax_platform=jax_model.device.platform,
    )
    if comparison_requested and comparison_unavailable_reason is not None:
        raise SystemExit(f"PyTorch comparison requested but unavailable: {comparison_unavailable_reason}")
    compare_torch = comparison_requested and comparison_unavailable_reason is None

    rng = np.random.default_rng(0)
    x_np = rng.normal(
        size=(
            args.batch_size,
            args.seq_len,
            1,
            jax_model.config.feature.n_mels,
            jax_model.config.feature.seg_length,
        ),
    ).astype(np.float32)
    n_np = np.full((args.batch_size,), args.seq_len, dtype=np.int32)

    transfer_start = time.perf_counter()
    x_jax, n_jax = jax_model.device_segments(x_np, n_np)
    x_jax.block_until_ready()
    n_jax.block_until_ready()
    jax_transfer_seconds = time.perf_counter() - transfer_start

    first_forward_start = time.perf_counter()
    jax_model._forward(jax_model._compute_params, x_jax, n_jax).block_until_ready()
    jax_first_forward_seconds = time.perf_counter() - first_forward_start
    jax_seconds = _time_jax(lambda: jax_model._forward(jax_model._compute_params, x_jax, n_jax), args.steps)

    if compare_torch:
        assert torch is not None
        assert torch_checkpoint is not None
        torch_device = torch.device("cuda")
        source_root = Path(args.torch_source_root).expanduser().resolve() if args.torch_source_root else None
        torch_model, _ = _torch_model(torch_checkpoint, torch_device, source_root)

        torch.cuda.synchronize()
        transfer_start = time.perf_counter()
        x_torch = torch.from_numpy(x_np).to(torch_device)
        n_torch = torch.from_numpy(n_np).to(torch_device)
        torch.cuda.synchronize()
        torch_transfer_seconds = time.perf_counter() - transfer_start
        torch_input_dtype = str(x_torch.dtype)
        torch_parameter_dtype = str(next(torch_model.parameters()).dtype)
        with torch.no_grad():
            torch.cuda.synchronize()
            first_forward_start = time.perf_counter()
            torch_model(x_torch, n_torch)
            torch.cuda.synchronize()
            torch_first_forward_seconds = time.perf_counter() - first_forward_start
            torch_seconds = _time_cuda(torch, lambda: torch_model(x_torch, n_torch), args.steps)
        speedup = torch_seconds / jax_seconds

    result = {
        "checkpoint": checkpoint.name,
        "jax_device": str(jax_model.device),
        "precision": jax_model.precision,
        "jax_input_dtype": str(x_jax.dtype),
        "jax_compute_dtype": str(x_jax.dtype),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "jax_host_to_device_seconds": jax_transfer_seconds,
        # Includes compilation or a persistent-cache lookup plus one forward.
        "jax_first_forward_seconds": jax_first_forward_seconds,
        "jax_warmed_forward_seconds": jax_seconds,
        "jax_warmed_forward_latency_seconds": jax_seconds / args.steps,
        "jax_samples_per_second": args.batch_size * args.steps / jax_seconds,
        "torch_comparison_requested": comparison_requested,
        "torch_comparison_enabled": compare_torch,
        "torch_comparison_unavailable_reason": comparison_unavailable_reason,
        "torch_cuda_available": bool(torch.cuda.is_available()) if torch is not None else None,
        "torch_checkpoint": str(torch_checkpoint) if torch_checkpoint is not None else None,
        "torch_checkpoint_sha256": torch_checkpoint_sha256,
        "torch_input_dtype": torch_input_dtype,
        "torch_parameter_dtype": torch_parameter_dtype,
        "torch_host_to_device_seconds": torch_transfer_seconds,
        "torch_first_forward_seconds": torch_first_forward_seconds,
        "torch_warmed_forward_seconds": torch_seconds,
        "comparison_precision": "float32" if compare_torch else None,
        "speedup": speedup,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.min_speedup is not None and speedup is None:
        raise SystemExit("--min_speedup requires a PyTorch CUDA comparison")
    if args.min_speedup is not None and speedup is not None and speedup < args.min_speedup:
        raise SystemExit(f"speedup {speedup:.3f} < required {args.min_speedup:.3f}")


if __name__ == "__main__":
    main()
