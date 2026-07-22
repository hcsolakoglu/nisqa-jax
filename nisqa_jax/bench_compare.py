from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

from .checkpoint import load_model


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


def _torch_model(checkpoint: Path, device):
    sys.path.insert(0, str(checkpoint.parents[1]))
    from nisqa import NISQA_lib as NL

    torch = _torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for reference comparison")
    ck = torch.load(checkpoint, map_location="cpu")
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
    parser.add_argument("--cache_dir")
    args = parser.parse_args()
    if args.transfer_guard != "allow":
        jax.config.update("jax_transfer_guard", args.transfer_guard)

    checkpoint = Path(args.pretrained_model).resolve()
    jax_device_selector = args.device
    if jax_device_selector is None and any(device.platform == "gpu" for device in jax.devices()):
        jax_device_selector = "gpu"
    jax_model = load_model(checkpoint, device=jax_device_selector, cache_dir=args.cache_dir, precision=args.precision)

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

    x_jax, n_jax = jax_model.device_segments(x_np, n_np)

    compile_start = time.perf_counter()
    jax_model._forward(jax_model._compute_params, x_jax, n_jax).block_until_ready()
    jax_compile_seconds = time.perf_counter() - compile_start
    jax_seconds = _time_jax(lambda: jax_model._forward(jax_model._compute_params, x_jax, n_jax), args.steps)

    torch_seconds = None
    speedup = None
    torch = _torch()
    torch_checkpoint = checkpoint if checkpoint.suffix == ".tar" else jax_model.config.source_path
    compare_torch = (
        not args.no_torch
        and torch is not None
        and torch.cuda.is_available()
        and jax_model.device.platform == "gpu"
        and Path(torch_checkpoint).exists()
    )
    if compare_torch:
        torch_device = torch.device("cuda")
        torch_model, _ = _torch_model(Path(torch_checkpoint), torch_device)
        x_torch = torch.from_numpy(x_np).to(torch_device)
        n_torch = torch.from_numpy(n_np).to(torch_device)
        with torch.no_grad():
            torch_model(x_torch, n_torch)
            torch_seconds = _time_cuda(torch, lambda: torch_model(x_torch, n_torch), args.steps)
        speedup = torch_seconds / jax_seconds

    result = {
        "checkpoint": checkpoint.name,
        "jax_device": str(jax_model.device),
        "precision": jax_model.precision,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "jax_compile_seconds": jax_compile_seconds,
        "jax_forward_seconds": jax_seconds,
        "jax_forward_latency_seconds": jax_seconds / args.steps,
        "jax_samples_per_second": args.batch_size * args.steps / jax_seconds,
        "torch_comparison_enabled": compare_torch,
        "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "torch_forward_seconds": torch_seconds,
        "speedup": speedup,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.min_speedup is not None and speedup is None:
        raise SystemExit("--min_speedup requires a PyTorch CUDA comparison")
    if args.min_speedup is not None and speedup is not None and speedup < args.min_speedup:
        raise SystemExit(f"speedup {speedup:.3f} < required {args.min_speedup:.3f}")


if __name__ == "__main__":
    main()
