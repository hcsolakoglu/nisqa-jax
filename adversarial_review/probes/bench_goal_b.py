"""Goal B benchmark: TTS BiLSTM JAX (new impl) vs PT eager + PT compile.

Methodology (reuses adversarial_review/results/decisive_benchmark_optimized_pytorch.txt):
- JAX: jitted forward + block_until_ready, inputs pre-staged on-GPU. Median of 60
  timed iters after warmup; compile excluded.
- PT: numerically-identical pack/pad bypass (run CNN/LSTM on the full cropped
  sequence + mask invalid timesteps; identical to stock when n_wins=steps, which
  is the case here). Capturable by torch.compile (no pack_padded_sequence).
  - eager: cudnn.benchmark=True, eval, no_grad, TF32 off.
  - compile: torch.compile(mode="max-autotune").
- Grid: bs {1,8,16} x steps {64,256,512}. Speedup = PT_latency / JAX_latency.
"""
from __future__ import annotations

import importlib
import os
import statistics
import sys
import time
from pathlib import Path

# Limit JAX GPU preallocation so PyTorch (cuDNN LSTM) has room on the 8GB RTX 3070.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nisqa pytorch"))

import jax
from nisqa_jax.checkpoint import load_model

torch = importlib.import_module("torch")
nl = importlib.import_module("nisqa.NISQA_lib")


def _model_args(args):
    keys = ["ms_seg_length","ms_n_mels","cnn_model","cnn_c_out_1","cnn_c_out_2",
            "cnn_c_out_3","cnn_kernel_size","cnn_dropout","cnn_pool_1","cnn_pool_2",
            "cnn_pool_3","cnn_fc_out_h","td","td_sa_d_model","td_sa_nhead","td_sa_pos_enc",
            "td_sa_num_layers","td_sa_h","td_sa_dropout","td_lstm_h","td_lstm_num_layers",
            "td_lstm_dropout","td_lstm_bidirectional","td_2","td_2_sa_d_model","td_2_sa_nhead",
            "td_2_sa_pos_enc","td_2_sa_num_layers","td_2_sa_h","td_2_sa_dropout",
            "td_2_lstm_h","td_2_lstm_num_layers","td_2_lstm_dropout","td_2_lstm_bidirectional",
            "pool","pool_att_h","pool_att_dropout"]
    return {k: args[k] for k in keys}


def build_pt_model(device):
    ck = torch.load(ROOT / "nisqa pytorch" / "weights" / "nisqa_tts.tar", map_location="cpu")
    args = ck["args"]
    cls = {"NISQA": nl.NISQA, "NISQA_DIM": nl.NISQA_DIM}[args["model"]]
    m = cls(**_model_args(args))
    m.load_state_dict(ck["model_state_dict"], strict=True)
    return m.to(device).eval(), args


def bypass_forward(model, x, n_wins):
    """Pack/pad-free forward: CNN + BiLSTM on the full cropped sequence + mask.

    Identical to stock pack/pad when n_wins == steps for all rows (no padding).
    """
    bs, length = x.shape[0], x.shape[1]
    valid = torch.arange(length, device=x.device)[None, :] < n_wins[:, None]
    # CNN: run inner conv net on the flattened batch, restore [bs, length, fan_out].
    x_flat = x.reshape(bs * length, *x.shape[2:])
    cnn_out = model.cnn.model(x_flat).reshape(bs, length, -1)
    cnn_out = cnn_out * valid[:, :, None]
    # BiLSTM directly on the full sequence (cuDNN fused kernel).
    td_out, _ = model.time_dependency.model.lstm(cnn_out)
    td_out = td_out * valid[:, :, None]
    # td_2 == skip for TTS; pool == last_step_bi.
    return model.pool(td_out, n_wins)


def median_lat(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    return statistics.median(ts)


def median_lat_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - s)
    return statistics.median(ts)


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    jax_model = load_model(ROOT / "weights" / "nisqa_tts.npz", device="gpu",
                           cache_dir=ROOT / ".jax_cache")
    feat = jax_model.config.feature
    pt_eager, _ = build_pt_model(device)
    # Compile the pack/pad-free bypass (stock pack/pad is not dynamo-capturable).
    pt_compiled = torch.compile(lambda x, n: bypass_forward(pt_eager, x, n),
                                mode="max-autotune", fullgraph=False)

    rng = np.random.default_rng(0)
    GRID = [(1, 64), (1, 256), (1, 512), (8, 64), (8, 256), (8, 512),
            (16, 64), (16, 256), (16, 512)]
    WARM, ITERS = 20, 60

    # parity check: bypass eager vs stock eager at one shape
    x0 = rng.normal(size=(8, 256, 1, feat.n_mels, feat.seg_length)).astype(np.float32)
    n0 = np.full((8,), 256, dtype=np.int32)
    with torch.no_grad():
        xt0 = torch.from_numpy(x0).to(device); nt0 = torch.from_numpy(n0).to(device)
        stock = pt_eager(xt0, nt0)
        bp = bypass_forward(pt_eager, xt0, nt0)
        print(f"bypass vs stock parity (bs=8 sl=256): max abs diff = "
              f"{(stock-bp).abs().max().item():.3e}\n")

    print(f"{'bs':>3} {'steps':>5} {'JAX(ms)':>10} {'PTeager(ms)':>12} {'PTcompile(ms)':>14} "
          f"{'sp/eager':>9} {'sp/compile':>11}")
    print("-" * 80)
    sp_e, sp_c = [], []
    for bsz, steps in GRID:
        x = rng.normal(size=(bsz, steps, 1, feat.n_mels, feat.seg_length)).astype(np.float32)
        n = np.full((bsz,), steps, dtype=np.int32)
        xj, nj = jax_model.device_segments(x, n)
        jax_model._forward(jax_model._compute_params, xj, nj).block_until_ready()
        jax_lat = median_lat(lambda: jax_model._forward(jax_model._compute_params, xj, nj).block_until_ready(), WARM, ITERS)
        xt = torch.from_numpy(x).to(device)
        nt = torch.from_numpy(n).to(device)
        with torch.no_grad():
            bypass_forward(pt_eager, xt, nt)
            eager_lat = median_lat_cuda(lambda: bypass_forward(pt_eager, xt, nt), WARM, ITERS)
            pt_compiled(xt, nt)
            comp_lat = median_lat_cuda(lambda: pt_compiled(xt, nt), WARM, ITERS)
        se = eager_lat / jax_lat
        sc = comp_lat / jax_lat
        sp_e.append(se); sp_c.append(sc)
        print(f"{bsz:>3} {steps:>5} {jax_lat*1e3:>10.3f} {eager_lat*1e3:>12.3f} {comp_lat*1e3:>14.3f} "
              f"{se:>8.2f}x {sc:>10.2f}x")

    geomean = statistics.geometric_mean
    print("-" * 80)
    print(f"geomean speedup vs eager:   {geomean(sp_e):.2f}x  (min {min(sp_e):.2f}, max {max(sp_e):.2f})")
    print(f"geomean speedup vs compile:  {geomean(sp_c):.2f}x  (min {min(sp_c):.2f}, max {max(sp_c):.2f})")


if __name__ == "__main__":
    main()
