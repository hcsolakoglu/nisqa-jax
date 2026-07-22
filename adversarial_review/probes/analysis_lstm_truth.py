"""Goal A analysis: f64 ground-truth LSTM vs JAX-cpu / PT-cpu / PT-cuDNN-gpu.

Isolates the BiLSTM accumulation by feeding the LSTM directly (bypassing CNN)
with realistic-distribution inputs at the failing shapes. n_wins = steps for all
rows so masking is a no-op and we measure pure recurrence accumulation drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nisqa_jax.checkpoint import load_converted_checkpoint

import jax
import jax.numpy as jnp

cfg, params = load_converted_checkpoint(Path(__file__).resolve().parent / "weights/nisqa_tts.npz")
H = cfg.td_lstm_h  # 128
fwd = params["time_dependency"]["forward"]
rev = params["time_dependency"]["reverse"]
IN = fwd["w_ih"].shape[0]  # 20


def to_f64(d):
    return {k: v.astype(np.float64) for k, v in d.items()}


fwd64 = to_f64(fwd)
rev64 = to_f64(rev)


def lstm_dir_f64(p, x, reverse):
    """Exact JAX _lstm_direction algorithm in float64 (numpy)."""
    bsz, steps = x.shape[0], x.shape[1]
    xproj = x @ p["w_ih"] + p["b_ih"]
    seq = np.swapaxes(xproj, 0, 1)
    if reverse:
        seq = seq[::-1]
    h = np.zeros((bsz, H), dtype=np.float64)
    c = np.zeros((bsz, H), dtype=np.float64)
    outs = []
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    for t in range(steps):
        xt = seq[t]
        gates = xt + h @ p["w_hh"] + p["b_hh"]
        i, f, g, o = np.split(gates, 4, axis=-1)
        c_new = sig(f) * c + sig(i) * np.tanh(g)
        h_new = sig(o) * np.tanh(c_new)
        h, c = h_new, c_new
        outs.append(h)
    out = np.stack(outs, axis=0)
    if reverse:
        out = out[::-1]
    return np.swapaxes(out, 0, 1)


def bilstm_f64(x):
    fw = lstm_dir_f64(fwd64, x, reverse=False)
    bw = lstm_dir_f64(rev64, x, reverse=True)
    return np.concatenate([fw, bw], axis=-1)


def jax_bilstm_f32(x_np, device):
    from nisqa_jax.model import _bidirectional_lstm
    p = {
        "forward": {k: jnp.asarray(v) for k, v in fwd.items()},
        "reverse": {k: jnp.asarray(v) for k, v in rev.items()},
    }
    xj = jnp.asarray(x_np)
    nj = jnp.full((x_np.shape[0],), x_np.shape[1], dtype=jnp.int32)
    with jax.default_matmul_precision("float32"):
        out = jax.jit(_bidirectional_lstm, backend=device)(p, xj, nj)
    return np.asarray(out)


def build_torch_lstm():
    import torch.nn as nn
    lstm = nn.LSTM(input_size=IN, hidden_size=H, num_layers=1,
                   batch_first=True, bidirectional=True)
    sd = {
        "weight_ih_l0": torch.from_numpy(fwd["w_ih"].T.copy()),
        "weight_hh_l0": torch.from_numpy(fwd["w_hh"].T.copy()),
        "bias_ih_l0": torch.from_numpy(fwd["b_ih"].copy()),
        "bias_hh_l0": torch.from_numpy(fwd["b_hh"].copy()),
        "weight_ih_l0_reverse": torch.from_numpy(rev["w_ih"].T.copy()),
        "weight_hh_l0_reverse": torch.from_numpy(rev["w_hh"].T.copy()),
        "bias_ih_l0_reverse": torch.from_numpy(rev["b_ih"].copy()),
        "bias_hh_l0_reverse": torch.from_numpy(rev["b_hh"].copy()),
    }
    lstm.load_state_dict(sd)
    return lstm.eval()


def pt_bilstm(x_np, device, dtype=torch.float32):
    lstm = build_torch_lstm().to(device).to(dtype)
    x = torch.from_numpy(x_np).to(device).to(dtype)
    with torch.no_grad():
        out, _ = lstm(x)
    return out.cpu().numpy()


SHAPES = [(32, 64), (8, 128), (8, 6000), (2, 24)]
rng = np.random.default_rng(1234)

print(f"{'shape':>14} {'JAXcpu':>12} {'PTcpu':>12} {'PTgpu(cuDNN)':>14}   (max|.-f64|)")
print("-" * 72)
for bsz, steps in SHAPES:
    x = (rng.normal(size=(bsz, steps, IN)) * 0.5).astype(np.float32)
    truth = bilstm_f64(x.astype(np.float64))
    jax_cpu = jax_bilstm_f32(x, "cpu")
    pt_cpu = pt_bilstm(x, "cpu")
    pt_gpu = pt_bilstm(x, "cuda")
    ej = np.max(np.abs(jax_cpu.astype(np.float64) - truth))
    ep = np.max(np.abs(pt_cpu.astype(np.float64) - truth))
    eg = np.max(np.abs(pt_gpu.astype(np.float64) - truth))
    print(f"  bs={bsz:<3} sl={steps:<5} {ej:12.3e} {ep:12.3e} {eg:14.3e}")

print("\nPairwise (max abs diff):")
print(f"{'shape':>14} {'JAXvsPTcpu':>12} {'JAXvsPTgpu':>12} {'PTcpuvsPTgpu':>14}")
for bsz, steps in SHAPES:
    x = (rng.normal(size=(bsz, steps, IN)) * 0.5).astype(np.float32)
    jax_cpu = jax_bilstm_f32(x, "cpu")
    pt_cpu = pt_bilstm(x, "cpu")
    pt_gpu = pt_bilstm(x, "cuda")
    jjp = np.max(np.abs(jax_cpu.astype(np.float64) - pt_cpu.astype(np.float64)))
    jjg = np.max(np.abs(jax_cpu.astype(np.float64) - pt_gpu.astype(np.float64)))
    ppg = np.max(np.abs(pt_cpu.astype(np.float64) - pt_gpu.astype(np.float64)))
    print(f"  bs={bsz:<3} sl={steps:<5} {jjp:12.3e} {jjg:12.3e} {ppg:14.3e}")
