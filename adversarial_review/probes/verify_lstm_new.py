"""Verify new fused BiLSTM: vs f64 truth, vs old impl (<=1e-6 reorder noise)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nisqa_jax.checkpoint import load_converted_checkpoint
import jax
import jax.numpy as jnp
from nisqa_jax.model import _bidirectional_lstm, _lstm_direction

cfg, params = load_converted_checkpoint(Path(__file__).resolve().parent / "weights/nisqa_tts.npz")
H = cfg.td_lstm_h
fwd = params["time_dependency"]["forward"]
rev = params["time_dependency"]["reverse"]
IN = fwd["w_ih"].shape[0]


def to_f64(d):
    return {k: v.astype(np.float64) for k, v in d.items()}


fwd64, rev64 = to_f64(fwd), to_f64(rev)


def lstm_dir_f64(p, x, reverse):
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
        gates = seq[t] + h @ p["w_hh"] + p["b_hh"]
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
    return np.concatenate([lstm_dir_f64(fwd64, x, False), lstm_dir_f64(rev64, x, True)], axis=-1)


# old impl (reference) for reorder-noise check: replicate original gate order
def bilstm_old(x_np):
    def dir_old(p, x, reverse):
        bsz, steps = x.shape[0], x.shape[1]
        def step(carry, item):
            h, c = carry
            x_t, valid = item
            gates = (jnp.matmul(x_t, p["w_ih"]) + jnp.matmul(h, p["w_hh"])
                     + p["b_ih"] + p["b_hh"])
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            c_new = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
            h_new = jax.nn.sigmoid(o) * jnp.tanh(c_new)
            valid = valid[:, None]
            h = jnp.where(valid, h_new, h)
            c = jnp.where(valid, c_new, c)
            out = jnp.where(valid, h, jnp.zeros_like(h))
            return (h, c), out
        time = jnp.arange(steps, dtype=jnp.int32)
        valid = time[None, :] < jnp.full((bsz,), steps, dtype=jnp.int32)[:, None]
        seq = jnp.swapaxes(x, 0, 1)
        valid_seq = jnp.swapaxes(valid, 0, 1)
        if reverse:
            seq = seq[::-1]; valid_seq = valid_seq[::-1]
        init = (jnp.zeros((bsz, H), dtype=x.dtype), jnp.zeros((bsz, H), dtype=x.dtype))
        _, out = jax.lax.scan(step, init, (seq, valid_seq), unroll=32)
        if reverse:
            out = out[::-1]
        return jnp.swapaxes(out, 0, 1)
    pj = {k: {kk: jnp.asarray(vv) for kk, vv in v.items()} for k, v in
          {"forward": fwd, "reverse": rev}.items()}
    xj = jnp.asarray(x_np)
    with jax.default_matmul_precision("float32"):
        fw = jax.jit(lambda: dir_old(pj["forward"], xj, False))()
        bw = jax.jit(lambda: dir_old(pj["reverse"], xj, True))()
    return np.asarray(jnp.concatenate([fw, bw], axis=-1))


def new_bilstm(x_np, device):
    p = {"forward": {k: jnp.asarray(v) for k, v in fwd.items()},
         "reverse": {k: jnp.asarray(v) for k, v in rev.items()}}
    xj = jnp.asarray(x_np)
    nj = jnp.full((x_np.shape[0],), x_np.shape[1], dtype=jnp.int32)
    with jax.default_matmul_precision("float32"):
        out = jax.jit(_bidirectional_lstm, backend=device)(p, xj, nj)
    return np.asarray(out)


def new_dir(x_np, reverse, device):
    p = {k: jnp.asarray(v) for k, v in (fwd if not reverse else rev).items()}
    xj = jnp.asarray(x_np)
    nj = jnp.full((x_np.shape[0],), x_np.shape[1], dtype=jnp.int32)
    with jax.default_matmul_precision("float32"):
        out = jax.jit(_lstm_direction, backend=device)(p, xj, nj, reverse=reverse)
    return np.asarray(out)


rng = np.random.default_rng(1234)
SHAPES = [(32, 64), (8, 128), (8, 6000), (2, 24), (1, 512), (16, 256)]

# Variant A: fma-friendly association, but OLD structure (two scans, no precompute).
# Isolates the Goal B structural change (precompute + fused scan) as pure reorder.
def bilstm_variant_a(x_np):
    def dir_a(p, x, reverse):
        bsz, steps = x.shape[0], x.shape[1]
        def step(carry, item):
            h, c = carry; x_t, valid = item
            gates = (jnp.matmul(x_t, p["w_ih"]) + p["b_ih"]) + (jnp.matmul(h, p["w_hh"]) + p["b_hh"])
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            c_new = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
            h_new = jax.nn.sigmoid(o) * jnp.tanh(c_new)
            valid = valid[:, None]
            h = jnp.where(valid, h_new, h); c = jnp.where(valid, c_new, c)
            out = jnp.where(valid, h, jnp.zeros_like(h))
            return (h, c), out
        time = jnp.arange(steps, dtype=jnp.int32)
        valid = time[None, :] < jnp.full((bsz,), steps, dtype=jnp.int32)[:, None]
        seq = jnp.swapaxes(x, 0, 1); valid_seq = jnp.swapaxes(valid, 0, 1)
        if reverse: seq = seq[::-1]; valid_seq = valid_seq[::-1]
        init = (jnp.zeros((bsz, H), dtype=x.dtype), jnp.zeros((bsz, H), dtype=x.dtype))
        _, out = jax.lax.scan(step, init, (seq, valid_seq), unroll=32)
        if reverse: out = out[::-1]
        return jnp.swapaxes(out, 0, 1)
    pj = {"forward": {k: jnp.asarray(v) for k, v in fwd.items()},
          "reverse": {k: jnp.asarray(v) for k, v in rev.items()}}
    xj = jnp.asarray(x_np)
    with jax.default_matmul_precision("float32"):
        fw = jax.jit(lambda: dir_a(pj["forward"], xj, False))()
        bw = jax.jit(lambda: dir_a(pj["reverse"], xj, True))()
    return np.asarray(jnp.concatenate([fw, bw], axis=-1))

print("=== Goal B structural delta: new(fused+precompute) vs variantA(fma,2-scan) ===")
for bsz, steps in SHAPES:
    x = (rng.normal(size=(bsz, steps, IN)) * 0.5).astype(np.float32)
    new_cpu = new_bilstm(x, "cpu")
    var_a = bilstm_variant_a(x)
    d = np.max(np.abs(new_cpu.astype(np.float64) - var_a.astype(np.float64)))
    print(f"  bs={bsz:<3} sl={steps:<5} max|new-varA|={d:.3e}")

print("\n=== New fused BiLSTM vs f64 truth (CPU) and vs old impl (reorder noise) ===")
print(f"{'shape':>14} {'new-f64(cpu)':>14} {'old-f64(cpu)':>14} {'new-old(cpu)':>14} {'new-f64(gpu)':>14}")
for bsz, steps in SHAPES:
    x = (rng.normal(size=(bsz, steps, IN)) * 0.5).astype(np.float32)
    truth = bilstm_f64(x.astype(np.float64))
    new_cpu = new_bilstm(x, "cpu")
    old_cpu = bilstm_old(x)
    new_gpu = new_bilstm(x, "cuda")
    en = np.max(np.abs(new_cpu.astype(np.float64) - truth))
    eo = np.max(np.abs(old_cpu.astype(np.float64) - truth))
    eno = np.max(np.abs(new_cpu.astype(np.float64) - old_cpu.astype(np.float64)))
    eng = np.max(np.abs(new_gpu.astype(np.float64) - truth))
    print(f"  bs={bsz:<3} sl={steps:<5} {en:14.3e} {eo:14.3e} {eno:14.3e} {eng:14.3e}")

# variable n_wins masking check: new fused must match old under masking
print("\n=== Masking parity: new fused vs old (variable n_wins) ===")
for bsz, steps in [(2, 24), (8, 64), (4, 128)]:
    x = (rng.normal(size=(bsz, steps, IN)) * 0.5).astype(np.float32)
    nw = np.sort(rng.integers(1, steps + 1, size=bsz)).astype(np.int32)[::-1]  # unsorted-ish
    # zero out invalid timesteps (as the pipeline does)
    for b in range(bsz):
        x[b, nw[b]:] = 0.0
    pj = {"forward": {k: jnp.asarray(v) for k, v in fwd.items()},
          "reverse": {k: jnp.asarray(v) for k, v in rev.items()}}
    xj = jnp.asarray(x); nj = jnp.asarray(nw)
    with jax.default_matmul_precision("float32"):
        new = np.asarray(jax.jit(_bidirectional_lstm, backend="cpu")(pj, xj, nj))
    # old with masking
    def dir_old(p, x, nw, reverse):
        bsz_, steps_ = x.shape[0], x.shape[1]
        def step(carry, item):
            h, c = carry; xt, valid = item
            gates = jnp.matmul(xt, p["w_ih"]) + jnp.matmul(h, p["w_hh"]) + p["b_ih"] + p["b_hh"]
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            c_new = jax.nn.sigmoid(f)*c + jax.nn.sigmoid(i)*jnp.tanh(g)
            h_new = jax.nn.sigmoid(o)*jnp.tanh(c_new)
            valid = valid[:, None]
            h = jnp.where(valid, h_new, h); c = jnp.where(valid, c_new, c)
            out = jnp.where(valid, h, jnp.zeros_like(h))
            return (h, c), out
        time = jnp.arange(steps_, dtype=nw.dtype)
        valid = time[None, :] < nw[:, None]
        seq = jnp.swapaxes(x, 0, 1); valid_seq = jnp.swapaxes(valid, 0, 1)
        if reverse: seq = seq[::-1]; valid_seq = valid_seq[::-1]
        init = (jnp.zeros((bsz_, H), dtype=x.dtype), jnp.zeros((bsz_, H), dtype=x.dtype))
        _, out = jax.lax.scan(step, init, (seq, valid_seq), unroll=32)
        if reverse: out = out[::-1]
        return jnp.swapaxes(out, 0, 1)
    with jax.default_matmul_precision("float32"):
        fw = dir_old(pj["forward"], xj, nj, False)
        bw = dir_old(pj["reverse"], xj, nj, True)
        old = np.asarray(jnp.concatenate([fw, bw], axis=-1))
    d = np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))
    print(f"  bs={bsz:<3} sl={steps:<5} nw={nw.tolist()} max|new-old|={d:.3e}")
