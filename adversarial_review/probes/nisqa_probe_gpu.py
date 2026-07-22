#!/usr/bin/env python3
"""Adversarial probe on GPU (CUDA JAX). Read-only: imports the package, never edits it."""
import os, sys, json, time, tempfile, traceback
from pathlib import Path
import numpy as np

ROOT = Path("/media/mithex/NVME 2/Codex Linux/NISQA PORT PROJECT")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nisqa pytorch"))
WEIGHTS = ROOT / "weights"
PT_WEIGHTS = ROOT / "nisqa pytorch" / "weights"

import jax
import jax.numpy as jnp
import torch
import nisqa.NISQA_lib as NL
from nisqa_jax.checkpoint import load_model
from nisqa_jax import predict as P
from nisqa_jax import features as F
import soundfile as sf

print("JAX", jax.__version__, "devices", jax.devices())
GPU = [d for d in jax.devices() if d.platform == "gpu"]
assert GPU, "No GPU device!"
DEV = "gpu"

ART = {"mos": WEIGHTS/"nisqa_mos_only.npz", "dim": WEIGHTS/"nisqa.npz", "tts": WEIGHTS/"nisqa_tts.npz"}
PT = {"mos": PT_WEIGHTS/"nisqa_mos_only.tar", "dim": PT_WEIGHTS/"nisqa.tar", "tts": PT_WEIGHTS/"nisqa_tts.tar"}
res=[]
def rep(n,ok,d): res.append((n,ok,d)); print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")

def pt_args(args):
    keys=["ms_seg_length","ms_n_mels","cnn_model","cnn_c_out_1","cnn_c_out_2","cnn_c_out_3","cnn_kernel_size",
          "cnn_dropout","cnn_pool_1","cnn_pool_2","cnn_pool_3","cnn_fc_out_h","td","td_sa_d_model","td_sa_nhead",
          "td_sa_pos_enc","td_sa_num_layers","td_sa_h","td_sa_dropout","td_lstm_h","td_lstm_num_layers",
          "td_lstm_dropout","td_lstm_bidirectional","td_2","td_2_sa_d_model","td_2_sa_nhead","td_2_sa_pos_enc",
          "td_2_sa_num_layers","td_2_sa_h","td_2_sa_dropout","td_2_lstm_h","td_2_lstm_num_layers","td_2_lstm_dropout",
          "td_2_lstm_bidirectional","pool","pool_att_h","pool_att_dropout"]
    return {k:args[k] for k in keys if k in args}
def pt_model(key):
    ck=torch.load(PT[key],map_location="cpu"); a=ck["args"]
    cls={"NISQA":NL.NISQA,"NISQA_DIM":NL.NISQA_DIM}[a["model"]]
    m=cls(**pt_args(a)); m.load_state_dict(ck["model_state_dict"],strict=True); m.eval(); return m,a
def synth(model, batch=2, steps=24, seed=0):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(batch,steps,1,model.config.feature.n_mels,model.config.feature.seg_length)).astype(np.float32)
    n=np.full((batch,),steps,dtype=np.int32); n[0]=max(1,steps-5); x[0,n[0]:]=0.0
    return x,n

# ---------- 1. GPU parity vs PyTorch CPU reference ----------
for key in ART:
    try:
        jm=load_model(ART[key], device=DEV)
        x,n=synth(jm)
        act=jm.predict_segments(x,n)
        ptm,a=pt_model(key)
        with torch.no_grad():
            exp=ptm(torch.from_numpy(x),torch.from_numpy(n)).numpy()
        md=float(np.max(np.abs(act-exp)))
        rep(f"gpu_parity_vs_pt:{key}", md<1e-3, f"max_abs={md:.2e} (JAX-GPU vs PT-CPU) device={jm.device}")
    except Exception as e:
        traceback.print_exc(); rep(f"gpu_parity_vs_pt:{key}", False, f"{type(e).__name__}: {e}")

# ---------- 2. GPU vs CPU JAX self-consistency (float32) ----------
for key in ART:
    try:
        jg=load_model(ART[key], device="gpu")
        jc=load_model(ART[key], device="cpu")
        x,n=synth(jg, steps=20)
        og=jg.predict_segments(x,n); oc=jc.predict_segments(x,n)
        md=float(np.max(np.abs(og-oc)))
        rep(f"gpu_vs_cpu_jax:{key}", md<1e-4, f"max_abs={md:.2e}")
    except Exception as e:
        traceback.print_exc(); rep(f"gpu_vs_cpu_jax:{key}", False, str(e))

# ---------- 3. bf16 on GPU ----------
for key in ART:
    try:
        mf=load_model(ART[key], device=DEV, precision="float32")
        mb=load_model(ART[key], device=DEV, precision="bf16")
        x,n=synth(mf, steps=32)
        ef=mf.predict_segments(x,n); eb=mb.predict_segments(x,n)
        drift=float(np.max(np.abs(ef-eb)))
        rep(f"gpu_bf16:{key}", np.isfinite(eb).all() and drift<0.5, f"finite={np.isfinite(eb).all()} drift={drift:.4f}")
    except Exception as e:
        traceback.print_exc(); rep(f"gpu_bf16:{key}", False, f"{type(e).__name__}: {e}")

# ---------- 4. TF32 matmul precision (default on GPU) ----------
try:
    jm=load_model(ART["dim"], device=DEV, precision="float32")
    x,n=synth(jm, steps=40)
    # default matmul precision on Ampere is 'tensorfloat32' unless overridden
    with jax.default_matmul_precision("float32"):
        strict=jm.predict_segments(x,n)
    # the model __post_init__ already wraps in default_matmul_precision("float32") -> strict
    with jax.default_matmul_precision("tensorfloat32"):
        # rebuild a tf32 forward by re-jitting without the strict wrapper
        from nisqa_jax.model import forward
        cp=jm._compute_params
        fwd_tf32=jax.jit(lambda p,x,n: forward(p,x,n,cfg=jm.config))
        out_tf32=np.asarray(fwd_tf32(cp, *jm.device_segments(x,n)).block_until_ready())
    drift=float(np.max(np.abs(strict-out_tf32)))
    rep("gpu_tf32_vs_f32", drift<1e-2, f"max_abs_drift={drift:.4f} (TF32 faster but less precise on Ampere)")
except Exception as e:
    traceback.print_exc(); rep("gpu_tf32_vs_f32", False, f"{type(e).__name__}: {e}")

# ---------- 5. transfer_guard: detect implicit host/device transfers ----------
for key in ART:
    try:
        jm=load_model(ART[key], device=DEV)
        x,n=synth(jm, steps=16)
        logs=[]
        import io, contextlib
        with jax.transfer_guard("log"):
            with contextlib.redirect_stderr(io.StringIO()) as buf:
                jm.predict_segments(x,n)
            logs.append(buf.getvalue())
        has_transfer = any("transfer" in l.lower() for l in logs)
        rep(f"gpu_transfer_guard:{key}", True, f"implicit_transfers_detected={has_transfer} (should be False for clean impl)")
    except Exception as e:
        rep(f"gpu_transfer_guard:{key}", False, f"{type(e).__name__}: {e}")

# ---------- 6. GPU throughput (warmed forward, model-only) ----------
for key in ART:
    try:
        jm=load_model(ART[key], device=DEV)
        cfg=jm.config.feature
        for bs,sl in [(1,128),(8,128),(16,128)]:
            x=np.zeros((bs,sl,1,cfg.n_mels,cfg.seg_length),dtype=np.float32)
            n=np.full((bs,),sl,dtype=np.int32)
            xd,nd=jm.device_segments(x,n)
            # warmup
            for _ in range(3): jm._forward(jm._compute_params,xd,nd).block_until_ready()
            t0=time.perf_counter()
            steps=30
            for _ in range(steps): jm._forward(jm._compute_params,xd,nd).block_until_ready()
            el=time.perf_counter()-t0
            sps=bs*steps/el
            print(f"  gpu_throughput:{key} bs={bs} sl={sl}: {sps:.1f} samples/s  latency={el/steps*1000:.2f}ms")
        rep(f"gpu_throughput:{key}", True, "see above")
    except Exception as e:
        traceback.print_exc(); rep(f"gpu_throughput:{key}", False, str(e))

# ---------- 7. GPU compile time + recompile cost ----------
try:
    jm=load_model(ART["mos"], device=DEV)
    cfg=jm.config.feature
    x=np.zeros((8,128,1,cfg.n_mels,cfg.seg_length),dtype=np.float32); n=np.full((8,),128,dtype=np.int32)
    xd,nd=jm.device_segments(x,n)
    t0=time.perf_counter(); jm._forward(jm._compute_params,xd,nd).block_until_ready(); c1=time.perf_counter()-t0
    # new shape -> recompile
    x2=np.zeros((8,256,1,cfg.n_mels,cfg.seg_length),dtype=np.float32); n2=np.full((8,),256,dtype=np.int32)
    xd2,nd2=jm.device_segments(x2,n2)
    t0=time.perf_counter(); jm._forward(jm._compute_params,xd2,nd2).block_until_ready(); c2=time.perf_counter()-t0
    rep("gpu_recompile_cost", True, f"compile bs8sl128={c1:.2f}s  bs8sl256={c2:.2f}s (recompiles per shape)")
except Exception as e:
    traceback.print_exc(); rep("gpu_recompile_cost", False, str(e))

# ---------- 8. GPU memory at full 1300 segments ----------
try:
    import subprocess
    def gpu_mem():
        r=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True)
        return int(r.stdout.strip())
    base=gpu_mem()
    jm=load_model(ART["dim"], device=DEV)
    cfg=jm.config.feature
    x=np.zeros((8,cfg.max_segments,1,cfg.n_mels,cfg.seg_length),dtype=np.float32)
    n=np.full((8,),cfg.max_segments,dtype=np.int32)
    xd,nd=jm.device_segments(x,n)
    out=jm._forward(jm._compute_params,xd,nd).block_until_ready()
    peak=gpu_mem()
    rep("gpu_mem_full1300_bs8", np.isfinite(np.asarray(out)).all(), f"mem_used_base={base}MB peak={peak}MB delta={peak-base}MB shape={out.shape}")
except Exception as e:
    traceback.print_exc(); rep("gpu_mem_full1300_bs8", False, f"{type(e).__name__}: {e}")

# ---------- 9. end-to-end WAV parity on GPU ----------
try:
    td=tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    fails=[]
    for key in ART:
        wav=Path(td)/f"{key}.wav"; sf.write(str(wav),0.05*np.sin(2*np.pi*440*t),sr)
        jm=load_model(ART[key],device=DEV)
        out=P.predict_file(jm,wav)
        x,nw=F.preprocess_file(wav,jm.config.feature)
        ptm,a=pt_model(key)
        with torch.no_grad():
            exp=ptm(torch.from_numpy(x[None,:]),torch.from_numpy(nw.reshape(1))).numpy()[0]
        exp_d={nm:float(exp[i]) for i,nm in enumerate(jm.config.output_names)}
        md=float(max(abs(out[k]-exp_d[k]) for k in out))
        if md>=1e-3: fails.append((key,md))
        rep(f"gpu_e2e_wav:{key}", md<1e-3, f"max_abs={md:.2e}")
    rep("gpu_e2e_wav_all", not fails, str(fails))
except Exception as e:
    traceback.print_exc(); rep("gpu_e2e_wav", False, str(e))

# ---------- 10. persistent compilation cache effectiveness ----------
try:
    cache=tempfile.mkdtemp()
    jm1=load_model(ART["mos"],device=DEV,cache_dir=cache)
    cfg=jm1.config.feature
    x=np.zeros((8,128,1,cfg.n_mels,cfg.seg_length),dtype=np.float32); n=np.full((8,),128,dtype=np.int32)
    xd,nd=jm1.device_segments(x,n)
    t0=time.perf_counter(); jm1._forward(jm1._compute_params,xd,nd).block_until_ready(); cold=time.perf_counter()-t0
    # second model load with same cache -> should hit cache
    jm2=load_model(ART["mos"],device=DEV,cache_dir=cache)
    t0=time.perf_counter(); jm2._forward(jm2._compute_params,xd,nd).block_until_ready(); warm=time.perf_counter()-t0
    rep("gpu_persistent_cache", warm<cold, f"cold={cold:.2f}s warm={warm:.2f}s speedup={cold/warm:.1f}x")
except Exception as e:
    traceback.print_exc(); rep("gpu_persistent_cache", False, str(e))

print("\n"+"="*60)
f=[r for r in res if r[1] is False]
print(f"TOTAL {len(res)} | FAIL {len(f)}")
for n,ok,d in f: print(f"  FAIL {n}: {d}")
