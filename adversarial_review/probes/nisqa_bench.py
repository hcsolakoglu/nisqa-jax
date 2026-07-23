#!/usr/bin/env python3
"""Adversarial benchmark: JAX-GPU vs PyTorch-GPU. Read-only vs the port."""
import os, sys, time, json, tempfile, traceback, statistics
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("NISQA_JAX_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(os.environ.get("NISQA_PT_ROOT", ROOT / "nisqa_pytorch"))))
WEIGHTS = ROOT / "nisqa_jax" / "weights"
PT_WEIGHTS = Path(os.environ.get("NISQA_PT_ROOT", ROOT / "nisqa_pytorch")) / "weights"

import torch
import jax
import jax.numpy as jnp
import nisqa.NISQA_lib as NL
from nisqa_jax.checkpoint import load_model
from nisqa_jax.model import forward as jax_forward

ART = {"mos": WEIGHTS/"nisqa_mos_only.npz", "dim": WEIGHTS/"nisqa.npz", "tts": WEIGHTS/"nisqa_tts.npz"}
PT = {"mos": PT_WEIGHTS/"nisqa_mos_only.tar", "dim": PT_WEIGHTS/"nisqa.tar", "tts": PT_WEIGHTS/"nisqa_tts.tar"}

torch.backends.cudnn.benchmark = True  # PyTorch default-ish for inference; fair

def pt_args(args):
    keys=["ms_seg_length","ms_n_mels","cnn_model","cnn_c_out_1","cnn_c_out_2","cnn_c_out_3","cnn_kernel_size",
          "cnn_dropout","cnn_pool_1","cnn_pool_2","cnn_pool_3","cnn_fc_out_h","td","td_sa_d_model","td_sa_nhead",
          "td_sa_pos_enc","td_sa_num_layers","td_sa_h","td_sa_dropout","td_lstm_h","td_lstm_num_layers",
          "td_lstm_dropout","td_lstm_bidirectional","td_2","td_2_sa_d_model","td_2_sa_nhead","td_2_sa_pos_enc",
          "td_2_sa_num_layers","td_2_sa_h","td_2_sa_dropout","td_2_lstm_h","td_2_lstm_num_layers","td_2_lstm_dropout",
          "td_2_lstm_bidirectional","pool","pool_att_h","pool_att_dropout"]
    return {k:args[k] for k in keys if k in args}

def pt_model(key, device):
    ck=torch.load(PT[key],map_location="cpu"); a=ck["args"]
    cls={"NISQA":NL.NISQA,"NISQA_DIM":NL.NISQA_DIM}[a["model"]]
    m=cls(**pt_args(a)); m.load_state_dict(ck["model_state_dict"],strict=True); m.eval()
    return m.to(device), a

def make_input(batch, steps, n_mels, seg_length, dist="normal", seed=0):
    rng=np.random.default_rng(seed)
    if dist=="normal":
        x=rng.normal(size=(batch,steps,1,n_mels,seg_length)).astype(np.float32)
    elif dist=="zeros":
        x=np.zeros((batch,steps,1,n_mels,seg_length),dtype=np.float32)
    elif dist=="large":  # adversarial: large values
        x=(rng.normal(size=(batch,steps,1,n_mels,seg_length))*1e3).astype(np.float32)
    elif dist=="uniform":
        x=rng.uniform(-1,1,size=(batch,steps,1,n_mels,seg_length)).astype(np.float32)
    elif dist=="mixed_len":  # adversarial: vary n_wins across batch
        x=rng.normal(size=(batch,steps,1,n_mels,seg_length)).astype(np.float32)
    n=np.full((batch,),steps,dtype=np.int32)
    if dist=="mixed_len":
        for i in range(batch): n[i]=max(1, steps*(i+1)//batch); x[i,n[i]:]=0.0
    return x,n

def time_torch(fn, steps, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(steps): fn()
    torch.cuda.synchronize()
    return time.perf_counter()-t0

def time_jax(fn, steps, warmup=5):
    for _ in range(warmup): fn().block_until_ready()
    t0=time.perf_counter()
    for _ in range(steps): fn().block_until_ready()
    return time.perf_counter()-t0

def bench_case(key, batch, steps, dist, precision, jax_model, pt_model_gpu, n_mels, seg_length, measure_steps=50):
    x_np, n_np = make_input(batch, steps, n_mels, seg_length, dist)
    out = {"key":key,"batch":batch,"steps":steps,"dist":dist,"precision":precision}
    # ---- JAX ----
    xd, nd = jax_model.device_segments(x_np, n_np)
    cp = jax_model._compute_params
    # compile
    t0=time.perf_counter(); jax_model._forward(cp,xd,nd).block_until_ready(); out["jax_compile"]=time.perf_counter()-t0
    jax_sec = time_jax(lambda: jax_model._forward(cp,xd,nd), measure_steps)
    out["jax_sec"]=jax_sec; out["jax_lat_ms"]=jax_sec/measure_steps*1000
    out["jax_sps"]=batch*measure_steps/jax_sec
    jax_out = np.asarray(jax_model._forward(cp,xd,nd).block_until_ready())
    # ---- PyTorch ----
    dev = torch.device("cuda")
    xt = torch.from_numpy(x_np).to(dev); nt = torch.from_numpy(n_np).to(dev)
    with torch.no_grad():
        # warmup + compile (cudnn autotune)
        for _ in range(3): pt_model_gpu(xt, nt)
        torch.cuda.synchronize()
        t0=time.perf_counter(); pt_model_gpu(xt,nt); torch.cuda.synchronize(); out["pt_warmup_compile"]=time.perf_counter()-t0
        pt_sec = time_torch(lambda: pt_model_gpu(xt,nt), measure_steps)
    out["pt_sec"]=pt_sec; out["pt_lat_ms"]=pt_sec/measure_steps*1000
    out["pt_sps"]=batch*measure_steps/pt_sec
    out["speedup"]=pt_sec/jax_sec
    with torch.no_grad():
        pt_out = pt_model_gpu(xt,nt).cpu().numpy()
    out["parity_max_abs"]=float(np.max(np.abs(jax_out-pt_out)))
    return out

results=[]
def pr(r):
    results.append(r)
    flag = "JAX WINS" if r["speedup"]>1 else ("TIE" if abs(r["speedup"]-1)<0.05 else "PT WINS")
    print(f"{r['key']:4s} bs={r['batch']:<3d} sl={r['steps']:<5d} {r['dist']:9s} {r['precision']:8s} | "
          f"JAX {r['jax_lat_ms']:6.2f}ms ({r['jax_sps']:6.1f}sps) vs PT {r['pt_lat_ms']:6.2f}ms ({r['pt_sps']:6.1f}sps) | "
          f"speedup {r['speedup']:5.2f}x | parity {r['parity_max_abs']:.1e} | {flag}")

print("="*120)
print("ADVERSARIAL BENCHMARK: JAX-GPU vs PyTorch-GPU (RTX 3070)")
print("="*120)

for key in ["mos","dim","tts"]:
    jm = load_model(ART[key], device="gpu", precision="float32")
    ptm, _ = pt_model(key, torch.device("cuda"))
    cfg = jm.config.feature
    print(f"\n### {key.upper()} (cnn={jm.config.cnn_model} td={jm.config.td} pool={jm.config.pool}) ###")
    # Standard sweep: batch sizes x sequence lengths
    for batch in [1, 8, 16, 32]:
        for steps in [64, 128, 256, 512]:
            try:
                r = bench_case(key, batch, steps, "normal", "float32", jm, ptm, cfg.n_mels, cfg.seg_length)
                pr(r)
            except Exception as e:
                print(f"  {key} bs={batch} sl={steps} ERROR: {type(e).__name__}: {str(e)[:80]}")
    # Adversarial input distributions at bs=8, sl=128
    for dist in ["zeros","large","uniform","mixed_len"]:
        try:
            r = bench_case(key, 8, 128, dist, "float32", jm, ptm, cfg.n_mels, cfg.seg_length)
            pr(r)
        except Exception as e:
            print(f"  {key} dist={dist} ERROR: {type(e).__name__}: {str(e)[:80]}")
    # bf16 comparison (JAX bf16 vs PyTorch float16 — closest GPU fast-math)
    try:
        jm_bf = load_model(ART[key], device="gpu", precision="bf16")
        r = bench_case(key, 8, 128, "normal", "bf16", jm_bf, ptm, cfg.n_mels, cfg.seg_length)
        r["precision"]="bf16(jax)"
        pr(r)
    except Exception as e:
        print(f"  {key} bf16 ERROR: {type(e).__name__}: {str(e)[:80]}")
    try:
        # PyTorch autocast fp16
        ptm_h, _ = pt_model(key, torch.device("cuda"))
        x_np,n_np = make_input(8,128,cfg.n_mels,cfg.seg_length,"normal")
        xt=torch.from_numpy(x_np).cuda(); nt=torch.from_numpy(n_np).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(3): ptm_h(xt,nt)
            torch.cuda.synchronize(); t0=time.perf_counter()
            for _ in range(50):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    ptm_h(xt,nt)
            torch.cuda.synchronize(); pt_h=time.perf_counter()-t0
        # jax bf16 already measured above; find it
        jbf = [r for r in results if r["key"]==key and r["precision"]=="bf16(jax)"]
        if jbf:
            sp = pt_h/jbf[0]["jax_sec"]
            print(f"  {key:4s} bs=8   sl=128  normal    bf16(pt)  | JAX {jbf[0]['jax_lat_ms']:6.2f}ms vs PT-fp16 {pt_h/50*1000:6.2f}ms | speedup {sp:5.2f}x (pt autocast fp16)")
    except Exception as e:
        print(f"  {key} pt-fp16 ERROR: {type(e).__name__}: {str(e)[:80]}")
    del jm, ptm
    torch.cuda.empty_cache()

# Summary
print("\n"+"="*120)
print("SUMMARY")
print("="*120)
jax_wins=[r for r in results if r["speedup"]>1.0]
pt_wins=[r for r in results if r["speedup"]<1.0]
print(f"Total cases: {len(results)} | JAX faster: {len(jax_wins)} | PyTorch faster: {len(pt_wins)}")
for key in ["mos","dim","tts"]:
    kr=[r for r in results if r["key"]==key and r["precision"]=="float32" and r["dist"]=="normal"]
    if kr:
        sps=[r["speedup"] for r in kr]
        print(f"  {key:4s} normal/float32: speedup median={statistics.median(sps):.2f}x min={min(sps):.2f}x max={max(sps):.2f}x (n={len(kr)})")
# parity check
bad=[r for r in results if r["parity_max_abs"]>1e-3]
print(f"Parity violations (>1e-3): {len(bad)}")
for r in bad: print(f"  {r['key']} bs={r['batch']} sl={r['steps']} {r['dist']}: {r['parity_max_abs']:.2e}")
# compile times
comp=[(r["key"],r["batch"],r["steps"],r["jax_compile"],r.get("pt_warmup_compile",0)) for r in results if r["dist"]=="normal" and r["precision"]=="float32"]
print("\nCompile times (first call, includes JIT/cudnn autotune):")
for k,b,s,jc,pc in comp[:6]:
    print(f"  {k} bs={b} sl={s}: JAX {jc:.3f}s vs PT {pc:.3f}s")
json.dump(results, open("/tmp/bench_results.json","w"), indent=2)
print("\nFull results: /tmp/bench_results.json")
