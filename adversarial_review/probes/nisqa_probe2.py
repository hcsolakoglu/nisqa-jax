#!/usr/bin/env python3
"""Adversarial probe round 2: deeper hidden requirements."""
import os, sys, json, tempfile, traceback
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("NISQA_JAX_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(os.environ.get("NISQA_PT_ROOT", ROOT / "nisqa_pytorch"))))
WEIGHTS = ROOT / "nisqa_jax" / "weights"
PT_WEIGHTS = Path(os.environ.get("NISQA_PT_ROOT", ROOT / "nisqa_pytorch")) / "weights"

import jax
import jax.numpy as jnp
import torch
import nisqa.NISQA_lib as NL
from nisqa_jax.checkpoint import load_model, convert_checkpoint
from nisqa_jax import predict as P
from nisqa_jax import features as F
import soundfile as sf

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
    m=cls(**pt_args(a)); m.load_state_dict(ck["model_state_dict"],strict=True); m.eval()
    return m, a

# 1. End-to-end WAV parity (full librosa pipeline) for all 3 ckpts
td=tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
for key in ART:
    try:
        wav=Path(td)/f"{key}.wav"
        sf.write(str(wav), 0.05*np.sin(2*np.pi*440*t), sr)
        jm=load_model(ART[key],device="cpu")
        out_j=P.predict_file(jm, wav)
        x,nw=F.preprocess_file(wav, jm.config.feature)
        ptm,a=pt_model(key)
        with torch.no_grad():
            exp=ptm(torch.from_numpy(x[None,:]),torch.from_numpy(nw.reshape(1))).numpy()[0]
        exp_d={nm:float(exp[i]) for i,nm in enumerate(jm.config.output_names)}
        md=float(max(abs(out_j[k]-exp_d[k]) for k in out_j))
        rep(f"e2e_wav_parity:{key}", md<1e-3, f"max_abs={md:.2e} jax={ {k:round(v,4) for k,v in out_j.items()} }")
    except Exception as e:
        traceback.print_exc(); rep(f"e2e_wav_parity:{key}", False, f"{type(e).__name__}: {e}")

# 2. TTS output column name vs PyTorch
try:
    jm=load_model(ART["tts"],device="cpu")
    rep("tts_output_name", jm.config.output_names==("naturalness",), f"names={jm.config.output_names} (PyTorch writes mos_pred)")
except Exception as e: rep("tts_output_name", False, str(e))

# 3. predict_csv mode
try:
    td=tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    for i in range(3): sf.write(f"{td}/c{i}.wav", 0.05*np.sin(2*np.pi*(300+30*i)*t), sr)
    import pandas as pd
    pd.DataFrame({"deg":["c0.wav","c1.wav","c2.wav"]}).to_csv(f"{td}/list.csv",index=False)
    import subprocess
    env=dict(os.environ); env["PYTHONPATH"]=str(ROOT)
    r=subprocess.run([sys.executable,"-m","nisqa_jax.predict","--mode","predict_csv",
        "--pretrained_model",str(ART["mos"]),"--data_dir",td,"--csv_file","list.csv","--csv_deg","deg",
        "--device","cpu"],capture_output=True,text=True,env=env,cwd=str(ROOT))
    rep("cli:predict_csv", r.returncode==0 and "c0.wav" in r.stdout, f"rc={r.returncode} stdout={r.stdout.strip()[:90]}")
except Exception as e: rep("cli:predict_csv", False, str(e))

# 4. Determinism: same input twice
try:
    m=load_model(ART["dim"],device="cpu")
    rng=np.random.default_rng(7)
    x=rng.normal(size=(2,20,1,m.config.feature.n_mels,m.config.feature.seg_length)).astype(np.float32)
    n=np.array([20,15],dtype=np.int32); x[1,15:]=0
    o1=m.predict_segments(x,n); o2=m.predict_segments(x,n)
    rep("determinism", np.array_equal(o1,o2), f"identical={np.array_equal(o1,o2)}")
except Exception as e: rep("determinism", False, str(e))

# 5. Direct .tar load (requires torch) — on-the-fly conversion
try:
    if PT["mos"].exists():
        m=load_model(PT["mos"], device="cpu", cache_dir=tempfile.mkdtemp())
        x=np.zeros((1,8,1,m.config.feature.n_mels,m.config.feature.seg_length),dtype=np.float32)
        n=np.array([8],dtype=np.int32)
        out=m.predict_segments(x,n)
        rep("direct_tar_load", np.isfinite(out).all(), f"shape={out.shape} finite={np.isfinite(out).all()}")
    else:
        rep("direct_tar_load", None, "no .tar")
except Exception as e:
    traceback.print_exc(); rep("direct_tar_load", False, f"{type(e).__name__}: {e}")

# 6. bf16 on TTS (LSTM) — does scan break?
try:
    mf=load_model(ART["tts"],device="cpu",precision="float32")
    mb=load_model(ART["tts"],device="cpu",precision="bf16")
    rng=np.random.default_rng(3)
    x=rng.normal(size=(2,30,1,mf.config.feature.n_mels,mf.config.feature.seg_length)).astype(np.float32)
    n=np.array([30,20],dtype=np.int32); x[1,20:]=0
    ef=mf.predict_segments(x,n); eb=mb.predict_segments(x,n)
    drift=float(np.max(np.abs(ef-eb)))
    rep("bf16_lstm_tts", np.isfinite(eb).all() and drift<1.0, f"finite={np.isfinite(eb).all()} drift={drift:.3f}")
except Exception as e:
    traceback.print_exc(); rep("bf16_lstm_tts", False, f"{type(e).__name__}: {e}")

# 7. Stereo channel out of range
try:
    td=tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    sf.write(f"{td}/st.wav", np.stack([0.03*np.sin(2*np.pi*220*t),0.05*np.sin(2*np.pi*440*t)],axis=1), sr)
    m=load_model(ART["mos"],device="cpu")
    try:
        P.predict_file(m, f"{td}/st.wav", channel=5)
        rep("stereo_oob_channel", False, "no error for channel=5 on stereo")
    except Exception as e:
        rep("stereo_oob_channel", True, f"raised {type(e).__name__}: {str(e)[:50]}")
except Exception as e: rep("stereo_oob_channel", False, str(e))

# 8. Empty batch
try:
    m=load_model(ART["mos"],device="cpu")
    P.predict_batch(m, [], batch_size=1)
    rep("empty_batch", False, "no error for empty path list")
except Exception as e:
    rep("empty_batch", True, f"raised {type(e).__name__}: {str(e)[:50]}")

# 9. nhead latent bug: config accepts td_sa_nhead but model ignores it (always 1 head)
try:
    from nisqa_jax.config import config_from_checkpoint_args
    args={"model":"NISQA","cnn_model":"adapt","td":"self_att","pool":"att","td_2":"skip",
          "ms_n_fft":4096,"ms_hop_length":0.01,"ms_win_length":0.02,"ms_n_mels":48,"ms_fmax":20000,
          "ms_seg_length":15,"ms_seg_hop_length":4,"ms_max_segments":1300,"td_sa_nhead":4}
    cfg=config_from_checkpoint_args(args, Path("x.tar"), "abc")
    rep("nhead_accepted_but_ignored", cfg.td_sa_nhead==4, f"nhead={cfg.td_sa_nhead} (model.py uses single-head path, would be wrong for nhead>1)")
except Exception as e: rep("nhand_config", False, str(e))

# 10. Recompilation across batch sizes (perf, not correctness) — measure compile cost
try:
    m=load_model(ART["mos"],device="cpu")
    import time
    cfg=m.config.feature
    times=[]
    for bs,sl in [(1,16),(2,16),(1,32)]:
        x=np.zeros((bs,sl,1,cfg.n_mels,cfg.seg_length),dtype=np.float32)
        n=np.full((bs,),sl,dtype=np.int32)
        xd,nd=m.device_segments(x,n)
        t0=time.perf_counter()
        m._forward(m._compute_params,xd,nd).block_until_ready()
        times.append(time.perf_counter()-t0)
    rep("recompile_per_shape", True, f"compile_times={[round(t,3) for t in times]} (each new shape recompiles; no static-shape cache key)")
except Exception as e: rep("recompile_per_shape", False, str(e))

# 11. conversion determinism (re-convert mos, compare to shipped artifact)
try:
    tmp=tempfile.mkdtemp()
    cfg,params=convert_checkpoint(PT["mos"],cache_dir=tmp)
    from nisqa_jax.checkpoint import load_converted_checkpoint
    a,b=load_converted_checkpoint(ART["mos"])
    fresh=np.load(next(Path(tmp).glob("*.npz")))
    ship=np.load(ART["mos"])
    same_keys=sorted(fresh.files)==sorted(ship.files)
    same_vals=all(np.array_equal(fresh[k],ship[k]) for k in fresh.files)
    rep("conversion_determinism", same_keys and same_vals, f"keys_match={same_keys} vals_match={same_vals}")
except Exception as e:
    traceback.print_exc(); rep("conversion_determinism", False, f"{type(e).__name__}: {e}")

# 12. PyTorch adds 'model' column — does JAX?
try:
    td=tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    sf.write(f"{td}/z.wav", 0.05*np.sin(2*np.pi*440*t), sr)
    import subprocess, pandas as pd
    env=dict(os.environ); env["PYTHONPATH"]=str(ROOT)
    outdir=f"{td}/out"
    r=subprocess.run([sys.executable,"-m","nisqa_jax.predict","--mode","predict_dir",
        "--pretrained_model",str(ART["mos"]),"--data_dir",td,"--output_dir",outdir,"--device","cpu"],
        capture_output=True,text=True,env=env,cwd=str(ROOT))
    df=pd.read_csv(f"{outdir}/NISQA_results.csv")
    rep("csv_model_column", "model" not in df.columns, f"cols={list(df.columns)} (PyTorch adds 'model' col; JAX omits)")
except Exception as e: rep("csv_model_column", False, str(e))

print("\n"+"="*60)
f=[r for r in res if r[1] is False]
print(f"TOTAL {len(res)} | FAIL {len(f)}")
for n,ok,d in f: print(f"  FAIL {n}: {d}")
