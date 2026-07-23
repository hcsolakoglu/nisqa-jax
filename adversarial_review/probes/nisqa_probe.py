#!/usr/bin/env python3
"""Adversarial probe of NISQA-JAX Port A. Read-only: imports the package, never edits it."""
import os, sys, json, traceback, tempfile, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

# Repo root: 3 levels up from adversarial_review/probes/<this>. Override with
# NISQA_JAX_ROOT for non-standard layouts. WEIGHTS now points at the in-package
# nisqa_jax/weights/ location (the canonical post-relocation path).
ROOT = Path(os.environ.get("NISQA_JAX_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
WEIGHTS = ROOT / "nisqa_jax" / "weights"
# Optional PyTorch reference root for parity probes (set NISQA_PT_ROOT); probes
# that need torch skip gracefully if it is unset.
PT_ROOT = Path(os.environ.get("NISQA_PT_ROOT", ROOT / "nisqa_pytorch"))
PT_WEIGHTS = PT_ROOT / "weights"

import jax
import jax.numpy as jnp
from nisqa_jax.checkpoint import load_model, load_converted_checkpoint, convert_checkpoint
from nisqa_jax import predict as P
from nisqa_jax import features as F

ARTIFACTS = {
    "mos": WEIGHTS / "nisqa_mos_only.npz",
    "dim": WEIGHTS / "nisqa.npz",
    "tts": WEIGHTS / "nisqa_tts.npz",
}
PT_CKPTS = {
    "mos": PT_WEIGHTS / "nisqa_mos_only.tar",
    "dim": PT_WEIGHTS / "nisqa.tar",
    "tts": PT_WEIGHTS / "nisqa_tts.tar",
}

results = []
def report(name, ok, detail=""):
    results.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}: {detail}")

def synth(model, batch=2, steps=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, steps, 1, model.config.feature.n_mels, model.config.feature.seg_length)).astype(np.float32)
    # n length must equal batch; first sample shorter than the rest
    n = np.full((batch,), steps, dtype=np.int32)
    n[0] = max(1, steps-5)
    x[0, n[0]:] = 0.0
    return x, n

# ---------- 1. Load + forward sanity ----------
for key, art in ARTIFACTS.items():
    try:
        m = load_model(art, device="cpu")
        x, n = synth(m)
        out = m.predict_segments(x, n)
        ok = out.shape == (2, len(m.config.output_names)) and np.isfinite(out).all()
        report(f"load+forward:{key}", ok, f"shape={out.shape} finite={np.isfinite(out).all()} out={np.round(out[0],4).tolist()}")
    except Exception as e:
        report(f"load+forward:{key}", False, f"{type(e).__name__}: {e}")

# ---------- 2. Parity vs PyTorch (synthetic segments) ----------
sys.path.insert(0, str(PT_ROOT))
try:
    import torch
    import nisqa.NISQA_lib as NL
    def pt_model_args(args):
        keys = ["ms_seg_length","ms_n_mels","cnn_model","cnn_c_out_1","cnn_c_out_2","cnn_c_out_3",
                "cnn_kernel_size","cnn_dropout","cnn_pool_1","cnn_pool_2","cnn_pool_3","cnn_fc_out_h",
                "td","td_sa_d_model","td_sa_nhead","td_sa_pos_enc","td_sa_num_layers","td_sa_h","td_sa_dropout",
                "td_lstm_h","td_lstm_num_layers","td_lstm_dropout","td_lstm_bidirectional","td_2",
                "td_2_sa_d_model","td_2_sa_nhead","td_2_sa_pos_enc","td_2_sa_num_layers","td_2_sa_h","td_2_sa_dropout",
                "td_2_lstm_h","td_2_lstm_num_layers","td_2_lstm_dropout","td_2_lstm_bidirectional","pool","pool_att_h","pool_att_dropout"]
        return {k: args[k] for k in keys if k in args}
    for key, art in ARTIFACTS.items():
        ck = PT_CKPTS[key]
        if not ck.exists():
            report(f"parity:{key}", None, "pytorch ckpt missing"); continue
        ckpt = torch.load(ck, map_location="cpu")
        args = ckpt["args"]
        cls = {"NISQA": NL.NISQA, "NISQA_DIM": NL.NISQA_DIM}[args["model"]]
        ptm = cls(**pt_model_args(args))
        ptm.load_state_dict(ckpt["model_state_dict"], strict=True)
        ptm.eval()
        jm = load_model(art, device="cpu")
        x, n = synth(jm)
        with torch.no_grad():
            exp = ptm(torch.from_numpy(x), torch.from_numpy(n)).numpy()
        act = jm.predict_segments(x, n)
        md = float(np.max(np.abs(act-exp)))
        ok = md < 1e-3
        report(f"parity_vs_pytorch:{key}", ok, f"max_abs_diff={md:.2e} (thresh 1e-3)")
except Exception as e:
    traceback.print_exc()
    report("parity_vs_pytorch", False, f"{type(e).__name__}: {e}")

# ---------- 3. Staged parity (CNN, TD) ----------
try:
    for key, art in ARTIFACTS.items():
        ck = PT_CKPTS[key]
        if not ck.exists(): continue
        ckpt = torch.load(ck, map_location="cpu"); args = ckpt["args"]
        cls = {"NISQA": NL.NISQA, "NISQA_DIM": NL.NISQA_DIM}[args["model"]]
        ptm = cls(**pt_model_args(args)); ptm.load_state_dict(ckpt["model_state_dict"], strict=True); ptm.eval()
        jm = load_model(art, device="cpu")
        x, n = synth(jm, steps=12)
        with torch.no_grad():
            xt, nt = torch.from_numpy(x), torch.from_numpy(n)
            ec = ptm.cnn(xt, nt)
            etd, en = ptm.time_dependency(ec, nt)
            etd, _ = ptm.time_dependency_2(etd, en)
        st = jm.predict_stages(x, n)
        dc = float(np.max(np.abs(st["cnn"]-ec.numpy())))
        dtd = float(np.max(np.abs(st["time_dependency"]-etd.numpy())))
        report(f"staged_cnn:{key}", dc < 1e-3, f"max_abs={dc:.2e}")
        report(f"staged_td:{key}", dtd < 1e-3, f"max_abs={dtd:.2e}")
except Exception as e:
    traceback.print_exc(); report("staged", False, f"{type(e).__name__}: {e}")

# ---------- 4. Edge cases ----------
m = load_model(ARTIFACTS["mos"], device="cpu")
# n_wins all max (no padding)
try:
    x, n = synth(m, batch=3, steps=8); n[:] = 8
    out = m.predict_segments(x, n); report("edge:no_padding", np.isfinite(out).all(), f"shape={out.shape}")
except Exception as e: report("edge:no_padding", False, str(e))
# n_wins = 1
try:
    x, n = synth(m, batch=2, steps=4); n[:] = 1
    out = m.predict_segments(x, n); report("edge:n_wins=1", np.isfinite(out).all(), f"out={np.round(out[:,0],4).tolist()}")
except Exception as e: report("edge:n_wins=1", False, str(e))
# n_wins = 0 should raise
try:
    x, _ = synth(m, batch=2, steps=4)
    m.predict_segments(x, np.zeros(2, dtype=np.int32))
    report("edge:n_wins=0", False, "no error raised")
except ValueError as e: report("edge:n_wins=0", True, f"raised: {e}")
except Exception as e: report("edge:n_wins=0", False, f"wrong type {type(e).__name__}")
# mixed batch n_wins [1, max]
try:
    x, n = synth(m, batch=2, steps=10); n[:] = [1, 10]
    out = m.predict_segments(x, n); report("edge:mixed_batch", np.isfinite(out).all(), f"shape={out.shape}")
except Exception as e: report("edge:mixed_batch", False, str(e))
# padding invariance (tail garbage ignored)
try:
    x, n = synth(m, batch=2, steps=12)
    base = m.predict_segments(x, n)
    xt = x.copy(); xt[0, n[0]:n.max()] = 999.0
    d1 = float(np.max(np.abs(m.predict_segments(xt, n)-base)))
    ext = np.concatenate([x, np.full_like(x[:, :3], -999.0)], axis=1)
    d2 = float(np.max(np.abs(m.predict_segments(ext, n)-base)))
    report("edge:padding_invariance", d1==0 and d2==0, f"tail_garbage={d1:.2e} extra_padded={d2:.2e}")
except Exception as e: report("edge:padding_invariance", False, str(e))
# bf16 finiteness + drift
try:
    mf = load_model(ARTIFACTS["mos"], device="cpu", precision="float32")
    mb = load_model(ARTIFACTS["mos"], device="cpu", precision="bf16")
    x, n = synth(mf, steps=16)
    ef = mf.predict_segments(x, n); eb = mb.predict_segments(x, n)
    drift = float(np.max(np.abs(ef-eb)))
    report("edge:bf16", np.isfinite(eb).all() and drift < 0.5, f"finite={np.isfinite(eb).all()} drift={drift:.3f}")
except Exception as e: report("edge:bf16", False, str(e))
# NaN input propagation
try:
    x, n = synth(m, batch=2, steps=8); x[0,0,0,0,0] = np.nan
    out = m.predict_segments(x, n)
    report("edge:nan_input", np.isnan(out).any(), f"nan_propagates={np.isnan(out).any()}")
except Exception as e: report("edge:nan_input", False, str(e))
# all-zero input
try:
    x = np.zeros((2, 8, 1, m.config.feature.n_mels, m.config.feature.seg_length), dtype=np.float32)
    n = np.array([8, 8], dtype=np.int32)
    out = m.predict_segments(x, n)
    report("edge:zero_input", np.isfinite(out).all(), f"out={np.round(out[:,0],4).tolist()}")
except Exception as e: report("edge:zero_input", False, str(e))
# very large input values (overflow check)
try:
    x, n = synth(m, batch=2, steps=8); x *= 1e4
    out = m.predict_segments(x, n)
    report("edge:large_input", np.isfinite(out).all(), f"finite={np.isfinite(out).all()}")
except Exception as e: report("edge:large_input", False, str(e))
# max_segments=1300 full length (recompile + memory)
try:
    md = load_model(ARTIFACTS["dim"], device="cpu")
    cfg = md.config.feature
    x = np.zeros((1, cfg.max_segments, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
    n = np.array([cfg.max_segments], dtype=np.int32)
    out = md.predict_segments(x, n)
    report("edge:full_1300", out.shape == (1, 5) and np.isfinite(out).all(), f"shape={out.shape}")
except Exception as e: report("edge:full_1300", False, f"{type(e).__name__}: {e}")

# ---------- 5. predict_batch edge cases ----------
try:
    # batch_size > len(paths)
    import soundfile as sf
    td = tempfile.mkdtemp()
    sr = 48000; t = np.arange(sr*2)/sr
    for i in range(3):
        sf.write(f"{td}/f{i}.wav", 0.05*np.sin(2*np.pi*440*t), sr)
    df = P.predict_batch(m, sorted(Path(td).glob("*.wav")), batch_size=8, preprocess_workers=1)
    report("batch:bs_gt_len", len(df)==3 and "deg" in df.columns, f"rows={len(df)} cols={list(df.columns)}")
except Exception as e: report("batch:bs_gt_len", False, str(e))
try:
    P.predict_batch(m, [Path("x.wav")], preprocess_workers=0)
    report("batch:workers=0", False, "no error")
except ValueError as e: report("batch:workers=0", True, str(e))
except Exception as e: report("batch:workers=0", False, f"{type(e).__name__}")
# parallel vs serial order preservation
try:
    td = tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    paths=[]
    for i in range(5):
        p=Path(f"{td}/a{i}.wav"); sf.write(str(p), 0.05*np.sin(2*np.pi*(200+50*i)*t), sr); paths.append(p)
    s = P.predict_batch(m, paths, batch_size=2, preprocess_workers=1)
    pa = P.predict_batch(m, paths, batch_size=2, preprocess_workers=2)
    ok = list(s["deg"]) == list(pa["deg"]) == [str(p) for p in paths]
    report("batch:parallel_order", ok, f"serial={list(s['deg'])} parallel={list(pa['deg'])}")
except Exception as e: report("batch:parallel_order", False, str(e))

# ---------- 6. CLI / hidden requirements ----------
def run_cli(args):
    import subprocess
    env = dict(os.environ); env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run([sys.executable, "-m", "nisqa_jax.predict"]+args, capture_output=True, text=True, env=env, cwd=str(ROOT))
    return r
try:
    td = tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    sf.write(f"{td}/x.wav", 0.05*np.sin(2*np.pi*440*t), sr)
    r = run_cli(["--mode","predict_file","--pretrained_model",str(ARTIFACTS["mos"]),"--deg",f"{td}/x.wav","--device","cpu"])
    report("cli:predict_file", r.returncode==0 and "mos" in r.stdout, f"rc={r.returncode} stdout={r.stdout.strip()[:80]}")
except Exception as e: report("cli:predict_file", False, str(e))
try:
    td = tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    for i in range(2): sf.write(f"{td}/w{i}.wav", 0.05*np.sin(2*np.pi*440*t), sr)
    outdir = f"{td}/out"
    r = run_cli(["--mode","predict_dir","--pretrained_model",str(ARTIFACTS["dim"]),"--data_dir",td,"--output_dir",outdir,"--device","cpu"])
    csv_ok = Path(outdir, "NISQA_results.csv").exists()
    cols = []
    if csv_ok:
        import pandas as pd
        cols = list(pd.read_csv(Path(outdir,"NISQA_results.csv")).columns)
    expected = ["deg","mos_pred","noi_pred","dis_pred","col_pred","loud_pred"]
    report("cli:predict_dir+NISQA_results.csv", r.returncode==0 and csv_ok and cols==expected, f"cols={cols}")
except Exception as e: report("cli:predict_dir", False, str(e))
try:
    td = tempfile.mkdtemp(); sr=48000; t=np.arange(sr*2)/sr
    sf.write(f"{td}/s.wav", np.stack([0.03*np.sin(2*np.pi*220*t), 0.05*np.sin(2*np.pi*440*t)], axis=1), sr)
    r = run_cli(["--mode","predict_file","--pretrained_model",str(ARTIFACTS["mos"]),"--deg",f"{td}/s.wav","--ms_channel","1","--device","cpu"])
    report("cli:stereo_channel", r.returncode==0 and "mos" in r.stdout, f"rc={r.returncode} stdout={r.stdout.strip()[:80]}")
except Exception as e: report("cli:stereo_channel", False, str(e))
try:
    r = run_cli(["--mode","predict_file","--pretrained_model",str(ARTIFACTS["mos"]),"--deg","/nonexistent/x.wav","--device","cpu"])
    report("cli:missing_file_error", r.returncode!=0, f"rc={r.returncode} err={r.stderr.strip()[:80]}")
except Exception as e: report("cli:missing_file_error", False, str(e))
try:
    r = run_cli(["--mode","bad_mode","--pretrained_model",str(ARTIFACTS["mos"])])
    report("cli:bad_mode_rejected", r.returncode!=0, f"rc={r.returncode}")
except Exception as e: report("cli:bad_mode", False, str(e))

# ---------- 7. Config / conversion robustness ----------
# load via .json sidecar
try:
    m2 = load_model(ARTIFACTS["mos"].with_suffix(".json"), device="cpu")
    report("config:load_via_json", m2.config.output_names==("mos",), f"names={m2.config.output_names}")
except Exception as e: report("config:load_via_json", False, str(e))
# conversion version mismatch
try:
    import json as J
    p = ARTIFACTS["mos"].with_suffix(".json")
    data = J.loads(p.read_text()); data["conversion_version"] = 999
    bad = Path(tempfile.mkdtemp())/"bad.json"
    bad.write_text(J.dumps(data))
    import shutil
    shutil.copy(ARTIFACTS["mos"], bad.with_suffix(".npz"))
    load_converted_checkpoint(bad)
    report("config:version_mismatch", False, "no error")
except ValueError as e: report("config:version_mismatch", True, str(e)[:60])
except Exception as e: report("config:version_mismatch", False, f"{type(e).__name__}: {e}")
# unsupported NISQA_DE args
try:
    from nisqa_jax.config import config_from_checkpoint_args
    args = {"model":"NISQA_DE","cnn_model":"adapt","td":"self_att","pool":"att","td_2":"skip",
            "ms_n_fft":4096,"ms_hop_length":0.01,"ms_win_length":0.02,"ms_n_mels":48,"ms_fmax":20000,
            "ms_seg_length":15,"ms_seg_hop_length":4,"ms_max_segments":1300}
    config_from_checkpoint_args(args, Path("x.tar"), "abc")
    report("config:NISQA_DE_rejected", False, "no error")
except NotImplementedError as e: report("config:NISQA_DE_rejected", True, str(e)[:60])
except Exception as e: report("config:NISQA_DE_rejected", False, f"{type(e).__name__}")
# td_2 != skip rejected
try:
    args = {"model":"NISQA","cnn_model":"adapt","td":"self_att","pool":"att","td_2":"self_att",
            "ms_n_fft":4096,"ms_hop_length":0.01,"ms_win_length":0.02,"ms_n_mels":48,"ms_fmax":20000,
            "ms_seg_length":15,"ms_seg_hop_length":4,"ms_max_segments":1300}
    config_from_checkpoint_args(args, Path("x.tar"), "abc")
    report("config:td_2_non_skip_rejected", False, "no error")
except NotImplementedError as e: report("config:td_2_non_skip_rejected", True, str(e)[:60])
except Exception as e: report("config:td_2_non_skip_rejected", False, f"{type(e).__name__}")
# unsupported combo (NISQA + standard + self_att)
try:
    args = {"model":"NISQA","cnn_model":"standard","td":"self_att","pool":"att","td_2":"skip",
            "ms_n_fft":4096,"ms_hop_length":0.01,"ms_win_length":0.02,"ms_n_mels":48,"ms_fmax":20000,
            "ms_seg_length":15,"ms_seg_hop_length":4,"ms_max_segments":1300}
    config_from_checkpoint_args(args, Path("x.tar"), "abc")
    report("config:bad_combo_rejected", False, "no error")
except NotImplementedError as e: report("config:bad_combo_rejected", True, str(e)[:60])
except Exception as e: report("config:bad_combo_rejected", False, f"{type(e).__name__}")

# ---------- 8. feature extraction edge cases ----------
cfg = m.config.feature
from dataclasses import replace
# even seg_length rejected
try:
    F.segment_melspec("x.wav", np.zeros((cfg.n_mels, 10), dtype=np.float32), replace(cfg, seg_length=4))
    report("feat:even_seg_rejected", False, "no error")
except ValueError as e: report("feat:even_seg_rejected", True, str(e)[:50])
except Exception as e: report("feat:even_seg_rejected", False, f"{type(e).__name__}")
# too short
try:
    F.segment_melspec("tiny.wav", np.zeros((cfg.n_mels, cfg.seg_length-1), dtype=np.float32), cfg)
    report("feat:too_short_rejected", False, "no error")
except ValueError as e: report("feat:too_short_rejected", "tiny.wav" in str(e), str(e)[:60])
except Exception as e: report("feat:too_short_rejected", False, f"{type(e).__name__}")
# n_wins > max_segments rejected
try:
    big = np.zeros((cfg.n_mels, cfg.max_segments*cfg.seg_hop_length + cfg.seg_length), dtype=np.float32)
    F.segment_melspec("big.wav", big, cfg)
    report("feat:over_max_rejected", False, "no error")
except ValueError as e: report("feat:over_max_rejected", "big.wav" in str(e) and "max" in str(e).lower(), str(e)[:60])
except Exception as e: report("feat:over_max_rejected", False, f"{type(e).__name__}")
# segment parity vs pytorch
try:
    seg = np.arange(cfg.n_mels*32, dtype=np.float32).reshape(cfg.n_mels, 32)
    jx, jn = F.segment_melspec("s.wav", seg, cfg)
    ptx, ptn = NL.segment_specs("s.wav", seg, cfg.seg_length, cfg.seg_hop_length, cfg.max_segments)
    report("feat:segment_parity", np.array_equal(jx, ptx.numpy()) and int(jn)==int(ptn), f"equal={np.array_equal(jx, ptx.numpy())} jn={int(jn)} ptn={int(ptn)}")
except Exception as e: report("feat:segment_parity", False, f"{type(e).__name__}: {e}")

# ---------- SUMMARY ----------
print("\n" + "="*70)
fails = [r for r in results if r[1] is False]
passes = [r for r in results if r[1] is True]
skips = [r for r in results if r[1] is None]
print(f"TOTAL {len(results)} | PASS {len(passes)} | FAIL {len(fails)} | SKIP {len(skips)}")
if fails:
    print("--- FAILURES ---")
    for n,ok,d in fails: print(f"  FAIL {n}: {d}")
