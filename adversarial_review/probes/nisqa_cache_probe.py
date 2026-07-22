#!/usr/bin/env python3
"""Verify persistent compilation cache behavior across reloads on GPU."""
import sys, time, tempfile
from pathlib import Path
import numpy as np
ROOT = Path("/media/mithex/NVME 2/Codex Linux/NISQA PORT PROJECT")
sys.path.insert(0, str(ROOT))
import jax
from nisqa_jax.checkpoint import load_model
ART = ROOT/"weights"/"nisqa_mos_only.npz"

cache = Path(tempfile.mkdtemp())
def time_compile(cache_dir):
    m = load_model(ART, device="gpu", cache_dir=str(cache_dir) if cache_dir else None)
    cfg = m.config.feature
    x = np.zeros((8,128,1,cfg.n_mels,cfg.seg_length),dtype=np.float32)
    n = np.full((8,),128,dtype=np.int32)
    xd,nd = m.device_segments(x,n)
    t0=time.perf_counter()
    m._forward(m._compute_params,xd,nd).block_until_ready()
    return time.perf_counter()-t0, m

# No cache
c1,_ = time_compile(None)
print(f"no_cache:           {c1:.3f}s")
# With cache, first load (cold, populates cache)
c2,_ = time_compile(cache)
print(f"cache_cold(1st):    {c2:.3f}s")
# With cache, second load (should be warm if cache works across reloads)
c3,_ = time_compile(cache)
print(f"cache_warm(2nd):    {c3:.3f}s")
# With cache, third load
c4,_ = time_compile(cache)
print(f"cache_warm(3rd):    {c4:.3f}s")
print(f"\nspeedup warm/cold = {c2/c3:.2f}x  (1.0x = cache NOT helping across reloads)")
print(f"cache dir contents: {list(cache.rglob('*'))[:5]}")

# Also: does the SAME model instance benefit (in-process cache)?
m = load_model(ART, device="gpu")
cfg = m.config.feature
x = np.zeros((8,128,1,cfg.n_mels,cfg.seg_length),dtype=np.float32); n=np.full((8,),128,dtype=np.int32)
xd,nd=m.device_segments(x,n)
t0=time.perf_counter(); m._forward(m._compute_params,xd,nd).block_until_ready(); ic1=time.perf_counter()-t0
t0=time.perf_counter(); m._forward(m._compute_params,xd,nd).block_until_ready(); ic2=time.perf_counter()-t0
print(f"\nin-process same-instance: first={ic1:.3f}s second={ic2:.4f}s (in-process cache works: {ic1/ic2:.0f}x)")
