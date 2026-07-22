"""O2(b) real-GPU sanity check: auto_batch recovers from an intentionally
too-large batch size on full-length audio.

Generates full-length wavs (~1200 windows, under the 1300 cap), runs
predict_batch with bs=48 (which OOMs on 8 GB at full length) and auto_batch=True,
and confirms every row is returned despite the OOM.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")

from nisqa_jax.checkpoint import load_model
from nisqa_jax.predict import predict_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

cfg_path = "weights/nisqa_mos_only.npz"


def main() -> None:
    model = load_model(cfg_path, device="gpu", precision="float32")
    f = model.config.feature
    sr = int(f.sr or 48000)
    hop = int(sr * f.hop_length_seconds)
    # Target ~1200 windows (safely under max_segments=1300).
    target_wins = 1200
    n_frames = target_wins * f.seg_hop_length + (f.seg_length - 1)
    n_samples = (n_frames - 1) * hop
    dur = n_samples / sr
    print(f"generating 2 wavs of {dur:.1f}s ({target_wins} windows each)", flush=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        paths = []
        for i in range(2):
            t = np.arange(n_samples, dtype=np.float32) / sr
            wav = 0.05 * np.sin(2 * np.pi * (220 + 110 * i) * t).astype(np.float32)
            p = td_path / f"tone_{i}.wav"
            sf.write(p, wav, sr)
            paths.append(p)

        # bs=48 OOMs at full length on 8 GB (measured ceiling is 32); auto_batch
        # must halve and recover, returning both rows.
        df = predict_batch(model, paths, batch_size=48, auto_batch=True)
        print(df.to_string(index=False), flush=True)
        assert len(df) == 2, f"expected 2 rows, got {len(df)}"
        assert df["mos_pred"].notna().all()
        assert np.isfinite(df["mos_pred"].to_numpy()).all()
        print("auto_batch recovered from bs=48 OOM: OK", flush=True)


if __name__ == "__main__":
    main()
