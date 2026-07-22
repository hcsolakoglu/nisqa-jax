from __future__ import annotations

from pathlib import Path

import librosa as lb
import numpy as np

from .config import FeatureConfig


def load_melspec(file_path: str | Path, cfg: FeatureConfig, *, channel: int | None = None) -> np.ndarray:
    path = Path(file_path)
    try:
        if channel is None:
            y, sr = lb.load(path, sr=cfg.sr)
        else:
            y, sr = lb.load(path, sr=cfg.sr, mono=False)
            if y.ndim > 1:
                y = y[channel, :]
    except Exception as exc:  # pragma: no cover - preserves original error shape.
        raise ValueError(f"Could not load file {path}") from exc

    hop_length = int(sr * cfg.hop_length_seconds)
    win_length = int(sr * cfg.win_length_seconds)
    spec = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=cfg.n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=cfg.n_mels,
        fmin=0.0,
        fmax=cfg.fmax,
        htk=False,
        norm="slaney",
    )
    return lb.core.amplitude_to_db(spec, ref=1.0, amin=1e-4, top_db=80.0).astype(np.float32)


def segment_melspec(
    file_path: str | Path,
    spec: np.ndarray,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.seg_length % 2 == 0:
        raise ValueError(f"seg_length must be odd! (seg_lenth={cfg.seg_length})")
    if spec.ndim != 2:
        raise ValueError(f"Expected mel spectrogram [mels, frames], got shape {spec.shape}")

    n_wins_raw = spec.shape[1] - (cfg.seg_length - 1)
    if n_wins_raw < 1:
        raise ValueError(
            f"Sample too short. Only {spec.shape[1]} windows available but seg_length={cfg.seg_length}. "
            f"Consider zero padding the audio sample. File: {file_path}"
        )

    idx = np.arange(cfg.seg_length)[None, :] + np.arange(n_wins_raw)[:, None]
    segments = spec.T[idx, :].transpose(0, 2, 1)[:, None, :, :]
    if cfg.seg_hop_length > 1:
        segments = segments[:: cfg.seg_hop_length, :]
    n_wins = int(np.ceil(n_wins_raw / cfg.seg_hop_length))

    if cfg.max_segments < n_wins:
        raise ValueError(
            f"n_wins {n_wins} > max_length {cfg.max_segments} --- {file_path}. "
            "Increase max window length ms_max_segments!"
        )

    padded = np.zeros((cfg.max_segments, segments.shape[1], segments.shape[2], segments.shape[3]), dtype=np.float32)
    padded[:n_wins, :] = segments.astype(np.float32)
    return padded, np.asarray(n_wins, dtype=np.int32)


def preprocess_file(
    file_path: str | Path,
    cfg: FeatureConfig,
    *,
    channel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    spec = load_melspec(file_path, cfg, channel=channel)
    return segment_melspec(file_path, spec, cfg)
