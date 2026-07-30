from __future__ import annotations

from pathlib import Path

import librosa as lb
import numpy as np
import soundfile as sf

from .config import FeatureConfig


def _validate_channel(channel: int | None) -> None:
    """Channel must be None or a true integer (not bool, not float)."""
    if channel is None:
        return
    # bool is a subclass of int; reject it so True/False are not silently treated
    # as channel 1/0. Floats (e.g. 0.0) are also rejected — a channel index is integral.
    if isinstance(channel, bool) or not isinstance(channel, int):
        raise ValueError(f"channel must be None or an int (not bool/float), got {channel!r} ({type(channel).__name__})")


def load_melspec(file_path: str | Path, cfg: FeatureConfig, *, channel: int | None = None) -> np.ndarray:
    path = Path(file_path)
    _validate_channel(channel)
    try:
        if channel is None:
            y, sr = lb.load(path, sr=cfg.sr)
        else:
            y, sr = lb.load(path, sr=cfg.sr, mono=False)
    except Exception as exc:  # pragma: no cover - preserves original error shape.
        raise ValueError(f"Could not load file {path}") from exc

    # Select channel outside the load try-block so an out-of-range index surfaces as a precise
    # error instead of being swallowed into a generic "Could not load file". The guard must
    # run whenever a channel is requested, even for mono (ndim==1) files: previously the check
    # sat inside `y.ndim > 1`, so `load_melspec(mono.wav, channel=5)` was silently accepted.
    if channel is not None:
        channel_count = y.shape[0] if y.ndim > 1 else 1
        if channel < 0 or channel >= channel_count:
            raise ValueError(f"Channel {channel} out of range for file with {channel_count} channels: {path}")
        if channel_count > 1:
            y = y[channel, :]

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
    if cfg.seg_hop_length < 1:
        raise ValueError(f"seg_hop_length must be >= 1, got {cfg.seg_hop_length}")

    n_wins_raw = spec.shape[1] - (cfg.seg_length - 1)
    if n_wins_raw < 1:
        raise ValueError(
            f"Sample too short. Only {spec.shape[1]} windows available but seg_length={cfg.seg_length}. "
            f"Consider zero padding the audio sample. File: {file_path}"
        )

    # Reject unsupported lengths before allocating the window index or copied
    # segment tensor. For long audio, constructing every overlapping window
    # first can exhaust host memory before this documented limit is reached.
    n_wins = (n_wins_raw + cfg.seg_hop_length - 1) // cfg.seg_hop_length
    if cfg.max_segments < n_wins:
        raise ValueError(
            f"n_wins {n_wins} > max_length {cfg.max_segments} --- {file_path}. "
            "Increase max window length ms_max_segments!"
        )

    # Construct only the hop-selected windows. Building all n_wins_raw windows
    # and slicing afterward amplifies memory by seg_hop_length for no benefit.
    starts = np.arange(0, n_wins_raw, cfg.seg_hop_length)
    idx = starts[:, None] + np.arange(cfg.seg_length)[None, :]
    segments = spec.T[idx, :].transpose(0, 2, 1)[:, None, :, :]

    # Reject non-finite mel-spectrograms with a path-bearing message: a corrupt
    # or all-silent WAV can yield NaN/Inf bins that would silently propagate to
    # NaN scores. predict_segments also guards this, but only the feature path
    # knows the originating file path.
    if not np.isfinite(segments).all():
        n_bad = int(np.sum(~np.isfinite(segments)))
        raise ValueError(
            f"Mel-spectrogram contains {n_bad} non-finite (NaN/Inf) values for file: {file_path}. "
            "The audio may be corrupt or empty."
        )

    # Return ONLY the real [n_wins, 1, n_mels, seg_length] segments (no max_segments
    # zero-padding). Padding is deferred to batch-assembly time in predict_batch so
    # host-RAM is not wasted on per-file zero buffers (was 3.74MB/file self-att,
    # 17.28MB/file TTS). predict_file pads trivially to its own n_wins.
    return np.ascontiguousarray(segments, dtype=np.float32), np.asarray(n_wins, dtype=np.int32)


def preprocess_file(
    file_path: str | Path,
    cfg: FeatureConfig,
    *,
    channel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    spec = load_melspec(file_path, cfg, channel=channel)
    return segment_melspec(file_path, spec, cfg)


def estimate_n_wins(file_path: str | Path, cfg: FeatureConfig) -> int:
    """Cheap n_wins estimate from the audio header only (no mel-spec decode).

    Reads just the file metadata via ``soundfile.info`` and replicates librosa's
    framing (``center=True``: ``n_frames = 1 + n_samples // hop``) to derive the
    segment-window count without loading/decoding audio. Used by the length-aware
    batch scheduler for *sorting/grouping only*; the actual per-chunk padding max
    is computed from the real preprocessed ``n_wins``, so an off-by-one here (only
    possible when ``cfg.sr`` triggers resampling) is harmless for correctness and
    at worst slightly increases padding. For the shipped checkpoints ``sr=None``
    (no resampling) so the estimate is exact.
    """
    info = sf.info(str(file_path))
    native_sr = info.samplerate
    n_samples = info.frames
    target_sr = cfg.sr if cfg.sr is not None else native_sr
    if target_sr != native_sr:
        # librosa resamples to target_sr; sample count is the rounded ratio.
        n_samples = int(round(n_samples * target_sr / native_sr))
    hop_length = int(target_sr * cfg.hop_length_seconds)
    n_frames = 1 + n_samples // hop_length  # center=True reflect padding
    n_wins_raw = n_frames - (cfg.seg_length - 1)
    if n_wins_raw < 1:
        raise ValueError(
            f"Sample too short. Only {n_frames} frames available but seg_length={cfg.seg_length}. "
            f"Consider zero padding the audio sample. File: {file_path}"
        )
    n_wins = int(np.ceil(n_wins_raw / cfg.seg_hop_length))
    if cfg.max_segments < n_wins:
        raise ValueError(
            f"n_wins {n_wins} > max_length {cfg.max_segments} --- {file_path}. "
            "Increase max window length ms_max_segments!"
        )
    return n_wins
