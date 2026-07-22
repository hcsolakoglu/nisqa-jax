from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .checkpoint import load_model
from .config import ModelConfig
from .features import estimate_n_wins, preprocess_file
from .model import NisqaJaxModel

# Internal output name -> PyTorch-compatible CSV column. The original NISQA
# (gabrielmittag/NISQA: NISQA_model.py:76-79, NISQA_lib.py:1461-1465) writes a
# `*_pred` suffix for every head and a `model` column holding the run name.
# The TTS checkpoint exposes its head as `naturalness` internally, but PyTorch
# still reports it as `mos_pred` in the CSV, so map it accordingly.
_CSV_COLUMN_FOR_OUTPUT: dict[str, str] = {
    "mos": "mos_pred",
    "noi": "noi_pred",
    "dis": "dis_pred",
    "col": "col_pred",
    "loud": "loud_pred",
    "naturalness": "mos_pred",
}


def _format_prediction(model: NisqaJaxModel, values: np.ndarray) -> dict[str, float]:
    # Dict API (programmatic): keep clean output_names keys unchanged.
    return {name: float(values[idx]) for idx, name in enumerate(model.config.output_names)}


def _csv_row(model: NisqaJaxModel, path: Path, values: np.ndarray) -> dict:
    # Presentation layer: emit PyTorch-compatible columns (`deg`, `*_pred`, `model`).
    row: dict = {"deg": str(path)}
    for idx, name in enumerate(model.config.output_names):
        row[_CSV_COLUMN_FOR_OUTPUT[name]] = float(values[idx])
    row["model"] = model.config.source_path.stem
    return row


def default_length_bucket(cfg: ModelConfig) -> int:
    """Model-derived length-bucket grid for the batch scheduler.

    Self-attention checkpoints (``max_segments=1300``) compile fewest distinct
    shapes on a 32-grid; TTS/LSTM (``max_segments=6000``) on a coarser 64-grid.
    Callers may override via ``predict_batch(..., length_bucket=...)``; pass
    ``length_bucket=1`` to disable grid rounding (exact chunk-max).
    """
    return 32 if cfg.td == "self_att" else 64


def _round_up(value: int, bucket: int) -> int:
    """Round ``value`` up to the next multiple of ``bucket`` (1 = no rounding)."""
    if bucket <= 1:
        return value
    return int(np.ceil(value / bucket) * bucket)


def predict_file(model: NisqaJaxModel, wav_path: str | Path, *, channel: int | None = None) -> dict[str, float]:
    # segment_melspec now returns the real [n_wins, 1, n_mels, seg_length] array
    # (no max_segments padding); a single sample pads trivially to its own n_wins
    # inside device_segments (max_steps == n_wins), so no extra padding is needed.
    x, n_wins = preprocess_file(wav_path, model.config.feature, channel=channel)
    out = model.predict_segments(x[None, :], n_wins.reshape(1))[0]
    return _format_prediction(model, out)


def predict_batch(
    model: NisqaJaxModel,
    wav_paths: Sequence[str | Path],
    *,
    batch_size: int = 1,
    channel: int | None = None,
    preprocess_workers: int = 1,
    length_bucket: int | None = None,
    sort_by_length: bool = True,
) -> pd.DataFrame:
    """Length-aware batched prediction.

    Pipeline: cheap header-only n_wins pass -> stable length sort -> adjacent
    fixed-size batches -> per-chunk unpadded preprocess + pad to bucket-aligned
    batch-max -> GPU predict (ThreadPoolExecutor prefetch overlap) -> restore
    original input order. ``length_bucket=None`` selects the model-derived default
    (``default_length_bucket``); ``length_bucket=1`` disables grid rounding.
    """
    if preprocess_workers < 1:
        raise ValueError("preprocess_workers must be >= 1")
    paths = [Path(p) for p in wav_paths]
    if not paths:
        raise ValueError("No wav files provided")
    feat = model.config.feature
    if length_bucket is None:
        length_bucket = default_length_bucket(model.config)

    # Cheap first pass: estimate n_wins from audio headers (no mel-spec decode).
    # Used only for length-aware scheduling; actual padding uses real n_wins, so
    # an estimate off-by-one (only possible under resampling) is correctness-safe.
    n_wins_est = [estimate_n_wins(p, feat) for p in paths]

    # Stable sort by estimated n_wins (descending). Python's sort is stable, so
    # ties preserve original input order -> deterministic, order-restorable.
    order = sorted(range(len(paths)), key=lambda i: n_wins_est[i], reverse=True)
    if not sort_by_length:
        order = list(range(len(paths)))
    # Adjacent fixed-size chunks over the (sorted) order.
    chunked_idx = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]

    # results[orig_index] -> row dict; filled out-of-order, emitted in original order.
    results: list[dict[str, float] | None] = [None] * len(paths)

    def prepare(chunk_paths: list[Path]):
        return [preprocess_file(path, feat, channel=channel) for path in chunk_paths]

    def predict_prepared(chunk_idx: list[int], chunk_paths: list[Path], prepared) -> None:
        # prepared: list of (segments[n_wins, 1, n_mels, seg_length], n_wins).
        actual_n = np.stack([item[1] for item in prepared], axis=0)
        # Pad to the chunk's real max rounded up to the bucket grid (fewer compiles).
        max_steps = _round_up(int(actual_n.max()), length_bucket)
        # Fixed remainder padding: fill short samples with zeros (masked by n_wins);
        # if the final chunk is partial, repeat the last real sample to keep the
        # batch dimension == batch_size (avoids an extra compile for a smaller batch),
        # then discard those dummy outputs.
        bsz = len(prepared)
        if bsz < batch_size:
            # Repeat the last real sample cropped to n_wins=1 so the dummy row's
            # segment shape matches its (discarded) n_wins entry.
            last_seg = prepared[-1][0][:1]
            dummy = (last_seg, np.asarray(1, dtype=np.int32))
            prepared = list(prepared) + [dummy] * (batch_size - bsz)
            actual_n = np.concatenate([actual_n, np.asarray([1] * (batch_size - bsz), dtype=np.int32)])
        x = np.zeros((len(prepared), max_steps, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
        for j, (seg, nw) in enumerate(prepared):
            x[j, : int(nw)] = seg[: int(nw)]
        out = model.predict_segments(x, actual_n)
        for j, orig in enumerate(chunk_idx):
            results[orig] = _csv_row(model, paths[orig], out[j])

    if preprocess_workers == 1 or len(chunked_idx) < 2:
        for chunk_idx in chunked_idx:
            chunk_paths = [paths[i] for i in chunk_idx]
            predict_prepared(chunk_idx, chunk_paths, prepare(chunk_paths))
        return pd.DataFrame([r for r in results if r is not None])

    # Prefetch pipeline: overlap CPU preprocess of chunk[i+1] with GPU compute of
    # chunk[i]. Operates over the scheduled (sorted) chunk order.
    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
        first_paths = [paths[i] for i in chunked_idx[0]]
        futures = [executor.submit(preprocess_file, path, feat, channel=channel) for path in first_paths]
        for idx, chunk_idx in enumerate(chunked_idx):
            prepared = [future.result() for future in futures]
            if idx + 1 < len(chunked_idx):
                next_paths = [paths[i] for i in chunked_idx[idx + 1]]
                futures = [executor.submit(preprocess_file, path, feat, channel=channel) for path in next_paths]
            chunk_paths = [paths[i] for i in chunk_idx]
            predict_prepared(chunk_idx, chunk_paths, prepared)
    return pd.DataFrame([r for r in results if r is not None])


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    if args.mode == "predict_file":
        if args.deg is None:
            raise ValueError("--deg argument with path to input file needed")
        return [Path(args.deg)]
    if args.mode == "predict_dir":
        if args.data_dir is None:
            raise ValueError("--data_dir argument with folder with input files needed")
        return sorted(Path(args.data_dir).glob("*.wav"))
    if args.mode == "predict_csv":
        if args.csv_file is None:
            raise ValueError("--csv_file argument with csv file name needed")
        if args.csv_deg is None:
            raise ValueError("--csv_deg argument with csv column name of the filenames needed")
        data_dir = Path(args.data_dir or "")
        df = pd.read_csv(data_dir / args.csv_file)
        return [data_dir / value for value in df[args.csv_deg].tolist()]
    raise NotImplementedError("--mode given not available")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["predict_file", "predict_dir", "predict_csv"])
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--deg")
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--csv_file")
    parser.add_argument("--csv_deg")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--ms_channel", type=int)
    parser.add_argument("--device")
    parser.add_argument("--cache_dir")
    parser.add_argument("--precision", choices=["float32", "bf16"], default="float32")
    parser.add_argument("--preprocess_workers", type=int, default=1)
    parser.add_argument("--length_bucket", type=int, default=None,
                        help="pad batch-max up to this grid (default: model-derived 32 self-att / 64 TTS; 1 = exact)")
    parser.add_argument("--no_sort_by_length", action="store_true",
                        help="disable stable length sort (use naive in-order batching)")
    args = parser.parse_args(argv)

    model = load_model(args.pretrained_model, device=args.device, cache_dir=args.cache_dir, precision=args.precision)
    paths = _collect_paths(args)
    if not paths:
        raise ValueError("No wav files found")
    df = predict_batch(
        model,
        paths,
        batch_size=args.bs,
        channel=args.ms_channel,
        preprocess_workers=args.preprocess_workers,
        length_bucket=args.length_bucket,
        sort_by_length=not args.no_sort_by_length,
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "NISQA_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
