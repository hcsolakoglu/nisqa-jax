from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .checkpoint import load_model, prewarm
from .config import ModelConfig
from .features import estimate_n_wins, preprocess_file
from .model import NisqaJaxModel

logger = logging.getLogger(__name__)

# JAX device OOM surfaces as jaxlib.ResourceExhaustedError, or as a
# jaxlib.XlaRuntimeError / plain RuntimeError whose message carries the XLA
# status token ``RESOURCE_EXHAUSTED``. We match on the dedicated
# ResourceExhaustedError type (always OOM) and on the OOM token in the message
# (covers XlaRuntimeError and RuntimeError across jaxlib versions and backends:
# GPU, TPU, and CPU). We intentionally do NOT treat every XlaRuntimeError as
# OOM: a non-OOM XLA status (INVALID_ARGUMENT, UNIMPLEMENTED, ...) on any
# backend would otherwise be misclassified and trigger a futile auto_batch
# retry. The message-token check is backend-portable: TPU/GPU OOM messages
# contain ``RESOURCE_EXHAUSTED``/``out of memory``; CPU does not raise OOM
# (it over-commits), so non-OOM CPU errors never match.
_OOM_TOKENS = ("resourceexhausted", "out of memory", "oom")


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "resourceexhausted" in name:
        return True
    return any(tok in str(exc).lower() for tok in _OOM_TOKENS)

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


def _error_row(model: NisqaJaxModel, path: Path, message: str) -> dict:
    # collect-mode row for a file that failed preprocessing: prediction columns
    # are NaN, `model` is None, and `error` carries the exception message so the
    # caller sees a complete per-file record in a single DataFrame.
    row: dict = {"deg": str(path)}
    for name in model.config.output_names:
        row[_CSV_COLUMN_FOR_OUTPUT[name]] = np.nan
    row["model"] = None
    row["error"] = message
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
    on_error: str = "raise",
    auto_batch: bool = False,
) -> pd.DataFrame:
    """Length-aware batched prediction.

    Pipeline: cheap header-only n_wins pass -> stable length sort -> adjacent
    fixed-size batches -> per-chunk unpadded preprocess + pad to bucket-aligned
    batch-max -> GPU predict (ThreadPoolExecutor prefetch overlap) -> restore
    original input order. ``length_bucket=None`` selects the model-derived default
    (``default_length_bucket``); ``length_bucket=1`` disables grid rounding.

    Error handling (``on_error``):
      * ``"raise"`` (default): the first file that fails preprocessing aborts the
        whole batch; the exception is re-raised wrapped with the failing file
        path. Preserves the original fail-fast behavior with a clearer message.
      * ``"collect"``: failed files are skipped, the rest of the batch completes,
        and the returned DataFrame contains one row per input file in original
        order — successful rows carry their predictions with ``error=NaN``;
        failed rows carry ``NaN`` predictions and an ``error`` message string.
        A single corrupt file therefore never discards the completed rows.

    Memory recovery (``auto_batch``): on a JAX GPU out-of-memory
    (``ResourceExhaustedError``) during a chunk's forward pass, the chunk is
    halved and retried (recursively, down to a single sample) and each reduction
    is logged. Off by default; only the failing chunk is re-run, completed
    chunks are kept.
    """
    if preprocess_workers < 1:
        raise ValueError("preprocess_workers must be >= 1")
    if on_error not in {"raise", "collect"}:
        raise ValueError(f"on_error must be 'raise' or 'collect', got {on_error!r}")
    paths = [Path(p) for p in wav_paths]
    if not paths:
        raise ValueError("No wav files provided")
    feat = model.config.feature
    if length_bucket is None:
        length_bucket = default_length_bucket(model.config)
    collect = on_error == "collect"

    # results[orig_index] -> row dict (success or error row); emitted in original
    # input order. In collect mode a failure fills an error row; in raise mode it
    # re-raises immediately, so only successful rows ever accumulate.
    results: list[dict | None] = [None] * len(paths)

    def _record_error(orig: int, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if collect:
            results[orig] = _error_row(model, paths[orig], message)
        else:
            # raise mode: surface the failing path explicitly so a corrupt wav
            # among thousands is identifiable without re-running one-by-one.
            raise RuntimeError(f"predict_batch failed on file {paths[orig]}: {exc}") from exc

    # Cheap first pass: estimate n_wins from audio headers (no mel-spec decode).
    # Used only for length-aware scheduling; actual padding uses real n_wins, so
    # an estimate off-by-one (only possible under resampling) is correctness-safe.
    # In collect mode a too-short/too-long file is recorded and excluded here so
    # it never reaches the (batched) GPU stage.
    n_wins_est: list[int] = []
    ok_idx: list[int] = []
    for i, p in enumerate(paths):
        try:
            n_wins_est.append(estimate_n_wins(p, feat))
            ok_idx.append(i)
        except Exception as exc:
            n_wins_est.append(-1)
            _record_error(i, exc)

    # Stable sort by estimated n_wins (descending) over the viable files only.
    # Python's sort is stable, so ties preserve original input order.
    order = sorted(ok_idx, key=lambda i: n_wins_est[i], reverse=True)
    if not sort_by_length:
        order = sorted(ok_idx)
    # Adjacent fixed-size chunks over the (sorted) order.
    chunked_idx = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]

    def _predict_padded(prepared: list, cur_bs: int) -> np.ndarray:
        # prepared: list of (segments[n_wins, 1, n_mels, seg_length], n_wins).
        # Pad to the chunk's real max rounded up to the bucket grid (fewer compiles).
        actual_n = np.stack([item[1] for item in prepared], axis=0)
        max_steps = _round_up(int(actual_n.max()), length_bucket)
        bsz = len(prepared)
        # Fixed remainder padding: fill short samples with zeros (masked by n_wins);
        # if the chunk is smaller than cur_bs, repeat the last real sample to keep
        # the batch dimension == cur_bs (avoids an extra compile for a smaller
        # batch), then discard those dummy outputs.
        if bsz < cur_bs:
            last_seg = prepared[-1][0][:1]
            dummy = (last_seg, np.asarray(1, dtype=np.int32))
            prepared = list(prepared) + [dummy] * (cur_bs - bsz)
            actual_n = np.concatenate([actual_n, np.asarray([1] * (cur_bs - bsz), dtype=np.int32)])
        x = np.zeros((len(prepared), max_steps, 1, feat.n_mels, feat.seg_length), dtype=np.float32)
        for j, (seg, nw) in enumerate(prepared):
            x[j, : int(nw)] = seg[: int(nw)]
        out = model.predict_segments(x, actual_n)
        return np.asarray(out)[:bsz]  # discard dummy rows

    def _predict_with_auto_batch(prepared: list, cur_bs: int) -> np.ndarray:
        try:
            return _predict_padded(prepared, cur_bs)
        except Exception as exc:
            if not auto_batch or not _is_oom(exc) or cur_bs <= 1:
                raise
            new_bs = max(1, cur_bs // 2)
            logger.warning(
                "auto_batch: GPU OOM at batch_size=%d (%d samples); retrying at %d",
                cur_bs, len(prepared), new_bs,
            )
            outs = [
                _predict_with_auto_batch(prepared[s : s + new_bs], new_bs)
                for s in range(0, len(prepared), new_bs)
            ]
            return np.concatenate(outs, axis=0)

    def _store(chunk_idx: list[int], prepared: list) -> None:
        outs = _predict_with_auto_batch(prepared, batch_size)
        for j, orig in enumerate(chunk_idx):
            results[orig] = _csv_row(model, paths[orig], outs[j])

    def _prepare_one(orig: int):
        try:
            return preprocess_file(paths[orig], feat, channel=channel)
        except Exception as exc:
            _record_error(orig, exc)
            return None

    def predict_prepared(chunk_idx: list[int], prepared) -> None:
        # prepared may contain None placeholders for files that failed in the
        # prefetch path (collect mode); drop them and keep the surviving indices.
        pairs = [(idx, item) for idx, item in zip(chunk_idx, prepared) if item is not None]
        if not pairs:
            return
        live_idx = [p[0] for p in pairs]
        live_prepared = [p[1] for p in pairs]
        _store(live_idx, live_prepared)

    if preprocess_workers == 1 or len(chunked_idx) < 2:
        for chunk_idx in chunked_idx:
            prepared = [_prepare_one(orig) for orig in chunk_idx]
            predict_prepared(chunk_idx, prepared)
        return _emit(results, collect)

    # Prefetch pipeline: overlap CPU preprocess of chunk[i+1] with GPU compute of
    # chunk[i]. Operates over the scheduled (sorted) chunk order. In collect mode
    # a per-file failure is recorded and that file is dropped from its chunk.
    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
        futures = {orig: executor.submit(_prepare_one, orig) for orig in chunked_idx[0]}
        for idx, chunk_idx in enumerate(chunked_idx):
            prepared = [futures.pop(orig).result() for orig in chunk_idx]
            if idx + 1 < len(chunked_idx):
                futures = {orig: executor.submit(_prepare_one, orig) for orig in chunked_idx[idx + 1]}
            predict_prepared(chunk_idx, prepared)
    return _emit(results, collect)


def _emit(results: list[dict | None], collect: bool) -> pd.DataFrame:
    # Emit rows in original input order. In collect mode every input file has a
    # row (success or error); in raise mode only successfully predicted rows
    # exist (a failure would have re-raised already).
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)
    if collect and "error" in df.columns:
        # Stable column order: deg, *_pred, model, error.
        cols = [c for c in df.columns if c != "error"] + ["error"]
        df = df[cols]
    return df


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
    parser.add_argument("--on_error", choices=["raise", "collect"], default="raise",
                        help="raise: abort batch on first bad file (default); "
                             "collect: skip bad files, add an `error` column")
    parser.add_argument("--auto_batch", action="store_true",
                        help="on GPU OOM, halve batch_size and retry down to 1 (logs each reduction)")
    parser.add_argument("--prewarm", action="store_true",
                        help="pre-compile the model's default length-bucket grid at --bs before "
                             "predicting, so the first real batch hits the persistent cache")
    args = parser.parse_args(argv)

    model = load_model(args.pretrained_model, device=args.device, cache_dir=args.cache_dir, precision=args.precision)
    if args.prewarm:
        # Warm the persistent cache for the model's default bucket grid at the
        # requested batch size so the first real batch of that shape is a cache
        # hit (no compile stall). Uses dummy zeros; output is discarded.
        prewarm(model, [args.bs], [default_length_bucket(model.config)], cache_dir=args.cache_dir)
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
        on_error=args.on_error,
        auto_batch=args.auto_batch,
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "NISQA_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
