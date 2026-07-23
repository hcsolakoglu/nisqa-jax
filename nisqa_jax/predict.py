from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

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


def _model_identity(cfg: ModelConfig) -> str:
    """CSV `model` column value: the run/checkpoint label.

    Preference order:
      1. ``source_name`` — the checkpoint lane's explicit run label (added by
         the checkpoint-loading lane from the training-run ``name`` arg). This
         is the canonical identity when present.
      2. ``display_name`` / ``model_label`` — alternate explicit-label attrs a
         future lane may add; kept for forward compatibility.
      3. ``source_path.stem`` — the checkpoint filename stem, so this module
         cherry-picks independently of either lane.

    The architecture ``model_name`` (``NISQA``/``NISQA_DIM``) is intentionally
    NOT used here — PyTorch writes the run label, not the architecture class.
    """
    for attr in ("source_name", "display_name", "model_label"):
        value = getattr(cfg, attr, None)
        if value:
            return str(value)
    return cfg.source_path.stem


def _format_prediction(model: NisqaJaxModel, values: np.ndarray) -> dict[str, float]:
    # Dict API (programmatic): keep clean output_names keys unchanged.
    return {name: float(values[idx]) for idx, name in enumerate(model.config.output_names)}


def _csv_row(model: NisqaJaxModel, path: Path, values: np.ndarray) -> dict:
    # Presentation layer: emit PyTorch-compatible columns (`deg`, `*_pred`, `model`).
    row: dict = {"deg": str(path)}
    for idx, name in enumerate(model.config.output_names):
        row[_CSV_COLUMN_FOR_OUTPUT[name]] = float(values[idx])
    row["model"] = _model_identity(model.config)
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


def default_prewarm_grid(
    cfg: ModelConfig,
    bucket: int | None = None,
    *,
    max_segments: int | None = None,
) -> list[int]:
    """A sensible, documented default length-bucket grid to pre-compile.

    Returns a sorted list of bucket-aligned sequence lengths to warm the JIT
    cache for, covering the model's full ``max_segments`` range without
    pre-compiling every multiple (which would be ~40 for self_att / ~94 for TTS
    and dominate startup). The grid is a geometric progression (doubling from
    the bucket size) plus ``max_segments`` itself so the longest-possible real
    batch is always a cache hit. ``predict_batch`` caps bucket rounding at
    ``max_segments``, so the cap shape is exactly what a longest-batch chunk
    compiles to:

      self_att (bucket=32, max=1300) -> [32, 64, 128, 256, 512, 1024, 1300]
      tts      (bucket=64, max=6000) -> [64, 128, 256, 512, 1024, 2048, 4096, 6000]

    This is a heuristic, not an exhaustive cover: real traffic may compile a few
    extra in-between bucket multiples on first hit (the persistent cache then
    retains them). Callers wanting full control pass explicit ``bucket_lengths``
    to ``prewarm``. ``bucket=1`` yields ``[1, 2, 4, ..., max_segments]``.
    """
    if bucket is None:
        bucket = default_length_bucket(cfg)
    cap = int(max_segments if max_segments is not None else cfg.feature.max_segments)
    if bucket < 1:
        raise ValueError(f"bucket must be >= 1, got {bucket}")
    if cap < 1:
        raise ValueError(f"max_segments must be >= 1, got {cap}")
    grid: list[int] = []
    length = max(1, bucket)
    while length < cap:
        grid.append(length)
        length *= 2
    # Always include the exact max (the longest real batch, also the bucket-
    # rounding cap inside predict_batch) and dedup/sort.
    grid.append(cap)
    return sorted(set(grid))


def _validate_batch_size(batch_size: Any) -> int:
    """batch_size must be a true integer >= 1 (bool/float rejected)."""
    # bool is a subclass of int — reject True/False so they are not silently 1/0.
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError(
            f"batch_size must be an int >= 1, got {batch_size!r} ({type(batch_size).__name__})"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return batch_size


def _validate_positive_int(value: Any, name: str) -> int:
    """A positive-integer parameter (bool/float rejected).

    Used for ``length_bucket`` and ``preprocess_workers``: bool is a subclass of
    int, so ``True`` would silently be 1 and ``False`` would be 0 (then fail the
    >= 1 check with a confusing message). Reject non-int types explicitly.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an int >= 1, got {value!r} ({type(value).__name__})"
        )
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _cost_exponent(td: str) -> int:
    """Padded-compute cost exponent for the temporal-dependency architecture.

    The cost proxy of a padded chunk is ``batch_size * max_length**exponent``:
      * ``self_att``: padded self-attention is ``B * L^2`` (every padded query
        attends to every padded key), so ``exponent=2`` — long outliers are
        isolated quadratically more aggressively.
      * ``lstm``: the LSTM scan is ``B * L`` (linear in sequence length), so
        ``exponent=1`` — outliers are isolated proportionally.

    This is a cost-proxy heuristic for batch sizing, not a claim about absolute
    FLOPs (which also depend on d_model, #layers, CNN front-end, etc.) or a
    global optimum. It only controls how aggressively the scheduler shrinks the
    batch for a long-tail chunk.
    """
    return 2 if td == "self_att" else 1


def _cost_aware_batch_size(length: int, ref_len: int, max_bs: int, *, exponent: int = 1) -> int:
    """Largest power-of-two batch size <= max_bs that isolates long outliers.

    The cost proxy of a chunk is ``batch_size * max_length**exponent`` (padded
    compute). To keep a heavy-tailed outlier from inflating a whole full-size
    chunk, the batch size for a chunk starting at length ``length`` is shrunk
    so that ``bs * length**exponent`` stays near ``max_bs * ref_len**exponent``:
    ``bs ≈ max_bs * (ref_len / length)**exponent``. With ``exponent=1`` (LSTM)
    a file ``k``x longer than the median runs in a batch ``~k``x smaller; with
    ``exponent=2`` (self_att) it runs in a batch ``~k^2``x smaller — isolating
    the quadratic-attention long tail far more aggressively. The result is
    bounded to the power-of-two set {1, 2, ..., max_bs}. This is a conservative
    heuristic — not a global optimum — but it deterministically isolates the
    long tail while keeping short files in full batches.
    """
    if max_bs < 1:
        return 1
    ratio = ref_len / max(length, 1)
    cap = max(1, int(max_bs * (ratio ** exponent)))
    p = 1
    while p * 2 <= cap and p * 2 <= max_bs:
        p *= 2
    return p


def _fixed_chunks(order: list[int], batch_size: int) -> list[list[int]]:
    """Adjacent fixed-size chunks over the scheduled order."""
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def _cost_aware_chunks(
    order: list[int], n_wins_est: list[int], batch_size: int, *, exponent: int = 1
) -> list[tuple[list[int], int]]:
    """Cost-aware chunking: vary batch size by length tier, preserve sorted order.

    ``order`` is sorted by ``n_wins_est`` descending (stable). Walk it left to
    right; each chunk's batch size is derived from its first (longest) member's
    length via ``_cost_aware_batch_size`` (using ``exponent``). The chunk then
    absorbs that many consecutive files. Returns ``(indices, compile_batch_size)``
    tuples matching the fixed-mode chunk shape. Deterministic and order-
    preserving; per-file results are identical to fixed mode (only grouping
    changes). ``exponent`` comes from ``_cost_exponent(cfg.td)``: 2 for
    self_att (B*L^2 cost), 1 for LSTM (B*L cost).
    """
    if not order:
        return []
    lengths = [n_wins_est[i] for i in order]
    # Reference length = median of the viable files' estimated lengths. Using the
    # median (not min) keeps the short majority in full batches while isolating a
    # heavy upper tail; using max would make every chunk bs=1.
    ref_len = int(np.median(lengths))
    chunks: list[tuple[list[int], int]] = []
    i = 0
    n = len(order)
    while i < n:
        bs_i = _cost_aware_batch_size(int(lengths[i]), ref_len, batch_size, exponent=exponent)
        end = min(n, i + bs_i)
        chunks.append((order[i:end], bs_i))
        i = end
    return chunks


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
    batch_mode: str = "fixed",
) -> pd.DataFrame:
    """Length-aware batched prediction.

    Pipeline: cheap header-only n_wins pass -> stable length sort -> chunking ->
    per-chunk unpadded preprocess + pad to bucket-aligned batch-max -> device
    predict (ThreadPoolExecutor prefetch overlap) -> restore original input
    order. ``length_bucket=None`` selects the model-derived default
    (``default_length_bucket``); ``length_bucket=1`` disables grid rounding.

    The bucket-rounded padded sequence dimension is passed explicitly to
    ``predict_segments`` via ``padded_steps`` so it survives into the JIT compile
    key: every distinct actual length that rounds to the same bucket reuses one
    compiled program / cache entry, while ``n_wins`` masks the padding so scores
    are bit-identical to an exact-length run.

    Batching (``batch_mode``):
      * ``"fixed"`` (default, compatibility): adjacent fixed-size chunks of
        ``batch_size`` over the sorted order.
      * ``"cost_aware"``: a conservative cost-aware scheduler for heavy-tailed
        length distributions. Uses a bounded set of batch sizes (powers of two
        up to ``batch_size``) so a long outlier runs in a small batch instead of
        inflating a whole full-size chunk's padded compute. The cost proxy is
        model-aware: ``batch_size * max_length**2`` for ``self_att`` (padded
        attention is quadratic) and ``batch_size * max_length`` for ``lstm``
        (linear scan), so self_att isolates long outliers quadratically more
        aggressively than LSTM. Deterministic and order-preserving; per-file
        scores are identical to fixed mode (only grouping changes). This is a
        heuristic — not a global optimum / DP solution — that reduces the
        cost proxy. Requires ``sort_by_length=True``.

    ``batch_size`` is clamped to ``len(wav_paths)`` for execution: a batch
    larger than the dataset would only pad with dummy rows (a pathological
    allocation), so the useful executable batch shape is ``min(batch_size,
    len(paths))``. The requested ``batch_size`` still controls the cost-aware
    tier ceiling and the fixed chunk size up to that clamp.

    Error handling (``on_error``):
      * ``"raise"`` (default): the first file that fails preprocessing aborts the
        whole batch; the exception is re-raised wrapped with the failing file
        path. Preserves the original fail-fast behavior with a clearer message.
      * ``"collect"``: failed files are skipped, the rest of the batch completes,
        and the returned DataFrame contains one row per input file in original
        order with a stable schema that always includes an ``error`` column —
        successful rows carry their predictions with ``error=NaN``; failed rows
        carry ``NaN`` predictions and an ``error`` message string that includes
        the failing file path. A single corrupt file therefore never discards
        the completed rows. NOTE: global model-forward failures (e.g. device
        OOM, XLA errors) are chunk-level, not per-file, and are NOT collected —
        they abort the batch even in ``collect`` mode (only ``auto_batch`` can
        recover an OOM, and only by reducing the batch size).

    Memory recovery (``auto_batch``): on a JAX GPU out-of-memory
    (``ResourceExhaustedError``) during a chunk's forward pass, the chunk is
    halved and retried (recursively, down to a single sample) and each reduction
    is logged. Off by default; only the failing chunk is re-run, completed
    chunks are kept.
    """
    if on_error not in {"raise", "collect"}:
        raise ValueError(f"on_error must be 'raise' or 'collect', got {on_error!r}")
    if batch_mode not in {"fixed", "cost_aware"}:
        raise ValueError(f"batch_mode must be 'fixed' or 'cost_aware', got {batch_mode!r}")
    if batch_mode == "cost_aware" and not sort_by_length:
        raise ValueError(
            "batch_mode='cost_aware' requires sort_by_length=True "
            "(it isolates the long tail by sorted rank)"
        )
    batch_size = _validate_batch_size(batch_size)
    preprocess_workers = _validate_positive_int(preprocess_workers, "preprocess_workers")
    if length_bucket is not None:
        length_bucket = _validate_positive_int(length_bucket, "length_bucket")
    paths = [Path(p) for p in wav_paths]
    if not paths:
        raise ValueError("No wav files provided")
    feat = model.config.feature
    if length_bucket is None:
        length_bucket = default_length_bucket(model.config)
    collect = on_error == "collect"
    # Clamp the executable batch to the dataset size: a batch larger than the
    # number of real samples can only be filled with dummy rows, which is a
    # pathological allocation (DoS) with no cache-stability benefit. This is the
    # only ceiling applied — it is principled (cannot exceed real sample count),
    # not an arbitrary magic number.
    effective_bs = min(batch_size, len(paths))

    # results[orig_index] -> row dict (success or error row); emitted in original
    # input order. In collect mode a failure fills an error row; in raise mode it
    # re-raises immediately, so only successful rows ever accumulate.
    results: list[dict | None] = [None] * len(paths)

    def _record_error(orig: int, exc: BaseException) -> None:
        # The error string itself carries the path so a row's `error` cell is
        # self-describing even if `deg` is later reordered/dropped.
        message = f"{paths[orig]}: {type(exc).__name__}: {exc}"
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
    # Chunk the scheduled order. Each chunk carries its compile batch size
    # (the padded batch dimension) for cache-stable dummy padding. The cost-aware
    # scheduler's exponent is model-derived: self_att pads B*L^2 (quadratic
    # attention), LSTM pads B*L (linear scan).
    if batch_mode == "cost_aware":
        exponent = _cost_exponent(model.config.td)
        chunked_idx = _cost_aware_chunks(order, n_wins_est, effective_bs, exponent=exponent)
    else:
        chunked_idx = [(chunk, effective_bs) for chunk in _fixed_chunks(order, effective_bs)]

    def _predict_padded(prepared: list, cur_bs: int) -> np.ndarray:
        # prepared: list of (segments[n_wins, 1, n_mels, seg_length], n_wins).
        # Pad to the chunk's real max rounded up to the bucket grid (fewer
        # compiles), capped at max_segments so the longest batch compiles at the
        # model's documented ceiling (matches default_prewarm_grid's final entry).
        actual_n = np.stack([item[1] for item in prepared], axis=0)
        max_steps = min(_round_up(int(actual_n.max()), length_bucket), feat.max_segments)
        bsz = len(prepared)
        # cur_bs is already clamped to <= len(paths) (effective_bs) and halves
        # only inside auto_batch, so dummy padding can never balloon far beyond
        # the real sample count.
        cur_bs = min(cur_bs, bsz) if cur_bs > bsz else cur_bs
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
        # padded_steps=max_steps keeps the bucket-rounded time axis through the
        # JIT (device_segments otherwise crops to max(n_wins), defeating bucketing).
        out = model.predict_segments(x, actual_n, padded_steps=max_steps)
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

    def _store(chunk_idx: list[int], prepared: list, cur_bs: int) -> None:
        outs = _predict_with_auto_batch(prepared, cur_bs)
        for j, orig in enumerate(chunk_idx):
            results[orig] = _csv_row(model, paths[orig], outs[j])

    def _prepare_one(orig: int):
        try:
            return preprocess_file(paths[orig], feat, channel=channel)
        except Exception as exc:
            _record_error(orig, exc)
            return None

    def predict_prepared(chunk_idx: list[int], prepared, cur_bs: int) -> None:
        # prepared may contain None placeholders for files that failed in the
        # prefetch path (collect mode); drop them and keep the surviving indices.
        pairs = [(idx, item) for idx, item in zip(chunk_idx, prepared, strict=True) if item is not None]
        if not pairs:
            return
        live_idx = [p[0] for p in pairs]
        live_prepared = [p[1] for p in pairs]
        # If files dropped in collect mode, do not pad the chunk beyond the
        # survivors' count for the compile shape — fall back to the survivor
        # count so no pathological dummy allocation occurs.
        run_bs = min(cur_bs, len(live_prepared)) if len(live_prepared) < cur_bs else cur_bs
        _store(live_idx, live_prepared, run_bs)

    if preprocess_workers == 1 or len(chunked_idx) < 2:
        for chunk_idx, cur_bs in chunked_idx:
            prepared = [_prepare_one(orig) for orig in chunk_idx]
            predict_prepared(chunk_idx, prepared, cur_bs)
        return _emit(results, collect, model)

    # Prefetch pipeline: overlap CPU preprocess of chunk[i+1] with GPU compute of
    # chunk[i]. Operates over the scheduled (sorted) chunk order. In collect mode
    # a per-file failure is recorded and that file is dropped from its chunk.
    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
        first_idx = chunked_idx[0][0]
        futures = {orig: executor.submit(_prepare_one, orig) for orig in first_idx}
        for idx, (chunk_idx, cur_bs) in enumerate(chunked_idx):
            prepared = [futures.pop(orig).result() for orig in chunk_idx]
            if idx + 1 < len(chunked_idx):
                next_idx = chunked_idx[idx + 1][0]
                futures = {orig: executor.submit(_prepare_one, orig) for orig in next_idx}
            predict_prepared(chunk_idx, prepared, cur_bs)
    return _emit(results, collect, model)


def _emit(results: list[dict | None], collect: bool, model: NisqaJaxModel) -> pd.DataFrame:
    # Emit rows in original input order. In collect mode every input file has a
    # row (success or error) and the schema is STABLE: the `error` column is
    # always present (NaN for successful rows) so downstream consumers can rely
    # on a constant column set regardless of whether any file failed. In raise
    # mode only successfully predicted rows exist (a failure would have
    # re-raised already) and no `error` column is synthesized.
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)
    if collect:
        if "error" not in df.columns:
            df["error"] = np.nan
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
        csv_path = data_dir / args.csv_file
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise ValueError(f"CSV file not found: {csv_path}") from exc
        if args.csv_deg not in df.columns:
            raise ValueError(
                f"csv_deg column {args.csv_deg!r} not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
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
    parser.add_argument("--batch_mode", choices=["fixed", "cost_aware"], default="fixed",
                        help="fixed (default): adjacent fixed-size chunks of --bs; "
                             "cost_aware: bounded power-of-two batch sizes that isolate long "
                             "outliers (reduces padded-compute cost on heavy-tailed lengths; "
                             "requires length sort, scores identical to fixed)")
    parser.add_argument("--prewarm", action="store_true",
                        help="pre-compile the model's default length-bucket grid at --bs before "
                             "predicting, so the first real batch of each grid shape hits the "
                             "persistent cache. Grid: bucket-aligned doubling lengths up to "
                             "max_segments (see default_prewarm_grid)")
    args = parser.parse_args(argv)

    model = load_model(args.pretrained_model, device=args.device, cache_dir=args.cache_dir, precision=args.precision)
    if args.prewarm:
        # Warm the persistent cache for a documented bucket grid (doubling
        # bucket-aligned lengths up to max_segments) at the requested batch size,
        # so the first real batch of each grid shape is a cache hit (no compile
        # stall). Previously this prewarmed a single length (the bucket size
        # itself), which left every other shape cold. Uses dummy zeros; output
        # is discarded.
        bucket = args.length_bucket if args.length_bucket is not None else default_length_bucket(model.config)
        grid = default_prewarm_grid(model.config, bucket)
        prewarm(model, [args.bs], grid, cache_dir=args.cache_dir)
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
        batch_mode=args.batch_mode,
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "NISQA_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
