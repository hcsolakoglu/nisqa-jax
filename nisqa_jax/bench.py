from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoint import load_model
from .features import preprocess_file
from .predict import predict_batch


def _collect_paths(args: argparse.Namespace) -> list[Path] | None:
    if args.csv_file is not None:
        if args.csv_deg is None:
            raise ValueError("--csv_deg is required when --csv_file is used")
        data_dir = Path(args.data_dir or "")
        df = pd.read_csv(data_dir / args.csv_file)
        return [data_dir / value for value in df[args.csv_deg].tolist()]
    if args.data_dir is not None:
        return sorted(Path(args.data_dir).glob("*.wav"))
    return None


def _preprocess_one(model, path: Path, *, channel: int | None):
    start = time.perf_counter()
    return preprocess_file(path, model.config.feature, channel=channel), time.perf_counter() - start


def _preprocess_chunk(model, paths: list[Path], *, channel: int | None):
    return [_preprocess_one(model, path, channel=channel) for path in paths]


def _run_synthetic(args: argparse.Namespace) -> None:
    model = load_model(args.pretrained_model, device=args.device, cache_dir=args.cache_dir, precision=args.precision)
    cfg = model.config.feature
    synthetic_steps = min(64, cfg.max_segments)
    # Allocate only the sequence that is actually benchmarked. In particular,
    # the TTS checkpoint supports 6000 frames but this synthetic probe uses 64;
    # allocating max_segments inflated host memory by ~94x before cropping.
    x = np.zeros((args.batch_size, synthetic_steps, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
    n_wins = np.full((args.batch_size,), synthetic_steps, dtype=np.int32)

    transfer_start = time.perf_counter()
    x_dev, n_dev = model.device_segments(x, n_wins)
    x_dev.block_until_ready()
    n_dev.block_until_ready()
    input_transfer_seconds = time.perf_counter() - transfer_start

    first_forward_start = time.perf_counter()
    model._forward(model._compute_params, x_dev, n_dev).block_until_ready()
    first_forward_seconds = time.perf_counter() - first_forward_start
    start = time.perf_counter()
    for _ in range(args.steps):
        model._forward(model._compute_params, x_dev, n_dev).block_until_ready()
    warmed_seconds = time.perf_counter() - start
    samples_per_second = args.batch_size * args.steps / warmed_seconds
    result = {
        "mode": "synthetic",
        "checkpoint": str(Path(args.pretrained_model)),
        "device": str(model.device),
        "precision": model.precision,
        "batch_size": args.batch_size,
        "seq_len": int(n_wins.max()),
        "steps": args.steps,
        "input_transfer_seconds": input_transfer_seconds,
        # This includes compilation or a persistent-cache lookup as well as one
        # execution. It is deliberately not labelled pure "compile time".
        "first_forward_seconds": first_forward_seconds,
        "warmed_forward_seconds": warmed_seconds,
        "warmed_forward_latency_seconds": warmed_seconds / args.steps,
        "samples_per_second": samples_per_second,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.min_samples_per_second is not None and samples_per_second < args.min_samples_per_second:
        raise SystemExit(
            {
                "samples_per_second": samples_per_second,
                "min_samples_per_second": args.min_samples_per_second,
            }
        )


def _run_end_to_end(args: argparse.Namespace, paths: list[Path]) -> None:
    if args.preprocess_workers < 1:
        raise ValueError("preprocess_workers must be >= 1")
    if not paths:
        raise ValueError("No wav files found")
    model = load_model(args.pretrained_model, device=args.device, cache_dir=args.cache_dir, precision=args.precision)
    cfg = model.config.feature

    # Optional: benchmark the real predict_batch scheduler (length-aware sort,
    # bucket-rounded padding, prefetch overlap) instead of this script's naive
    # in-order chunking. The two differ in scheduling; the label below makes the
    # difference explicit in the JSON output.
    if args.use_predict_batch:
        total_start = time.perf_counter()
        predict_batch(
            model,
            paths,
            batch_size=args.batch_size,
            channel=args.channel,
            preprocess_workers=args.preprocess_workers,
            length_bucket=args.length_bucket,
            on_error="raise",
            batch_mode=args.batch_mode,
        )
        total_seconds = time.perf_counter() - total_start
        result = {
            "mode": "end_to_end",
            "scheduler": "predict_batch",
            "checkpoint": str(Path(args.pretrained_model)),
            "device": str(model.device),
            "precision": model.precision,
            "batch_size": args.batch_size,
            "batch_mode": args.batch_mode,
            "preprocess_workers": args.preprocess_workers,
            "file_count": len(paths),
            "total_seconds": total_seconds,
            "files_per_second": len(paths) / total_seconds,
            "timing_scope": (
                "predict_batch total: preprocessing, input/output transfers, "
                "first-shape compilation or cache lookup, and inference"
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    preprocess_seconds = 0.0
    preprocess_worker_seconds = 0.0
    input_transfer_seconds = 0.0
    output_transfer_seconds = 0.0
    model_seconds = 0.0
    first_shape_forward_seconds = 0.0
    warmed_model_seconds = 0.0
    first_shape_call_count = 0
    warmed_model_call_count = 0
    seen_shapes: set[tuple[int, int]] = set()
    n_wins_values: list[int] = []
    chunks = [paths[start : start + args.batch_size] for start in range(0, len(paths), args.batch_size)]
    total_start = time.perf_counter()

    def prepare(chunk: list[Path]):
        start = time.perf_counter()
        prepared_with_times = _preprocess_chunk(model, chunk, channel=args.channel)
        prep_times = sum(item[1] for item in prepared_with_times)
        return [item[0] for item in prepared_with_times], time.perf_counter() - start, prep_times

    def run_model(prepared) -> None:
        nonlocal first_shape_call_count, first_shape_forward_seconds, input_transfer_seconds
        nonlocal model_seconds, output_transfer_seconds, warmed_model_call_count, warmed_model_seconds
        n_wins = np.stack([item[1] for item in prepared], axis=0)
        max_steps = int(np.max(n_wins))
        # Pad each sample's real segments [n_wins_i, 1, n_mels, seg_length] up to
        # the chunk max_steps with zeros (masked by n_wins). Previously this used
        # `item[0][:max_steps]`, which silently truncated shorter samples and
        # then raised on ragged shapes under np.stack — mixed/ragged lengths
        # were unbenchmarkable.
        x = np.zeros((len(prepared), max_steps, 1, cfg.n_mels, cfg.seg_length), dtype=np.float32)
        for j, (seg, nw) in enumerate(prepared):
            x[j, : int(nw)] = seg[: int(nw)]
        n_wins_values.extend(int(value) for value in n_wins.reshape(-1))

        transfer_start = time.perf_counter()
        x_dev, n_dev = model.device_segments(x, n_wins)
        x_dev.block_until_ready()
        n_dev.block_until_ready()
        input_transfer_seconds += time.perf_counter() - transfer_start

        shape_key = (len(prepared), max_steps)
        model_start = time.perf_counter()
        output = model._forward(model._compute_params, x_dev, n_dev)
        output.block_until_ready()
        elapsed = time.perf_counter() - model_start
        model_seconds += elapsed
        if shape_key in seen_shapes:
            warmed_model_seconds += elapsed
            warmed_model_call_count += 1
        else:
            seen_shapes.add(shape_key)
            first_shape_forward_seconds += elapsed
            first_shape_call_count += 1
        output_transfer_start = time.perf_counter()
        np.asarray(output)
        output_transfer_seconds += time.perf_counter() - output_transfer_start

    if args.preprocess_workers == 1:
        for chunk in chunks:
            prepared, elapsed, worker_elapsed = prepare(chunk)
            preprocess_seconds += elapsed
            preprocess_worker_seconds += worker_elapsed
            run_model(prepared)
    else:
        with ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
            futures = [executor.submit(_preprocess_one, model, path, channel=args.channel) for path in chunks[0]]
            for idx, _chunk in enumerate(chunks):
                wait_start = time.perf_counter()
                prepared_with_times = [future.result() for future in futures]
                elapsed = time.perf_counter() - wait_start
                prepared = [item[0] for item in prepared_with_times]
                worker_elapsed = sum(item[1] for item in prepared_with_times)
                preprocess_seconds += elapsed
                preprocess_worker_seconds += worker_elapsed
                if idx + 1 < len(chunks):
                    futures = [
                        executor.submit(_preprocess_one, model, path, channel=args.channel) for path in chunks[idx + 1]
                    ]
                run_model(prepared)
    total_seconds = time.perf_counter() - total_start

    n_wins_array = np.asarray(n_wins_values, dtype=np.int32)
    result = {
        "mode": "end_to_end",
        # This script's own scheduler is naive in-order chunking with exact
        # chunk-max padding (no length sort, no bucket rounding). It differs from
        # predict_batch's length-aware scheduler; use --use_predict_batch to
        # benchmark the real one. Labeled explicitly so results are not misread.
        "scheduler": "naive_in_order",
        "checkpoint": str(Path(args.pretrained_model)),
        "device": str(model.device),
        "precision": model.precision,
        "batch_size": args.batch_size,
        "preprocess_workers": args.preprocess_workers,
        "file_count": len(paths),
        "n_wins_min": int(n_wins_array.min()),
        "n_wins_max": int(n_wins_array.max()),
        "n_wins_mean": float(n_wins_array.mean()),
        "preprocessing_seconds": preprocess_seconds,
        "preprocessing_worker_seconds": preprocess_worker_seconds,
        "input_transfer_seconds": input_transfer_seconds,
        "output_transfer_seconds": output_transfer_seconds,
        # A new batch shape causes a JIT compilation or persistent-cache
        # lookup. Repeated shapes are the only defensible warmed measurements.
        "first_shape_forward_seconds": first_shape_forward_seconds,
        "first_shape_call_count": first_shape_call_count,
        "warmed_model_seconds": warmed_model_seconds,
        "warmed_model_call_count": warmed_model_call_count,
        "warmed_model_latency_seconds": warmed_model_seconds / warmed_model_call_count
        if warmed_model_call_count
        else None,
        "model_seconds": model_seconds,
        "total_seconds": total_seconds,
        "files_per_second": len(paths) / total_seconds,
        "model_files_per_second": len(paths) / model_seconds if model_seconds else float("inf"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device")
    parser.add_argument("--cache_dir")
    parser.add_argument("--precision", choices=["float32", "bf16"], default="float32")
    parser.add_argument("--min_samples_per_second", type=float)
    parser.add_argument("--data_dir")
    parser.add_argument("--csv_file")
    parser.add_argument("--csv_deg")
    parser.add_argument("--preprocess_workers", type=int, default=1)
    parser.add_argument("--channel", type=int)
    parser.add_argument(
        "--length_bucket",
        type=int,
        default=None,
        help="predict_batch scheduler bucket grid (only with --use_predict_batch)",
    )
    parser.add_argument(
        "--use_predict_batch",
        action="store_true",
        help="benchmark the real predict_batch scheduler (length-aware, bucket-padded) "
        "instead of this script's naive in-order chunking",
    )
    parser.add_argument(
        "--batch_mode",
        choices=["fixed", "cost_aware"],
        default="fixed",
        help="predict_batch batch_mode (only with --use_predict_batch)",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    paths = _collect_paths(args)
    if paths is None:
        _run_synthetic(args)
    else:
        _run_end_to_end(args, paths)


if __name__ == "__main__":
    main()
