from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .checkpoint import load_model
from .features import preprocess_file
from .model import NisqaJaxModel


def _format_prediction(model: NisqaJaxModel, values: np.ndarray) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(model.config.output_names)}


def predict_file(model: NisqaJaxModel, wav_path: str | Path, *, channel: int | None = None) -> dict[str, float]:
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
) -> pd.DataFrame:
    if preprocess_workers < 1:
        raise ValueError("preprocess_workers must be >= 1")
    rows = []
    paths = [Path(p) for p in wav_paths]

    def prepare(chunk: list[Path]):
        return [preprocess_file(path, model.config.feature, channel=channel) for path in chunk]

    def predict_prepared(chunk: list[Path], prepared) -> None:
        n_wins = np.stack([item[1] for item in prepared], axis=0)
        max_steps = int(np.max(n_wins))
        x = np.stack([item[0][:max_steps] for item in prepared], axis=0)
        out = model.predict_segments(x, n_wins)
        for path, values in zip(chunk, out):
            rows.append({"deg": str(path), **_format_prediction(model, values)})

    chunks = [paths[start : start + batch_size] for start in range(0, len(paths), batch_size)]
    if preprocess_workers == 1 or len(chunks) < 2:
        for chunk in chunks:
            predict_prepared(chunk, prepare(chunk))
        return pd.DataFrame(rows)

    with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
        futures = [
            executor.submit(preprocess_file, path, model.config.feature, channel=channel)
            for path in chunks[0]
        ]
        for idx, chunk in enumerate(chunks):
            prepared = [future.result() for future in futures]
            if idx + 1 < len(chunks):
                futures = [
                    executor.submit(preprocess_file, path, model.config.feature, channel=channel)
                    for path in chunks[idx + 1]
                ]
            predict_prepared(chunk, prepared)
    return pd.DataFrame(rows)


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
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "NISQA_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
