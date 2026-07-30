# Benchmarking

Performance claims are accepted only when they are reproducible, versioned, and
separate from correctness evidence.

## Tools

JAX-only model-forward measurement:

```bash
W="$(python -c 'from nisqa_jax.weights import WEIGHTS_DIR; print(WEIGHTS_DIR / "nisqa_mos_only.npz")')"
python -m nisqa_jax.bench_compare \
  --pretrained_model "$W" \
  --device gpu --precision float32 \
  --batch_size 8 --seq_len 128 --no_torch
```

Hash-bound JAX/PyTorch comparison:

```bash
python -m nisqa_jax.bench_compare \
  --pretrained_model "$W" \
  --torch_checkpoint /path/to/NISQA/weights/nisqa_mos_only.tar \
  --torch_source_root /path/to/NISQA \
  --device gpu --precision float32 \
  --batch_size 8 --seq_len 128 --steps 100
```

End-to-end preprocessing and inference:

```bash
python -m nisqa_jax.bench \
  --pretrained_model "$W" \
  --device gpu --precision float32 \
  --batch_size 8 --data_dir /path/to/wavs
```

## Required evidence

Every report must record:

- repository commit;
- JAX, jaxlib, PyTorch, CUDA runtime, driver, and GPU;
- model, precision, batch sizes, sequence lengths, and input distribution;
- warmup and timed iteration counts;
- whether compilation and transfers are included;
- explicit synchronization and aggregation method;
- raw machine-readable output.

Compilation, host-to-device transfer, warmed model-forward, and preprocessing
must be reported separately. A smoke test proves execution, not performance.
Do not combine measurements from different revisions or PyTorch baselines.

## Current claim policy

The current JAX 0.6.2 implementation has correctness and CUDA smoke evidence,
but no committed full post-optimization comparison grid. The project therefore
makes no current universal speedup claim.

Historical JAX 0.4.30 evidence is preserved under
[historical/](historical/README.md) for provenance only.
