# NISQA-JAX

Standalone [JAX](https://github.com/google/jax) inference port for the three shipped [NISQA](https://github.com/gabrielmittag/NISQA) speech quality assessment checkpoints. No PyTorch dependency at inference time — runs on CPU, GPU, and TPU.

**3–7× faster than PyTorch on GPU** with numerical parity at ~1e-7 (1000× tighter than the 1e-3 acceptance threshold).

## Checkpoints

| Artifact | Architecture | Outputs | Source |
|---|---|---|---|
| `nisqa_mos_only.npz` | adapt-CNN + self-attention + att-pool | `mos` | `nisqa_mos_only.tar` |
| `nisqa.npz` | adapt-CNN + self-attention + att-pool | `mos, noi, dis, col, loud` | `nisqa.tar` (NISQA_DIM) |
| `nisqa_tts.npz` | standard-CNN + BiLSTM + last-step-bi | `naturalness` | `nisqa_tts.tar` |

Pre-converted `.npz` weights ship in `weights/` — zero-config inference after clone.

## Benchmark (JAX-GPU vs PyTorch-GPU, RTX 3070)

| Checkpoint | JAX speedup (median) | Range | Parity (max abs) |
|---|---|---|---|
| mos (self-att) | **3.04×** | 2.23×–4.73× | 2.4e-7 |
| dim (self-att, 5-out) | **3.43×** | 2.94×–7.20× | 4.8e-7 |
| tts (BiLSTM) | **1.96×** | 1.49×–2.39× | 7.2e-7 |

JAX bf16 vs PyTorch fp16 autocast: **5.8×–6.5×** on self-attention. Full benchmark results in [`adversarial_review/`](adversarial_review/).

## Install

```bash
pip install -e .
```

For GPU inference, install the CUDA JAX wheel matching your driver:
```bash
pip install "jax[cuda12_pip]==0.4.30"  # see https://docs.jax.dev/en/latest/installation.html
```

For checkpoint conversion or PyTorch parity tests (optional):
```bash
pip install -e '.[convert]'
```

## Predict

Single file:
```bash
python -m nisqa_jax.predict \
  --mode predict_file \
  --pretrained_model weights/nisqa_mos_only.npz \
  --deg sample.wav \
  --precision float32
```

Batch directory:
```bash
python -m nisqa_jax.predict \
  --mode predict_dir \
  --pretrained_model weights/nisqa.npz \
  --data_dir wavs \
  --bs 8 \
  --preprocess_workers 4 \
  --precision bf16
```

Python API:
```python
from nisqa_jax import load_model, predict_file

model = load_model("weights/nisqa_mos_only.npz", device="gpu", precision="bf16")
scores = predict_file(model, "sample.wav")
# {'mos': 1.324}
```

## Convert Original Checkpoints

Conversion is deterministic and produces an `.npz` artifact plus JSON metadata (source hash, conversion version, tensor shape manifest):

```python
from nisqa_jax.checkpoint import convert_checkpoint

convert_checkpoint("path/to/nisqa_mos_only.tar", cache_dir="weights")
```

## Validate

```bash
pytest -q                    # core JAX tests (standalone, no PyTorch needed)
pytest -q -m parity          # PyTorch reference parity tests (requires torch)
```

## Benchmark

Model-only (JAX only):
```bash
python -m nisqa_jax.bench_compare \
  --pretrained_model weights/nisqa_mos_only.npz \
  --device gpu --precision bf16 --batch_size 8 --seq_len 128 --no_torch
```

JAX vs PyTorch head-to-head (requires CUDA PyTorch):
```bash
python -m nisqa_jax.bench_compare \
  --pretrained_model weights/nisqa_mos_only.npz \
  --device gpu --batch_size 8 --seq_len 128 --steps 100
```

End-to-end with preprocessing:
```bash
python -m nisqa_jax.bench \
  --pretrained_model weights/nisqa_mos_only.npz \
  --device gpu --precision bf16 --batch_size 8 \
  --data_dir wavs --preprocess_workers 4
```

## Adversarial Review

A full read-only adversarial review (136 probes: correctness, edge cases, parity, GPU benchmark) is documented in [`adversarial_review/`](adversarial_review/). Key findings and the improvement roadmap are in [`adversarial_review/ISSUES_AND_ROADMAP.md`](adversarial_review/ISSUES_AND_ROADMAP.md).

## Architecture

```
nisqa_jax/
├── model.py         — JAX forward graph (CNN, self-attention, LSTM, attention pooling)
├── checkpoint.py    — .tar→.npz conversion + artifact loading
├── config.py        — checkpoint args → validated ModelConfig
├── features.py      — librosa mel-spectrogram + segmentation (matches PyTorch exactly)
├── predict.py       — CLI + batch prediction API
├── bench.py         — end-to-end benchmark
└── bench_compare.py — JAX vs PyTorch head-to-head benchmark
```

Key porting decisions:
- **BatchNorm folded into conv** at conversion time (no BN at runtime)
- **Adaptive max pool** uses exact PyTorch bin-edge algorithm
- **LSTM** via `jax.lax.scan` with invalid-timestep masking (`jnp.where`)
- **Self-attention** keeps PyTorch's single `in_proj`, splits q/k/v in forward
- **Pure JAX** — no Flax dependency, functional pytrees + `NisqaJaxModel` wrapper

## License

- **Source code:** MIT (see [LICENSE](LICENSE))
- **Model weights:** CC BY-NC-SA 4.0 (non-commercial, see [weights/LICENSE_model_weights](weights/LICENSE_model_weights))
- **Academic use:** please cite the original NISQA paper (see [CITATION.cff](CITATION.cff))

## Acknowledgements

Original NISQA model by Gabriel Mittag, Babak Naderi, Assmaa Chehadi, and Sebastian Möller (TU Berlin). See [gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA) for the original PyTorch implementation, training code, and datasets.
