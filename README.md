# Modal Finetune Server

A generic, production-tested **LoRA fine-tune → GGUF → serve** pipeline running on Modal GPUs. Profile-driven, provider-agnostic, and honest about whether a fine-tune actually learned your judgment or just parroted it.

---

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Modal](https://img.shields.io/badge/Deploy%20to-Modal-green.svg)](https://modal.com/docs)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/kylebrodeur)

---

## The Four Pillars

### 1. Profiles (`/profiles`) — the recipe
Model-specific, battle-tested configs. Each `profile.json` carries everything to run end-to-end: base model, LoRA rank/alpha/dropout, target/exclude regex modules, SFT hyperparameters, dtype, and publish targets.

**The `gemma4` profile is extracted from three published adapters** (two published on HF Hub, one on Ollama), so the regex scoping and PEFT exclusion rules solve the tricky **`Gemma4ClippableLinear`** incompatibility that breaks naive PEFT setups. See [profiles/gemma4/README.md](profiles/gemma4/README.md).

Want a new arch? Add a config — no code changes required.

### 2. Research (`/research`) — the dataset
Domain-agnostic dataset prep. Accepts any of three input shapes (`messages`, `prompt`/`response`, or `text`), normalizes, dedupes, shuffles, and splits — so a "held-out eval" actually means held-out, not "the last 20 rows".

### 3. Server (`/server`) — the pipeline
The Modal-hosted, end-to-end machinery:
- **`train_modal.py`** — LoRA SFT on Modal A10G, ~1 hr / <$2, pushes adapter straight to HF Hub.
- **`merge_modal.py`** — Merge LoRA adapter into the base model (for inference-ready artifacts).
- **`gguf_pipeline_modal.py`** — Merge → convert to GGUF → quantize on Modal. No local llama.cpp toolchain required.
- **`modal_serve.py`** — OpenAI-compatible `/v1/chat/completions` endpoint with scale-to-zero.

### 4. Eval (`/eval`) — the honesty gate
"The Well-Tuned gate": a fine-tune earns the label only if **TUNED >= BASE** on both JSON validity and a pluggable domain validator.

- **Generic validator** (default): output must contain a parseable JSON object.
- **Plugin validator**: pass `--validator dotted.path.to.validate` for your own schema/rules.
- **Sampled review**: it prints the raw samples so a human can verify "real judgment" by eye.

If it only parrots, you report that — a thin badge is worse than an honest no.

---

## Quick Start: The "Flight Path"

### Step 1: Prep a dataset
```bash
uv run python -m research.prep_dataset \
  --input your-raw-data.jsonl \
  --out-dir data/finetune \
  --eval-fraction 0.2
```

### Step 2: Train with a profile
```bash
modal run server/train_modal.py \
  --profile profiles/gemma4/profile.json \
  --push-to <hf-user>/<your-adapter-name>
```

### Step 3: Eval (base vs. tuned)
```bash
uv run python eval/run_eval.py \
  --base google/gemma-4-E4B-it \
  --adapter <hf-user>/<your-adapter-name>
```

### Step 4: (Optional) Convert + Quantize GGUF
```bash
modal run server/gguf_pipeline_modal.py \
  --adapter <hf-user>/<your-adapter-name> \
  --base google/gemma-4-E4B-it \
  --outtype q4_k_m
```

### Step 5: Deploy the OpenAI-compatible endpoint
```bash
modal deploy server/modal_serve.py
```

---

## Server API Summary

The inference endpoints (`server/modal_serve.py`) expose a standard OpenAI-compatible surface on Modal:

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check and loaded adapter info. |
| `POST` | `/v1/chat/completions` | Chat-completion endpoint (OpenAI-compatible). |

All routes require `Authorization: Bearer <TOKEN>`.

## Configuration

The server is fully configurable via environment variables (prefixed with `MODAL_FINETUNE_`):

- `FINETUNE_BASE`: The HF base model id (e.g., `google/gemma-4-E4B-it`).
- `FINETUNE_ADAPTER`: The HF adapter id (or local path) to serve.
- `MODAL_FINETUNE_GPU`: GPU class (e.g., `A10G`, `L4`, `H100`).
- `MODAL_FINETUNE_SCALEDOWN_WINDOW`: Seconds of inactivity before scale-to-zero.

---

## Part of the Modal Ecosystem

This repo is one of four standalone Modal utilities from the same author. Each is extractable and deployable on its own.

- **[modal-embedding-server](https://github.com/kylebrodeur/modal-embedding-server):** GPU-backed embeddings with a monotonic sync protocol for local-first search.
- **[modal-inference-server](https://github.com/kylebrodeur/modal-inference-server):** OpenAI-compatible LLM inference with hot-set routing and scale-to-zero.
- **[modal-vision-server](https://github.com/kylebrodeur/modal-vision-server):** Specialized vision classification (BioCLIP-2) with adaptive SAM 2.1 segmentation.

## Examples

See [`examples/`](examples/) for a minimal, stdlib-only client that trains a small adapter on toy data.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for ground rules and workflow.

---

Built by [Kyle Brodeur](https://kylebrodeur.com) · Model-selection deep-dive: [Choose the Right Embedding Model for Your Data](https://kylebrodeur.substack.com/p/choose-embedding-model-for-your-data)

---

## License

Apache-2.0 — see [LICENSE](LICENSE).