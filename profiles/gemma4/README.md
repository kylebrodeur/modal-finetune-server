# Gemma-4 LoRA SFT Profile

This profile is extracted from a *working* Gemma-4 fine-tune that produced and shipped 3 published adapters. It encodes the hard-won gotchas so you don't rediscover them.

**It is a reference recipe, not executable config.** Nothing in `server/` or `eval/` loads `profile.json` at runtime: `server/train_modal.py` builds the LoRA config inline. Treat this file as the record of settings that shipped, and pass the same values through the entrypoints' own arguments.

## The Gemma-4 PEFT gotcha (non-negotiable)

Gemma-4 wraps some layers in a custom class (`Gemma4ClippableLinear`) that PEFT doesn't recognize as a `Linear` target. If you point LoRA at them, PEFT fails loudly or: worse: trains the wrong subset silently.

**The fix:** regex-scope `target_modules` to *only* the language-model stack, and explicitly exclude the vision / audio / multimodal projector modules:

```
target_modules = r".*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
exclude_modules = [
    ".*vision_tower.*",
    ".*audio_tower.*",
    ".*multi_modal_projector.*",
]
```

Source: regex recommended by a PEFT maintainer in [peft#3129](https://github.com/huggingface/peft/issues/3129). If you're training a **text-only** Gemma, you probably don't need the exclude list: but it does no harm to include it and saves you a crash when someone swaps in a multimodal variant.

## Hyperparameters (battle-tested)

| Knob | Value | Why |
| :--- | :--- | :--- |
| `r` | 4 | Small capacity headroom; larger r causes parroting faster. |
| `lora_alpha` | 8 | 2× r: the conventional scaling. |
| `lora_dropout` | 0.05 | Light regularization. |
| `bias` | none | Don't train bias terms. |
| `task_type` | CAUSAL_LM | Standard. |
| `per_device_train_batch_size` | 2 | Leave headroom on a 24 GB A10G for the 1.5K ctx. |
| `gradient_accumulation_steps` | 4 | Effective batch of 8: stable grads on small SFT sets. |
| `learning_rate` | 2e-4 | The LoRA sweet spot between slow learn and parroting. |
| `max_length` | 1536 | Long enough for most advice-style JSON, short enough to keep speed. |
| `bf16` | true | Native on A10G/H100. |

## Anti-parroting

Small SFT sets (300-800 examples) are the "template memorization" danger zone. Mitigations baked into this profile's pipeline:

1. **Temperature ≠ 0 during dataset generation** (`temp=0.7, top_p=0.95`): the targets themselves vary, so exact memorization won't score well.
2. **Distinct eval split** held out *before* training, not after.
3. **Honest eval gate** (`eval/`): must show `TUNED >= BASE` on JSON validity, plus sample the actual advice by eye. The default validator only checks that the output contains a parseable JSON object; domain correctness is only scored if you pass a custom plugin validator (`--validator dotted.path.to.validate`) that checks your domain's rules. Templates look great on loss curves and parrot on real questions.

## Publishing

This profile only specifies **formats**, not destinations, so it's provider-agnostic:

- **HF Hub**: adapters publish via `trainer.model.push_to_hub(...)`.
- **Serving**: this package builds artifacts and does NOT host inference; serve the merged/GGUF artifact through the inference package (`modal-inference-server`), or host it anywhere HF/GGUF artifacts work. `server/gguf_pipeline_modal.py` is the optional quantized-artifact path (LoRA→merge→GGUF→llama.cpp quantization on Modal, no local toolchain).

## Quick start

```bash
# 1. Prep a dataset in openai-chat-jsonl form (see research/prep_dataset.py)
# 2. Train (one command)
modal run server/train_modal.py --profile profiles/gemma4/profile.json \
    --push-to <your-hf-user>/<your-adapter-name>

# 3. Eval (base vs. tuned)
modal run eval/eval_modal.py --adapter <your-hf-user>/<your-adapter-name>
```

For a working example of the dataset shape, check `examples/gemma4_train_example.py`.