"""LoRA fine-tune a small Gemma on the node's distilled judgment, on Modal.

This is the "Well-Tuned" frontier, realized: the live node uses retrieval (visible
memory); this bakes a copy of that judgment into the weights. It does NOT replace the
demo path. Run it in the background (e.g. while recording), then evaluate honestly
(eval.py) and only claim Well-Tuned if the eval earns it.

Import-guarded: `modal` is not a base dep and nothing in app.py imports this. Run from
the project root after `modal token set` and after running research/prep_dataset.py:

    uv run python -m research.prep_dataset          # writes data/finetune/sft.*.jsonl
    modal run train_modal.py --push-to <hf-user>/modal-finetune-lora

Pushes the LoRA adapter (small, ~10-50MB) to the Hub. Base model stays ungated-friendly;
set BASE_MODEL to a Gemma you can load (accept its license + HF_TOKEN secret).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import modal
except Exception:  # pragma: no cover
    modal = None  # type: ignore

BASE_MODEL = os.environ.get("FINETUNE_BASE", "google/gemma-4-E4B-it")  # matches live gemma4:e4b
# ROOT is only used locally for add_local_file paths; on Modal the file is at /root/
try:
    ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = Path(__file__).resolve().parent  # on Modal, file is at /root/train_modal.py

if modal is not None:
    app = modal.App("modal-finetune")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformers>=4.49", "peft>=0.11", "trl>=0.9",
                     "datasets>=2.19", "accelerate>=0.34", "bitsandbytes>=0.43", "huggingface_hub")
        .add_local_file(str(ROOT / "data" / "finetune" / "sft.train.jsonl"),
                        "/root/sft.train.jsonl")
        .add_local_file(str(ROOT / "data" / "finetune" / "sft.eval.jsonl"),
                        "/root/sft.eval.jsonl")
    )
    vol = modal.Volume.from_name("modal-finetune", create_if_missing=True)

    @app.function(image=image, gpu="A10G", timeout=3600,
                  volumes={"/out": vol},
                  secrets=[modal.Secret.from_name("modal-finetune-secrets")])
    def train(push_to: str = "", base: str = BASE_MODEL, epochs: int = 1) -> dict:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        ds = load_dataset("json", data_files={"train": "/root/sft.train.jsonl",
                                              "eval": "/root/sft.eval.jsonl"})

        def fmt(row):
            # Gemma chat template over the [user, assistant] messages we stored.
            return {"text": tok.apply_chat_template(row["messages"], tokenize=False)}

        ds = ds.map(fmt)
        # Gemma 4 fix: target only language_model layers (standard nn.Linear).
        # Vision/audio towers use Gemma4ClippableLinear which PEFT doesn't recognize.
        # Regex from PEFT maintainer: https://github.com/huggingface/peft/issues/3129
        peft_cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
            exclude_modules=[r".*vision_tower.*", r".*audio_tower.*", r".*multi_modal_projector.*"],
        )
        cfg = SFTConfig(output_dir="/out/adapter", num_train_epochs=epochs,
                        per_device_train_batch_size=2, gradient_accumulation_steps=4,
                        learning_rate=2e-4, logging_steps=10, bf16=True,
                        dataset_text_field="text", max_length=1536,
                        report_to=[])
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                             eval_dataset=ds["eval"], peft_config=peft_cfg,
                             processing_class=tok)
        trainer.train()
        trainer.save_model("/out/adapter")
        vol.commit()
        pushed = None
        if push_to:
            trainer.model.push_to_hub(push_to)
            tok.push_to_hub(push_to)
            pushed = push_to
        return {"status": "ok", "base": base, "adapter": "/out/adapter", "pushed": pushed}

    @app.local_entrypoint()
    def main(push_to: str = "", base: str = BASE_MODEL, epochs: int = 1):
        print(train.remote(push_to=push_to, base=base, epochs=epochs))


if __name__ == "__main__":
    print("Modal LoRA fine-tune. Install modal + `modal token set`, prep the dataset, then:")
    print("  modal run train_modal.py --push-to <hf-user>/modal-finetune-lora")
