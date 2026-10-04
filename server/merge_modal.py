"""Merge LoRA adapter into base model on Modal GPU.

Produces a full merged model ready for GGUF conversion via llama.cpp.
This is Step 5a of the fine-tune pipeline (PIPELINE.md).

Run:  modal run merge_modal.py --adapter <user>/modal-finetune-lora-v2
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import modal
except Exception:
    modal = None  # type: ignore

BASE_MODEL = os.environ.get("FINETUNE_BASE", "google/gemma-4-E4B-it")

try:
    ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = Path(__file__).resolve().parent

if modal is not None:
    app = modal.App("modal-finetune-merge")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformers>=4.49", "peft>=0.11",
                     "accelerate>=0.34", "huggingface_hub")
    )
    vol = modal.Volume.from_name("modal-finetune", create_if_missing=True)

    @app.function(image=image, gpu="A10G", timeout=1800,
                  volumes={"/out": vol},
                  secrets=[modal.Secret.from_name("modal-finetune-secrets")])
    def merge(base: str = BASE_MODEL, adapter: str = "") -> dict:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not adapter:
            return {"error": "No adapter specified. Use --adapter <user>/<repo>"}

        print(f"Loading base: {base}")
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                      device_map="auto")
        print(f"Base loaded on {model.device}")

        print(f"Loading adapter: {adapter}")
        tuned = PeftModel.from_pretrained(model, adapter)
        print("Adapter loaded")

        print("Merging LoRA into base weights...")
        merged = tuned.merge_and_unload()
        print("Merge complete")

        out_dir = "/out/merged"
        merged.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        vol.commit()

        print(f"Merged model saved to {out_dir}")
        print("Ready for GGUF conversion: python llama.cpp/convert_hf_to_gguf.py <downloaded_dir> --outtype q4_k_m")
        return {"status": "ok", "path": out_dir, "base": base, "adapter": adapter}

    @app.local_entrypoint()
    def main(base: str = BASE_MODEL, adapter: str = ""):
        result = merge.remote(base=base, adapter=adapter)
        print(result)


if __name__ == "__main__":
    print("Modal LoRA merge. Usage:")
    print("  modal run merge_modal.py --adapter <hf-user>/modal-finetune-lora-v2")
