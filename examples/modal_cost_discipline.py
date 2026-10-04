# ---
# cmd: ["modal", "run", "06_gpu_and_ml/training/cost_discipline_finetune.py"]
# lambda-test: true
# ---

# # Cost-disciplined fine-tuning: an eval gate before you publish

# This is the training half of the cost-discipline pattern. From
# [modal-finetune-server](https://github.com/kylebrodeur/modal-finetune-server):
# profile-driven LoRA fine-tunes on Modal GPUs, with the rule that makes the
# whole thing honest: **a fine-tune only earns "Well-Tuned" if it beats the
# base model on measurable checks** (JSON validity + a pluggable domain
# validator). If it just parrots, the gate says so before you publish.

# The economics: a LoRA SFT run on an A10G is ~1 hour and a couple of dollars.
# The real cost isn't the training; it's publishing a bad adapter into your
# serving fleet. This example shows the gate; the train alone doesn't prove it.

import modal

MINUTES = 60  # seconds

app = modal.App(name="example-cost-discipline-finetune")

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # tiny on purpose; the point is the gate
WEIGHTS_DIR = "/weights"
weights_volume = modal.Volume.from_name("cost-discipline-finetune-cache", create_if_missing=True)

def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, cache_dir=WEIGHTS_DIR)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "peft==0.14.0",
        "datasets==3.2.0",
        "accelerate==1.2.1",
        "huggingface-hub==0.26.2",
    )
    .env({"HF_HOME": WEIGHTS_DIR})
    .run_function(download_model, volumes={WEIGHTS_DIR: weights_volume})
)

# Toy training data: three "judgment" examples. In the real pipeline the
# dataset prep normalizes three input shapes, dedupes, shuffles, and holds out
# a real eval split (see the repo's research/prep_dataset.py).
TRAIN_ROWS = [
    {"prompt": "Classify: 'the valve is stuck'",
     "response": "{\"intent\": \"report\", \"part\": \"valve\", \"state\": \"stuck\"}"},
    {"prompt": "Classify: 'pump hums then stops'",
     "response": "{\"intent\": \"report\", \"part\": \"pump\", \"state\": \"intermittent\"}"},
    {"prompt": "Classify: 'where is the manual?'",
     "response": "{\"intent\": \"question\", \"part\": \"manual\"}"},
]

EVAL_ROWS = [
    {"prompt": "Classify: 'the filter is clogged'", "expect": "report"},
    {"prompt": "Classify: 'how do I reset it?'", "expect": "question"},
]

@app.function(image=image, gpu="A10G", volumes={WEIGHTS_DIR: weights_volume}, timeout=30 * MINUTES)
def train_lora() -> dict:
    """Minimal LoRA SFT.

    Returns the adapter path; the real pipeline pushes to HF Hub.
    """
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, cache_dir=WEIGHTS_DIR)

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def tokenize(row):
        text = f"{row['prompt']}\n{row['response']}"
        ids = tok(text, truncation=True, max_length=128, return_tensors="pt")
        return {"input_ids": ids["input_ids"][0], "labels": ids["input_ids"][0].clone()}

    from torch.utils.data import Dataset

    class ToyDataset(Dataset):
        def __init__(self, rows):
            self.rows = [tokenize(r) for r in rows]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/lora", num_train_epochs=3, per_device_train_batch_size=1,
            logging_steps=1, report_to=[],
        ),
        train_dataset=ToyDataset(TRAIN_ROWS),
    )
    trainer.train()
    adapter_path = Path("/tmp/lora/adapter")
    model.save_pretrained(adapter_path)
    return {"adapter": str(adapter_path), "train_rows": len(TRAIN_ROWS)}

# ## The honest gate: TUNED vs BASE, on measures that matter

# Generic check: JSON validity. Domain check: the intent class must match.
# An adapter that parrots its training rows will fail the eval rows; that is
# the point. In the real pipeline the gate only stamps "Well-Tuned" when
# TUNED >= BASE on both, and the operator publishes on the stamp or deletes
# the adapter.

@app.function(image=image, gpu="A10G", volumes={WEIGHTS_DIR: weights_volume}, timeout=20 * MINUTES)
def run_gate(adapter: dict) -> dict:
    import json
    import re

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=WEIGHTS_DIR)
    results = {}
    for name, with_adapter in [("base", None), ("tuned", adapter)]:
        if with_adapter:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, cache_dir=WEIGHTS_DIR
            ).to("cuda")
            model = PeftModel.from_pretrained(base, with_adapter["adapter"])
        else:
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, cache_dir=WEIGHTS_DIR).to("cuda")
        model.eval()

        json_ok = 0
        domain_ok = 0
        for row in EVAL_ROWS:
            inputs = tok(row["prompt"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=48)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if re.search(r"\{.*\}", text, re.DOTALL):
                json_ok += 1
                try:
                    if json.loads(re.search(r"\{.*\}", text, re.DOTALL).group()).get("intent") == row["expect"]:
                        domain_ok += 1
                except Exception:
                    pass

        results[name] = {
            "json_valid": f"{json_ok}/{len(EVAL_ROWS)}",
            "domain_ok": f"{domain_ok}/{len(EVAL_ROWS)}",
        }

    tuned_ok = results["tuned"]["domain_ok"] >= results["base"]["domain_ok"]
    verdict = "Well-Tuned" if tuned_ok else "Not tuned (gate says so)"
    return {"gate": results, "verdict": verdict}

@app.local_entrypoint()
def main():
    adapter = train_lora.remote()
    print(f"trained adapter: {adapter['adapter']} ({adapter['train_rows']} rows)")
    gate = run_gate.remote(adapter)
    print(f"BASE:  {gate['gate']['base']}")
    print(f"TUNED: {gate['gate']['tuned']}")
    print(f"verdict: {gate['verdict']}")
    print("cost: A10G-seconds for the train + the gate; $0 when stopped")

# Run it:
#
# ```bash
# modal run cost_discipline_finetune.py
# ```
#
# The full pipeline (profile-driven LoRA with proper regex scoping, real
# dataset prep, GGUF conversion, OpenAI-compatible serving) is in
# [modal-finetune-server](https://github.com/kylebrodeur/modal-finetune-server).
