"""Honest base-vs-LoRA eval on held-out data, with pluggable domain validators.

The generic gate is **JSON-valid**: the model's output must parse as a valid
JSON object (or whatever shape your domain uses). On top of that, you can plug
in a domain-specific **validator** to check things like safety bounds,
schema-correctness, or business rules.

A fine-tune is "Well-Tuned" only if it earns it:
1. **TUNED >= BASE** on both JSON-valid rate and validator-pass rate.
2. **Sampled advice reads as real judgment**, not a template.

If either fails, report honestly: a thin badge is worse than a plain answer.

Run (on a GPU box / Modal):
    uv run python eval/run_eval.py --base <base-model> --adapter <hf-user>/<adapter>

To plug in a custom validator, write a module that exposes:
    def validate(text: str, row: dict) -> bool

and pass its dotted path to `--validator`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import sys
from pathlib import Path

# Opt-in metrics (MODAL_FINETUNE_METRICS=1): the env is read once at import, so
# with metrics off every emission site below is a no-op. The vendored
# vm_metrics.py lives in the sibling server/ dir; the import is best-effort:
# metrics must never hard-require the vendored copy.
_METRICS_ON = os.environ.get("MODAL_FINETUNE_METRICS", "0") not in ("", "0", "false")


def _load_vm_metrics():
    """Import the vendored server/vm_metrics.py. None if unreachable."""
    server_dir = str(Path(__file__).resolve().parent.parent / "server")
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    try:
        import vm_metrics

        return vm_metrics
    except ImportError:
        return None


_vm_metrics = _load_vm_metrics() if _METRICS_ON else None


def _metric(name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
    """Emit one metrics point when MODAL_FINETUNE_METRICS=1; no-op otherwise."""
    if _vm_metrics is None:
        return
    merged = {"device": os.environ.get("MODAL_FINETUNE_DEVICE_TAG", "modal-finetune-server")}
    if tags:
        merged.update(tags)
    with contextlib.suppress(Exception):
        _vm_metrics.write_metric(name, value, tags=merged)


def _default_validator(text: str, row: dict) -> bool:
    """Generic validator: output must contain a parseable JSON object."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match is not None and json.loads(match.group(0)) is not None


def _load_validator(dotted: str):
    if not dotted:
        return _default_validator
    module_path, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _generate(model, tok, user_text: str, max_new: int = 512) -> str:
    import torch

    msgs = [{"role": "user", "content": user_text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _score(model, tok, rows: list[dict], validate) -> dict:

    valid = validator_ok = 0
    samples: list[dict] = []
    for r in rows:
        user = r["messages"][0]["content"]
        text = _generate(model, tok, user)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            samples.append({"prompt": user[:120], "response": text[:200], "json_valid": False})
            continue
        try:
            json.loads(match.group(0))
        except (ValueError, TypeError):
            samples.append({"prompt": user[:120], "response": text[:200], "json_valid": False})
            continue
        valid += 1
        if validate(text, r):
            validator_ok += 1
        if len(samples) < 5:
            samples.append({"prompt": user[:120], "response": text[:200], "json_valid": True})
    n = len(rows) or 1
    return {
        "n": len(rows),
        "json_valid_pct": round(100 * valid / n, 1),
        "validator_pass_pct": round(100 * validator_ok / n, 1),
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="HF repo id or local path of the base model")
    ap.add_argument("--adapter", default="", help="HF repo / local path of the LoRA adapter (optional)")
    ap.add_argument(
        "--eval-file",
        default="data/finetune/sft.eval.jsonl",
        help="Held-out JSONL of (prompt / messages) rows",
    )
    ap.add_argument("--limit", type=int, default=40, help="Rows to score (cost control)")
    ap.add_argument("--validator", default="", help="Dotted path to a custom `validate(text, row) -> bool`")
    args = ap.parse_args()

    validate = _load_validator(args.validator)

    rows = [json.loads(line) for line in Path(args.eval_file).read_text().splitlines() if line.strip()][
        : args.limit
    ]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto")
    base_score = _score(base, tok, rows, validate)
    print("BASE  ", json.dumps({k: v for k, v in base_score.items() if k != "samples"}, indent=2))

    if args.adapter:
        tuned = PeftModel.from_pretrained(base, args.adapter)
        tuned_score = _score(tuned, tok, rows, validate)
        print("TUNED ", json.dumps({k: v for k, v in tuned_score.items() if k != "samples"}, indent=2))

        well_tuned = (
            tuned_score["json_valid_pct"] >= base_score["json_valid_pct"]
            and tuned_score["validator_pass_pct"] >= base_score["validator_pass_pct"]
        )
        verdict = "WELL-TUNED (earns the gate)" if well_tuned else "NOT WELL-TUNED (does not beat base)"
        print(f"\nVerdict: {verdict}")
        # Line-protocol tags can't carry spaces; emit the compact token form.
        _metric("finetune_eval_done", tags={"verdict": "well_tuned" if well_tuned else "not_well_tuned"})

    print("Sample responses (first 5):")
    for i, sample in enumerate(base_score["samples"], 1):
        print(f"{i}. prompt={sample['prompt']}")
        print(f"   response={sample['response']}")
        print(f"   json_valid={sample['json_valid']}")


if __name__ == "__main__":
    main()
