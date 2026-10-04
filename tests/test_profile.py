"""Smoke tests for the modal-finetune profile and dataset prep modules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.prep_dataset import _normalize, prep

ROOT = Path(__file__).resolve().parent.parent

# The gemma4 profile must be valid JSON and preserve the verified LoRA/SFT values.
def test_gemma4_profile_loads():
    profile_path = ROOT / "profiles" / "gemma4" / "profile.json"
    profile = json.loads(profile_path.read_text())
    assert profile["base_model"] == "google/gemma-4-E4B-it"
    assert profile["lora"]["r"] == 4
    assert profile["lora"]["lora_alpha"] == 8
    assert profile["lora"]["lora_dropout"] == 0.05
    assert profile["lora"]["task_type"] == "CAUSAL_LM"
    assert ".*vision_tower.*" in profile["lora"]["exclude_modules"]
    assert profile["sft"]["learning_rate"] == pytest.approx(2e-4)
    assert profile["sft"]["max_length"] == 1536
    assert profile["sft"]["per_device_train_batch_size"] == 2


def test_profile_target_modules_regex_is_language_scoped():
    """The Gemma-4 PEFT gotcha: target_modules MUST NOT touch vision/audio towers."""
    profile = json.loads((ROOT / "profiles" / "gemma4" / "profile.json").read_text())
    regex = profile["lora"]["target_modules"]
    assert "language_model" in regex  # only language layers
    assert all(
        proj in regex for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


def test_normalize_chat_shape():
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    assert _normalize(row) == row


def test_normalize_prompt_response_shape():
    row = {"prompt": "q", "response": "a"}
    assert _normalize(row) == {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }


def test_normalize_text_shape():
    row = {"text": "raw"}
    assert _normalize(row) == {"text": "raw"}


def test_normalize_rejects_unknown_shape():
    row = {"wrong": "shape"}
    assert _normalize(row) is None


def test_prep_splits_and_dedupes(tmp_path):
    src = tmp_path / "in.jsonl"
    rows = []
    for i in range(10):
        rows.append({"prompt": f"q{i}", "response": f"a{i}"})
    rows.append({"prompt": "q0", "response": "a0"})  # duplicate
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    train, ev = prep(src, tmp_path, eval_fraction=0.2, seed=42)
    assert train + ev == 10  # dedupe drops the duplicate
    assert ev >= 1

    train_lines = (tmp_path / "sft.train.jsonl").read_text().splitlines()
    eval_lines = (tmp_path / "sft.eval.jsonl").read_text().splitlines()
    assert len(train_lines) == train
    assert len(eval_lines) == ev
    # All rows must be in normalized openai-chat shape
    for line in train_lines + eval_lines:
        assert "messages" in json.loads(line)
