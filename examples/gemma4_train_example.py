"""Example: train a tiny LoRA on toy data with the gemma4 profile, then serve it.

This is the "quickstart" example: shows the full flow at minimum scale.
Run it end-to-end with:
    modal run examples/gemma4_train_example.py

Change:
- The dataset (examples/data/toy.jsonl) to your own prompt-response pairs.
- The base model / profile path if you have a different arch.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def build_toy_dataset() -> Path:
    """Write a tiny (8 row) prompt-response dataset in the openai-chat shape."""
    pairs = [
        ("What is 2 + 2?", "2 + 2 = 4"),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Explain gravity in one sentence.", "Gravity is the force that pulls masses toward each other."),
        ("Translate 'hello' to Spanish.", "'hello' in Spanish is 'hola.'"),
        ("What is 10 * 5?", "10 * 5 = 50"),
        ("Who wrote 'Romeo and Juliet'?", "William Shakespeare wrote 'Romeo and Juliet.'"),
        ("What is the boiling point of water?", "Water boils at 100 degrees Celsius (at sea level)."),
        ("Name a primary color.", "Red is a primary color."),
    ]
    rows = [
        {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
        for q, a in pairs
    ]
    out = HERE / "data" / "toy.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out


def main() -> None:
    dataset = build_toy_dataset()
    print(f"Toy dataset at: {dataset}")
    print()
    print("Next: build the SFT split and train with:")
    print("  uv run python -m research.prep_dataset --input examples/data/toy.jsonl --out-dir data/finetune")
    print()
    print("Then run a LoRA tune:")
    print("  modal run server/train_modal.py --profile profiles/gemma4/profile.json --epochs 1 --push-to <hf-user>/<adapter>")
    print()
    print("Then serve:")
    print("  modal deploy server/modal_serve.py")


if __name__ == "__main__":
    main()