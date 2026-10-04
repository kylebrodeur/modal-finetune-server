"""Build a supervised fine-tune dataset from your own (prompt, response) pairs.

Reads a JSONL file of raw examples in **one of three accepted shapes** and
normalizes into the `openai-chat-jsonl` format that `train_modal.py` consumes:

    {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
    {"prompt": "...", "response": "..."}
    {"text": "..."}

Also handles:
- **Splitting** into train/eval sets with a held-out fraction so you never
  accidentally report training-set metrics as eval.
- **Shuffling** to prevent ordering bias.
- **Deduping** identical (prompt, response) pairs, so templates don't
  over-weight common patterns.

Run:
    uv run python -m research.prep_dataset --input my-data.jsonl --out-dir data/finetune

The pipeline is domain-agnostic: bring your own pairs from anywhere
(live-model distillation, human transcripts, synthetic generation).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _normalize(row: dict) -> dict | None:
    """Coerce the three accepted input shapes into openai-chat-jsonl."""
    if "messages" in row and isinstance(row["messages"], list):
        return {"messages": row["messages"]}
    if "prompt" in row and "response" in row:
        return {
            "messages": [
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["response"])},
            ]
        }
    if "text" in row:
        return {"text": row["text"]}
    return None


def prep(
    input_path: Path,
    out_dir: Path,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    """Normalize → dedupe → shuffle → split a raw JSONL file into sft.{train,eval}.jsonl."""
    seen: set[str] = set()
    rows: list[dict] = []
    with input_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            normalized = _normalize(json.loads(line))
            if normalized is None:
                continue
            key = json.dumps(normalized, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalized)

    random.Random(seed).shuffle(rows)
    eval_n = max(1, int(len(rows) * eval_fraction))
    eval_rows, train_rows = rows[:eval_n], rows[eval_n:]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sft.train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )
    (out_dir / "sft.eval.jsonl").write_text(
        "\n".join(json.dumps(r) for r in eval_rows) + "\n"
    )
    return len(train_rows), len(eval_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Raw JSONL of (prompt,response)/messages/text pairs")
    ap.add_argument("--out-dir", default="data/finetune", help="Output directory for sft.{train,eval}.jsonl")
    ap.add_argument("--eval-fraction", type=float, default=0.2, help="Held-out fraction for eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, ev = prep(Path(args.input), Path(args.out_dir), args.eval_fraction, args.seed)
    print(f"train={train} eval={ev} → {args.out_dir}/sft.train.jsonl + sft.eval.jsonl")


if __name__ == "__main__":
    main()
