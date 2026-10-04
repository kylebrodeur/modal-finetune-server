"""Lifecycle tests for modal-finetune-server's hooks wiring.

The instance lives in server/hooks_wiring.py (stdlib-cheap so every module
can import it without dragging the modal SDK in); the fire points live next
to the real stage boundaries in server/train_modal.py,
server/gguf_pipeline_modal.py, and eval/run_eval.py. The pipeline modules
need the `modal` SDK, so these tests verify the wiring behavior directly:
that the shared instance declares the closed tag set, and that the fire
points' payloads follow the family contract (payload dicts with ok=True at
the .post tags, verdict tokens at eval.verdict).
"""

from __future__ import annotations

import re
from pathlib import Path

import hooks_wiring  # vendored flat module; conftest puts server/ on sys.path
import pytest

EXPECTED_TAGS = (
    "train.pre",
    "train.post",
    "eval.pre",
    "eval.post",
    "eval.verdict",
    "gguf.pre",
    "gguf.post",
)


@pytest.fixture()
def fired():
    """Record (tag, args) per fire; drop registrations + error memory after."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    for tag in hooks_wiring.hooks.tags:
        hooks_wiring.hooks.register(tag, lambda payload, _tag=tag, _calls=calls: _calls.append((_tag, payload)))
    yield calls
    hooks_wiring.hooks.clear()


def test_instance_declares_the_closed_tag_set() -> None:
    assert hooks_wiring.hooks.name == "modal-finetune-server"
    assert hooks_wiring.hooks.tags == EXPECTED_TAGS
    # The docstring documents the closed set, per the family contract.
    assert "train.pre" in hooks_wiring.__doc__ and "gguf.post" in hooks_wiring.__doc__


def test_fire_sources_reference_every_tag() -> None:
    """Every declared tag must have a fire site next to a real stage boundary."""
    repo = Path(hooks_wiring.__file__).resolve().parents[1]
    sources = {
        "train": (repo / "server" / "train_modal.py", ("train.pre", "train.post")),
        "gguf": (repo / "server" / "gguf_pipeline_modal.py", ("gguf.pre", "gguf.post")),
        "eval": (repo / "eval" / "run_eval.py", ("eval.pre", "eval.post", "eval.verdict")),
    }
    for _name, (path, tags) in sources.items():
        text = path.read_text(encoding="utf-8")
        root_str = str(path)
        assert "hooks_wiring" in text, f"{root_str} does not wire the shared instance"
        for _tag in tags:
            pattern = re.compile(r'hooks\.fire\(\s*"' + _tag + '"')
            assert pattern.search(text), f"{_tag} not fired in {root_str}"


def test_hooks_fire_in_order_and_host_path_proceeds(fired) -> None:
    """Ordered fire, then the surrounding host code keeps running."""
    calls = fired

    def boom(_payload: dict) -> None:
        raise RuntimeError("lane exploded")

    hooks_wiring.hooks.register("train.pre", boom)
    hooks_wiring.hooks.register("gguf.pre", boom)

    errors = hooks_wiring.hooks.fire("train.pre", {"profile": "gemma4"})
    assert len(errors) == 1
    assert "RuntimeError: lane exploded" in errors[0]
    assert hooks_wiring.hooks.last_errors("train.pre") == errors
    hooks_wiring.hooks.fire("train.post", {"ok": True})

    # Contained: the host path proceeds (fire never raises) and eval.post is clean.
    assert hooks_wiring.hooks.fire("eval.post", {"ok": True}) == []
    assert hooks_wiring.hooks.last_errors("eval.post") == []
    assert calls[0][0] == "train.pre" and calls[1][0] == "train.post"
    assert [tag for tag, _payload in calls] == ["train.pre", "train.post", "eval.post"]


def test_eval_verdict_fires_with_both_possible_verdicts(fired) -> None:
    calls = fired
    hooks_wiring.hooks.fire("eval.verdict", {"verdict": "well_tuned"})
    hooks_wiring.hooks.fire("eval.verdict", {"verdict": "not_well_tuned"})
    assert [payload for _tag, payload in calls] == [
        {"verdict": "well_tuned"},
        {"verdict": "not_well_tuned"},
    ]


def test_unknown_tag_is_refused_by_the_shared_instance() -> None:
    with pytest.raises(ValueError, match=r"unknown hook tag 'train\.post\.post'"):
        hooks_wiring.hooks.register("train.post.post", lambda payload: None)
