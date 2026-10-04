"""modal-finetune-server's hooks INSTANCE (lifecycle extension seam).

Kept in its own module so every consumer imports the SAME instance without
drifting, and so imports stay stdlib-cheap: `server.train_modal` imports the
`modal` SDK at module level (guarded, but still heavy at deploy-parse on the
image), while this module only imports the vendored, stdlib-only `server.libs.
hooks`. Modules that cannot afford heavy imports (eval/run_eval.py, the gguf
pipeline, tests) wire `from server.hooks_wiring import hooks`.

Layout note: consumers reach this file under TWO spellings - the package path
(`server.hooks_wiring`, repo root on sys.path) and the flat name
(`hooks_wiring`, the conftest/Modal-image layout). The canonicalization below
whichever module object was imported first, so both spellings always bind ONE
Hooks instance (no instance drift).

Closed tag set (new tags = this package's releases):
- train.pre / train.post   : around the Modal LoRA training run
- eval.pre / eval.post     : around the honest base-vs-LoRA eval
- eval.verdict             : when the Well-Tuned verdict is computed
- gguf.pre / gguf.post     : around the Merge -> GGUF export pipeline
"""

from __future__ import annotations

import sys

try:
    from server.libs.hooks import Hooks
except ImportError:  # flat layout (conftest, /root/ image): hook classes are siblings
    from hooks import Hooks  # type: ignore[no-redef]


def _resolve_hooks() -> Hooks:
    """Reuse the first-imported spelling of this module; else build the instance."""
    my_mod = sys.modules[__name__]
    for twin in ("hooks_wiring", "server.hooks_wiring"):
        mod = sys.modules.get(twin)
        if mod is not None and mod is not my_mod and hasattr(mod, "hooks"):
            return mod.hooks  # type: ignore[no-any-return]
    return Hooks(
        ("train.pre", "train.post", "eval.pre", "eval.post", "eval.verdict", "gguf.pre", "gguf.post"),
        name="modal-finetune-server",
    )


hooks = _resolve_hooks()
