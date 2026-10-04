# Contributing

Thanks for your interest in contributing. This project is extracted from production work and kept intentionally lean, so contributions should follow the same spirit.

## Ground Rules

- **Keep it boring.** Prefer straightforward code over clever abstractions. This is a utility, not a framework.
- **No new heavyweight deps** unless absolutely required for the training/eval stack. The local test env must stay fast to `uv sync`.
- **Profiles, not code.** New base-model families get a `profiles/<name>/profile.json` + `README.md` recording the settings you actually trained with (as a reference recipe), and the training/eval code changes needed to run them. Profiles are not loaded at runtime; keep endpoint logic straightforward.
- **Honest eval, always.** A fine-tune earns "Well-Tuned" only if it scores `TUNED >= BASE` on json-valid and a domain validator, plus passes human sample review. Do not weaken the gate.
- **Env prefixes** follow `MODAL_FINETUNE_*` (and `FINETUNE_*` for legacy HuggingFace-style knobs). Never reintroduce project-specific branding.
- **No AI slop.** Comments and docs should describe *why* the code exists, not restate what it does.

## Workflow

1. Fork and create a feature branch.
2. Use `uv sync --group dev` to set up a local dev environment.
3. Ensure `uv run pytest tests -q` passes for any code change.
4. Run `uv run ruff check .` before submitting.
5. Submit a pull request against `main` with a clear description of what and why.

The project is licensed under Apache 2.0. By contributing you agree that your contributions will be licensed under the same terms.