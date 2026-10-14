# Quickstart: modal-finetune-server

A LoRA artifact pipeline on Modal: fine-tune, merge, export GGUF, then run an eval
gate. It builds artifacts only. Serving belongs to the inference server. Current
release: v1.3.0.

Start with the [toolkit quickstart](https://github.com/kylebrodeur/modal-toolkit/blob/main/docs/quickstart.md) if `mtk` is not
installed yet.

## 1. Create a project

```bash
mtk init tune-lab
cd tune-lab
```

## 2. Pull the package

```bash
mtk pull finetune
```

## 3. Write the overlay

`deploys/finetune/deploy.json` names the app `<project>-finetune`:

```json
{ "pkg": "finetune", "app_name": "tune-lab-finetune" }
```

This package runs jobs rather than a long-lived server. Its `deploy.sh` launches
each job from the pulled checkout by path.

## 4. Secrets and jobs

`mtk deploy finetune` checks two secrets, both supplied by you (nothing is
generated): `modal-finetune-secrets` (`HF_TOKEN`, for Hub upload and gated
base-model access) and `HF_TOKEN` (`HF_TOKEN`, the GGUF pipeline fallback).

```bash
mtk deploy finetune
```

Then run the jobs through your overlay:

- `mtk finetune train`: LoRA SFT training
- `mtk finetune eval`: the honesty gate on a trained adapter
- `mtk finetune gguf`: merge and export GGUF for the inference server

## 5. Verify

```bash
mtk doctor --pkg finetune
```

## Next

- Profiles, the eval gate, and publishing: [how-to](./how-to.md)
