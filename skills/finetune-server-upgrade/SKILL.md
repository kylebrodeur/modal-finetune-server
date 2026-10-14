---
name: finetune-server-upgrade
description: Upgrade + deploy-boundary discipline for the modal-finetune-server checkout: tag upgrades, artifact-factory posture (serve lives in inference), pipeline-stage hooks. Use when upgrading/redeploying this repo or before modal commands inside this checkout.
license: Apache-2.0
metadata:
  author: kylebrodeur
  family: modal-toolkit
  repo: modal-finetune-server
---

# finetune-server: upgrade + deploy boundary (lane agents)

See AGENTS.md; the working version:

```bash
# 0. provenance preflight (system workspace):
tools/guards/deploy-provenance.sh <overlay-dir>

# 1. upgrade by tag + verify
git fetch --tags && git checkout <tag>
uv run pytest tests -q
```

Finetune-specific cautions:

- This package is an ARTIFACT FACTORY: it trains/adapters/GGUFs and
  publishes; it does NOT serve (removed `serve` in v1.1.0 by design).
  Serving = register the artifact with modal-inference-server.
- Training runs cost real GPU-hours: any training job in a lane must
  be launched BY Kyle (or carry his explicit instruction WITH the
  profile) and uses the overlay's app name.
- Hooks: pipeline stages fire `train.pre/post`, `eval.pre/post` +
  `eval.verdict`, `gguf.pre/post` (instance resolved by
  `server/hooks_wiring.py`). Lane observers (e.g. cost telemetry)
  register here.
- Publishing artifacts lands on HF Hub under the lane's namespace:
  confirm the target repo/namespace before any push.

## Operating with mtk

Source of truth: the [quickstart](../../docs/quickstart.md) and
[how-to](../../docs/how-to.md) in this repo, and the toolkit's `mtk` skill.

```bash
mtk init <project-id>      # from the project directory; checks prerequisites and the account
mtk pull finetune             # clone this repo into the repos root
mtk deploy finetune           # preflight the overlay, confirm secrets, run deploys/finetune/deploy.sh
mtk doctor --pkg finetune     # health and drift
```

- Overlay: `deploys/finetune/deploy.json` must name the app `<project>-finetune`.
- Secrets this package declares: modal-finetune-secrets and HF_TOKEN (you supply). `mtk deploy` asks before creating any.
- Never run `modal deploy` inside this checkout.
