# How to operate modal-finetune-server

Current release: v1.3.0. Start with the [quickstart](./quickstart.md) if you
have not run a job yet.

## Profiles

A profile records a proven recipe: the base model, LoRA rank and alpha,
target modules, and the SFT hyperparameters. Profiles are documentation of
settings that worked. Nothing loads a profile at runtime, so apply the same
values through the job's own arguments.

## The eval gate

A tune earns the "Well-Tuned" label only when it beats the base model on JSON
validity and on a domain validator. If it does not beat the base, it does not
get the label, whatever the training loss says.

## Publishing

Training pushes the adapter to the Hugging Face Hub. The GGUF step merges and
quantizes, so the result runs on a llama.cpp runtime. For a LoRA on vLLM, serve
the adapter directly through the inference server instead.

## Scope

This repo produces artifacts. Serving, warm and cold management, and cost live
in the inference server.

## Related

- Quickstart: [quickstart.md](./quickstart.md)
