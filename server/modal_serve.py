"""Modal Inference API for Modal Finetune LoRA models.

Hosts the fine-tuned LoRA adapter behind an OpenAI-compatible /v1/chat/completions
endpoint on Modal GPU. Auto-scales to zero after inactivity.

Deploy:  modal deploy modal_serve.py
Test:    curl -X POST https://<user>--modal-finetune-inference.modal.run/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{"messages":[{"role":"user","content":"PLA overhang at 22C, 45% humidity"}],"max_tokens":512}'

Budget: Separate $100 serving budget (distinct from training budget).
Cost:   A10G ~$5.04/hr active. With scale-to-zero, ~$0.50-2.00/day typical.
"""

from __future__ import annotations

import os

try:
    import modal
except Exception:
    modal = None  # type: ignore

BASE_MODEL = os.environ.get("FINETUNE_BASE", "google/gemma-4-E4B-it")
ADAPTER = os.environ.get("FINETUNE_ADAPTER", "<hf-user>/modal-finetune-lora-v2")

if modal is not None:
    app = modal.App("modal-finetune-inference")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformers>=4.49", "peft>=0.11",
                     "accelerate>=0.34", "fastapi", "uvicorn")
    )

    @app.function(
        image=image,
        gpu="A10G",
        timeout=300,
        secrets=[modal.Secret.from_name("modal-finetune-secrets")],
        scaledown_window=300,  # Scale to zero after 5 min idle
    )
    @modal.concurrent(max_inputs=10)
    @modal.asgi_app()
    def serve():
        import torch
        from fastapi import FastAPI
        from peft import PeftModel
        from pydantic import BaseModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        web = FastAPI(title="Modal Finetune Inference API")

        # --- Model loading (once at container start) ---
        _tok = None
        _model = None

        def _ensure_loaded():
            nonlocal _tok, _model
            if _model is not None:
                return
            print(f"Loading base model: {BASE_MODEL}")
            _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
            )
            print(f"Loading adapter: {ADAPTER}")
            _model = PeftModel.from_pretrained(base, ADAPTER)
            print(f"Model ready on {_model.device}")

        # --- API types ---
        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatRequest(BaseModel):
            messages: list[ChatMessage]
            max_tokens: int = 512
            temperature: float = 0.7

        class ChatResponse(BaseModel):
            choices: list[dict]

        # --- Health check ---
        @web.get("/health")
        async def health():
            return {"status": "ok", "base": BASE_MODEL, "adapter": ADAPTER}

        # --- Chat completions ---
        @web.post("/v1/chat/completions")
        async def chat(req: ChatRequest):
            _ensure_loaded()

            msgs = [{"role": m.role, "content": m.content} for m in req.messages]
            prompt = _tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = _tok(prompt, return_tensors="pt").to(_model.device)

            with torch.no_grad():
                out = _model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=req.temperature > 0,
                    temperature=max(req.temperature, 1e-4),
                )

            text = _tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}

        return web


if __name__ == "__main__":
    print("Modal Inference API for Modal Finetune LoRA models.")
    print("Deploy:  modal deploy modal_serve.py")
    print("Test:    curl -X POST <url>/v1/chat/completions ...")
