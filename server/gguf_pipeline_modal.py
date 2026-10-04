"""Full Merge → GGUF pipeline on Modal.

1. Merge LoRA into base model (GPU)
2. Clone & build llama.cpp (CPU)
3. Convert merged model to GGUF (CPU)
4. Save GGUF to volume for download

No local llama.cpp needed. One command, one GGUF file out.

Run:
  modal run gguf_pipeline_modal.py --adapter <hf-user>/modal-finetune-lora-v2
  modal run gguf_pipeline_modal.py \
    --base google/gemma-4-E4B-it-qat-q4_0-unquantized \
    --adapter <hf-user>/modal-finetune-lora-v3-qat

Download:
  modal volume get modal-finetune gguf/ --force
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

try:
    import modal
except Exception:
    modal = None  # type: ignore

BASE_MODEL = os.environ.get("FINETUNE_BASE", "google/gemma-4-E4B-it")

try:
    ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = Path(__file__).resolve().parent

if modal is not None:
    app = modal.App("modal-finetune-gguf")
    vol = modal.Volume.from_name("modal-finetune", create_if_missing=True)

    # GPU image for merge step
    gpu_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformers>=4.49", "peft>=0.11",
                     "accelerate>=0.34", "huggingface_hub")
    )

    # CPU image for GGUF conversion — llama.cpp pre-built in the image (reused across runs)
    cpu_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "build-essential", "cmake")
        .run_commands(
            "git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /llama.cpp",
            "cd /llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
        )
        .pip_install(
            "huggingface_hub", "torch", "numpy", "tqdm", "pyyaml", "requests",
            "safetensors", "transformers", "sentencepiece", "gguf", "accelerate",
        )
    )

    @app.function(image=gpu_image, gpu="A10G", timeout=1800,
                  volumes={"/out": vol},
                  secrets=[modal.Secret.from_name("modal-finetune-secrets")])
    def merge(base: str, adapter: str) -> str:
        """Merge LoRA into base model. Returns path to merged model on volume."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base: {base}")
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                      device_map="auto")
        print(f"Base loaded on {model.device}")

        print(f"Loading adapter: {adapter}")
        tuned = PeftModel.from_pretrained(model, adapter)
        print("Merging LoRA into base weights...")
        merged = tuned.merge_and_unload()
        print("Merge complete")

        out_dir = "/out/merged"
        merged.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        vol.commit()
        print(f"Merged model saved to {out_dir}")
        return out_dir

    @app.function(image=cpu_image, timeout=3600,
                  volumes={"/out": vol},
                  secrets=[modal.Secret.from_name("modal-finetune-secrets")])
    def convert_to_gguf(merged_path: str, outtype: str = "q4_k_m", name: str = "modal-finetune") -> str:
        """Convert merged HF model to GGUF using llama.cpp.

        Workflow:
        1. Convert HF model to BF16 GGUF (intermediate).
        2. Quantize BF16 GGUF to the target outtype using llama-quantize.
        """
        import subprocess

        os.makedirs("/out/gguf", exist_ok=True)
        bf16_path = f"/out/gguf/{name}-bf16.gguf"
        final_path = f"/out/gguf/{name}.gguf"

        # Step 1: Conversion to high-precision GGUF
        print(f"Step 1/2: Converting {merged_path} → {bf16_path} (bf16)")
        conv_result = subprocess.run(
            ["python3", "/llama.cpp/convert_hf_to_gguf.py",
             merged_path, "--outtype", "bf16", "--outfile", bf16_path],
            capture_output=True, text=True, timeout=1800,
        )
        if conv_result.returncode != 0:
            print(f"CONVERSION STDERR: {conv_result.stderr}")
            raise RuntimeError(f"HF to GGUF conversion failed: {conv_result.stderr[-200:]}")

        # Step 2: Quantization
        # If the user requested bf16, we just move the file.
        # Otherwise, we use llama-quantize for the target type (e.g. q4_k_m).
        if outtype == "bf16":
            print("Target type is bf16, skipping quantization.")
            os.rename(bf16_path, final_path)
        else:
            print(f"Step 2/2: Quantizing {bf16_path} → {final_path} (type: {outtype})")
            quant_result = subprocess.run(
                ["/llama.cpp/build/bin/llama-quantize", bf16_path, final_path, outtype],
                capture_output=True, text=True, timeout=1800,
            )
            if quant_result.returncode != 0:
                print(f"QUANTIZATION STDERR: {quant_result.stderr}")
                raise RuntimeError(f"GGUF quantization failed: {quant_result.stderr[-200:]}")

            # Clean up intermediate BF16 file
            with contextlib.suppress(OSError):
                os.remove(bf16_path)

        vol.commit()

        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"GGUF saved: {final_path} ({size_mb:.0f}MB)")
        return final_path

    @app.function(image=cpu_image, timeout=3600,
                  volumes={"/out": vol},
                  secrets=[modal.Secret.from_name("modal-finetune-secrets"),
                           modal.Secret.from_name("HF_TOKEN")])
    def upload_to_hub(gguf_path: str, repo_id: str, name: str = "modal-finetune") -> str:
        """Upload the GGUF file to a HF Hub model repo. Creates the repo if missing."""
        from huggingface_hub import HfApi, create_repo
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN missing from modal-finetune-secrets")
        token = token.strip()  # secrets often contain trailing newlines
        api = HfApi(token=token)
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
        print(f"Uploading {gguf_path} ({size_mb:.0f}MB) → {repo_id}/{name}.gguf")
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo=f"{name}.gguf",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"upload {name}.gguf ({size_mb:.0f}MB)",
        )
        url = f"https://huggingface.co/{repo_id}/blob/main/{name}.gguf"
        print(f"✅ Uploaded: {url}")
        return url

    @app.local_entrypoint()
    def upload_only(name: str = "", repo: str = "", as_name: str = ""):
        """Upload an existing GGUF (already on the Modal volume) to HF Hub.
        Useful for v1/v2 which were converted before the upload step existed.
        --as-name lets the HF filename differ from the volume filename (e.g. for quant suffix).
        Example: modal run gguf_pipeline_modal.py::upload_only --name modal-finetune-v3-qat
                   --repo <hf-user>/modal-finetune-gguf --as-name modal-finetune-v3-qat-q4_0
        """
        if not name or not repo:
            print("ERROR: --name and --repo required")
            return
        gguf_path = f"/out/gguf/{name}.gguf"
        target = as_name if as_name else name
        print(f"Uploading {gguf_path} → {repo}/{target}.gguf")
        upload_to_hub.remote(gguf_path, repo, target)

    @app.local_entrypoint()
    def main(base: str = BASE_MODEL, adapter: str = "",
             outtype: str = "q4_k_m", name: str = "modal-finetune",
             upload: str = ""):
        if not adapter:
            print("ERROR: --adapter required. Example:")
            print(
                "  modal run gguf_pipeline_modal.py --adapter <hf-user>/modal-finetune-lora-v2 "
                "--name modal-finetune-v2 --upload <hf-user>/modal-finetune-gguf"
            )
            return

        print(f"=== GGUF Pipeline: {adapter} ===")
        print(f"Base: {base} | Outtype: {outtype} | Name: {name}")

        # Step 1: Merge on GPU
        print("\n--- Step 1: Merge LoRA (GPU) ---")
        merged_path = merge.remote(base, adapter)
        print(f"Merged model at: {merged_path}")

        # Step 2: Convert to GGUF on CPU
        print("\n--- Step 2: Convert to GGUF (CPU) ---")
        gguf_path = convert_to_gguf.remote(merged_path, outtype, name)

        # Step 3 (optional): Upload to HF Hub
        if upload:
            print("\n--- Step 3: Upload to HF Hub ---")
            upload_to_hub.remote(gguf_path, upload, name)

        print("\n=== PIPELINE COMPLETE ===")
        print(f"GGUF file: {gguf_path}")
        print("\nDownload:")
        print("  modal volume get modal-finetune gguf/ --force")
        print("\nOllama import:")
        print("  cat > Modelfile << 'EOF'")
        print(f"  FROM ./{name}.gguf")
        print('  TEMPLATE """{{ if .System }}<start_of_turn>system')
        print('  {{ .System }}<end_of_turn>')
        print('  {{ end }}<start_of_turn>user')
        print('  {{ .Prompt }}<end_of_turn>')
        print('  <start_of_turn>model')
        print('  """')
        print('  PARAMETER stop "<start_of_turn>user"')
        print('  PARAMETER stop "<end_of_turn>"')
        print('  EOF')
        print(f"  ollama create {name} -f Modelfile")
        print(f"  ollama run {name}")


if __name__ == "__main__":
    print("Full Merge → GGUF pipeline on Modal.")
    print("  modal run gguf_pipeline_modal.py --adapter <hf-user>/modal-finetune-lora-v2")
