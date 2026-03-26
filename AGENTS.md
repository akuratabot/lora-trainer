# AGENTS.md

Instructions for AI coding agents operating in this repository.

## Project Overview

This repo provides a K3s-deployable training pipeline for character-specific LoRA weights on **Qwen-Image-2512** (20B MMDiT). It has two distinct components:

1. **`scripts/caption.py`** -- Standalone local script. Calls an OpenAI-compatible VLM API to auto-caption training images. No dependencies beyond Python 3.8+ stdlib.
2. **K3s training pipeline** -- Dockerfile (NGC PyTorch 25.11 + SimpleTuner), K3s Job manifest, and SimpleTuner config files for running LoRA training on a GPU node.

## Repository Structure

```
scripts/caption.py           # Local captioning script (NOT in the container image)
config/config.json           # SimpleTuner training hyperparameters
config/multidatabackend.json # SimpleTuner dataset backend config
config/user_prompt_library.json  # Validation prompts
config/config.env            # SimpleTuner accelerate/env settings
Dockerfile                   # Training-only container (NGC PyTorch 25.11 + SimpleTuner)
k8s/pvc.yaml                # PersistentVolumeClaim (130Gi, local-path)
k8s/train-job.yaml          # K3s Job for SimpleTuner training
GUIDE.md                     # Step-by-step usage guide
```

## Target Hardware

- **GPU**: NVIDIA GB10 Blackwell (compute capability 12.1, sm_100)
- **Architecture**: ARM64 / aarch64 (NVIDIA Grace CPU)
- **Memory**: 128GB unified (shared CPU/GPU)
- **Driver**: 580.x (CUDA 13.0.x compatible, NOT 13.1+)
- **K3s**: Pre-configured with NVIDIA device plugin

The container base image MUST be `nvcr.io/nvidia/pytorch:25.11-py3` or equivalent that supports ARM64 SBSA + Blackwell + CUDA 13.0.x. Do NOT upgrade to 25.12+ or 26.x without confirming driver compatibility (requires >= 590 for CUDA 13.1).

## Key Constraints

### caption.py
- **Zero external dependencies.** Uses only Python stdlib (urllib, json, base64, pathlib, argparse). Do not add pip dependencies.
- Defaults to local paths (`./dataset/images`), not container paths.
- Writes `.txt` caption files alongside images in the same directory (SimpleTuner `textfile` caption strategy requires this).

### Dockerfile
- Installs SimpleTuner via `pip install 'simpletuner[cuda13]'`. Do not switch to `[cuda]` -- the `[cuda13]` extra is required for Blackwell.
- `TORCH_CUDA_ARCH_LIST="10.0"` must stay set for sm_100 kernel compilation.
- Runs as non-root (UID 1000) for restricted namespace compatibility. All runtime writes go to the PVC at `/data`.
- Does NOT include `scripts/caption.py` -- captioning is local-only.

### K3s Manifests
- `train-job.yaml` runs under the `restricted` Pod Security Standard (non-root, drop all capabilities, seccomp RuntimeDefault).
- The initContainer copies default configs from `/app/config/` in the image to `/data/config/` on the PVC, but only if they don't already exist. This lets users override configs on the PVC.
- Training requires the full GPU and 128Gi memory. A co-located vLLM pod (40GB) must be scaled down before training starts.
- `nvidia.com/gpu: 1` is the resource request. Do not change this.

### SimpleTuner Config
- `model_family` must be `qwen_image`, `model_flavour` must be `v1.0`.
- `mixed_precision` and `base_model_precision` must be `bf16`. fp16 does not work with Qwen-Image.
- `noise_scheduler` must be `flowmatch`. Qwen-Image uses flow matching, not DDPM.
- `max_grad_norm: 0.01` is Qwen-Image-specific. Default values (1.0) cause instability.
- All paths in config.json and multidatabackend.json reference `/data/...` (the PVC mount point inside the container).

## Making Changes

### Modifying training parameters
Edit `config/config.json`. The key parameters to tune are `lora_rank`, `learning_rate`, `train_batch_size`, `max_train_steps`, and `flow_schedule_shift`. See GUIDE.md "Tuning Parameters" for guidance.

### Modifying the dataset layout
Edit `config/multidatabackend.json`. This is a JSON array of dataset backends. The subject images use `caption_strategy: "textfile"` (reads `.txt` files co-located with images). Regularization images use `caption_strategy: "instanceprompt"`.

### Modifying validation prompts
Edit `config/user_prompt_library.json`. Keys are short names, values are full prompts. All should include the trigger word (`ohwx`).

### Adding new K3s resources
Place YAML files in `k8s/`. All pods must comply with the `restricted` Pod Security Standard: `runAsNonRoot`, `runAsUser: 1000`, `allowPrivilegeEscalation: false`, `capabilities.drop: ["ALL"]`, `seccompProfile: RuntimeDefault`.

## Testing

- **caption.py**: Can be tested locally against any OpenAI-compatible vision API endpoint. Create a temp directory with a test image and run `python scripts/caption.py --api-url <url> --image-dir <dir>`.
- **JSON configs**: Validate with `python -c "import json; json.load(open('config/config.json'))"` (and similarly for the other JSON files).
- **Dockerfile**: Build with `docker build .` (requires ARM64 host or buildx for cross-compilation). The NGC base image is ~15GB.
- **K8s manifests**: Dry-run with `kubectl apply --dry-run=client -f k8s/train-job.yaml`.

## Do NOT

- Deploy or run training from this dev environment. Build and test only.
- Change the base image without confirming ARM64 + Blackwell + CUDA 13.0.x compatibility.
- Add dependencies to `scripts/caption.py`.
- Remove the non-root user from the Dockerfile or the security contexts from `train-job.yaml`.
- Hardcode HuggingFace tokens or secrets into any file. Use K8s Secrets or env vars.
