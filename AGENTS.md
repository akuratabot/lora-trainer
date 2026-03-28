# AGENTS.md

Instructions for AI coding agents operating in this repository.

## Project Overview

This repo provides a K3s-deployable training pipeline for character-specific LoRA weights on **Qwen-Image-2512** (20B MMDiT). It has two distinct components:

1. **`scripts/caption.py`** -- Standalone local script. Calls an OpenAI-compatible VLM API to auto-caption training images. No dependencies beyond Python 3.8+ stdlib.
2. **K3s training pipeline** -- Dockerfile (NGC PyTorch 25.11 + SimpleTuner), K3s Job manifest, and SimpleTuner config files for running LoRA training on a GPU node.

## Repository Structure

```
scripts/caption.py               # Local captioning script (NOT in the container image)
Dockerfile                       # Training-only container (NGC PyTorch 25.11 + SimpleTuner)
k8s/pvc.yaml                     # PersistentVolumeClaim (150Gi, longhorn-ssd)
k8s/job/configmap.yaml           # ConfigMap: all SimpleTuner config files (edit here to change training params)
k8s/job/train-job.yaml           # K3s Job for SimpleTuner training
Makefile                         # Build and push the container image
GUIDE.md                         # Step-by-step usage guide
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
- **Single external dependency:** `openai` (install with `pip install openai`). All other imports are stdlib.
- Defaults to local paths (`./dataset/images`), not container paths.
- Writes `.txt` caption files alongside images in the same directory (SimpleTuner `textfile` caption strategy requires this).

### Dockerfile
- Installs SimpleTuner via `pip install 'simpletuner[cuda13]'`. Do not switch to `[cuda]` -- the `[cuda13]` extra is required for Blackwell.
- `TORCH_CUDA_ARCH_LIST="10.0"` must stay set for sm_100 kernel compilation.
- Runs as non-root (UID 1000) for restricted namespace compatibility. All runtime writes go to the PVC at `/data`.
- Does NOT include config files or `scripts/caption.py` -- configs are in the ConfigMap, captioning is local-only.

### K3s Manifests
- `k8s/job/configmap.yaml` contains all SimpleTuner config files as ConfigMap data. Edit this file to change training parameters -- no image rebuild needed.
- `k8s/job/train-job.yaml` mounts the ConfigMap files into `/data/config/` via `subPath` volume mounts. An initContainer downloads the training dataset and model weights.
- `k8s/job/train-job.yaml` runs under the `restricted` Pod Security Standard (non-root, drop all capabilities, seccomp RuntimeDefault).
- Training requires the full GPU and 128Gi memory. A co-located vLLM pod (40GB) must be scaled down before training starts.
- `nvidia.com/gpu: 1` is the resource request. Do not change this.

### SimpleTuner Config
- `model_family` must be `qwen_image`, `model_flavour` must be `v1.0`.
- `mixed_precision` must be `bf16`. fp16 does not work with Qwen-Image.
- `base_model_precision` must be `no_change` (meaning: load the model as-is, no quantization). Do NOT use `bf16` here -- that value is not accepted by SimpleTuner; `no_change` is the correct way to express "full precision, no quantization".
- `gradient_checkpointing` must be `true` -- the 20B model at batch_size>1 exceeds 128GB unified memory without it. The earlier SimpleTuner bug that forced this off (a "multiple values for encoder_hidden_states_mask" TypeError) was fixed in upstream commit a47da0d (2026-03-26); we install from git main so the fix is included.
- `train_batch_size` should be 2 -- batch_size=4 with no gradient checkpointing peaks above 128GB and OOMs. With gradient_checkpointing=true and batch_size=2 the peak stays comfortably within the 128GB unified memory budget.
- `max_grad_norm: 0.01` is Qwen-Image-specific. Default values (1.0) cause instability.
- All paths in config.json and multidatabackend.json reference `/data/...` (the PVC mount point inside the container).

## Making Changes

### Modifying training parameters
Edit the `config.json` key in `k8s/job/configmap.yaml`, then `kubectl apply -f k8s/job/configmap.yaml`. No image rebuild needed.

### Modifying the dataset layout
Edit the `multidatabackend.json` key in `k8s/job/configmap.yaml`. The subject images use `caption_strategy: "textfile"` (reads `.txt` files co-located with images).

### Modifying validation prompts
Edit the `user_prompt_library.json` key in `k8s/job/configmap.yaml`. Keys are short names, values are full prompts. All should include the trigger word (`ohwx`).

### Adding new K3s resources
Place YAML files in `k8s/`. All pods must comply with the `restricted` Pod Security Standard: `runAsNonRoot`, `runAsUser: 1000`, `allowPrivilegeEscalation: false`, `capabilities.drop: ["ALL"]`, `seccompProfile: RuntimeDefault`.

## Testing

- **caption.py**: Requires `pip install openai`. Test locally against any OpenAI-compatible vision API endpoint: `python scripts/caption.py --api-url <url> --image-dir <dir>`. For reasoning models (QwQ, Qwen3), add `--no-think`.
- **JSON configs**: Validate with `python -c "import json; json.load(open('k8s/job/configmap.yaml'))"` -- or use `make validate` if defined. Each JSON block inside the ConfigMap can be extracted and parsed individually.
- **Dockerfile**: Build with `make build` (requires ARM64 host or buildx for cross-compilation). The NGC base image is ~15GB.
- **K8s manifests**: Dry-run with `kubectl apply --dry-run=client -f k8s/job/train-job.yaml`.

## Do NOT

- Deploy or run training from this dev environment. Build and test only.
- Change the base image without confirming ARM64 + Blackwell + CUDA 13.0.x compatibility.
- Add heavy dependencies to `scripts/caption.py` (only `openai` is allowed as an external dep).
- Remove the non-root user from the Dockerfile or the security contexts from `k8s/job/train-job.yaml`.
- Hardcode HuggingFace tokens or secrets into any file. Use K8s Secrets or env vars.
