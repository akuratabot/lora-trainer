# Design: Colab-Compatible LoRA Training Notebook

**Date:** 2026-03-28  
**Status:** Approved

## Overview

Add a Google Colab-compatible Jupyter notebook (`colab/train_lora.ipynb`) that replicates the K3s training pipeline end-to-end. The notebook is designed to run on an A100 or H100 Colab runtime, is checkpoint-aware and idempotent in every section, and uses Google Drive for persistent input/output.

## Goals

- Mirror the K3s Job behaviour: download model + dataset, write config files, run SimpleTuner training
- Survive Colab disconnections: every section checks whether its work is already done before repeating it
- Zero hardcoded secrets: HuggingFace token read from Colab Secrets vault
- Persistent outputs: final LoRA weights saved to Google Drive AND offered as a browser download

## Non-goals

- Captioning (remains a local script via `scripts/caption.py`)
- Multi-GPU / multi-node training
- Pushing to HuggingFace Hub

## Repository location

```
colab/
└── train_lora.ipynb
```

## Directory layout inside the Colab runtime

```
/content/
├── drive/MyDrive/lora-trainer/         ← Google Drive mount point
│   ├── input/
│   │   └── batch1.zip                  ← USER-PROVIDED: captioned subject images
│   └── output/
│       └── qwen-batch1/                ← LoRA weights written here (persists across reconnects)
│
└── data/                               ← ephemeral working dir (runtime-local, fast SSD)
    ├── models/
    │   └── Qwen-Image-2512/            ← ~40 GB base model (re-downloaded on fresh runtime)
    ├── dataset/
    │   ├── images/                     ← extracted subject images + .txt caption files
    │   └── regularization/
    │       └── images/                 ← bghira/pseudo-camera-10k regularization set
    ├── cache/
    │   ├── vae/
    │   │   ├── subject/
    │   │   └── regularization/
    │   └── text/
    │       └── qwen_image/
    └── config/                         ← generated SimpleTuner config files
        ├── config.json
        ├── multidatabackend.json
        ├── config.env
        └── user_prompt_library.json
```

> **Optional model cache on Drive:** The notebook includes a clearly-labelled optional Markdown + code cell between Section 1 (mount Drive) and Section 2 (install deps) that, if run, creates `/content/drive/MyDrive/lora-trainer/model-cache/` and symlinks `/content/data/models` → that Drive path. This means the ~40 GB download persists across Colab reconnects at the cost of slower I/O from Drive. The cell is clearly marked optional and skippable; if skipped, the model re-downloads to ephemeral `/content/data/models/` on each fresh runtime.

## Notebook sections

### 0. Prerequisites (Markdown cell)

What the user must set up before running:

1. **Colab Secret**: `HF_TOKEN` — HuggingFace access token with read access to `Qwen/Qwen-Image-2512`
2. **Google Drive file**: `MyDrive/lora-trainer/input/batch1.zip` — zip of captioned subject images (images + matching `.txt` files)
3. **Runtime type**: A100 or H100 GPU (Colab Pro/Pro+). At least 40 GB VRAM required.

### 1. Config cell (user-editable Python variables)

```python
# ── User config ────────────────────────────────────────────────────────────────
TRIGGER_WORD        = "ohwx"
LORA_OUTPUT_NAME    = "qwen-batch1"
MAX_TRAIN_STEPS     = 3000
TRAIN_BATCH_SIZE    = 1          # A100 80 GB can handle 2; H100 can handle 2-4
LORA_RANK           = 64
LORA_ALPHA          = 64
LEARNING_RATE       = 3e-5

# Attention mechanism — choose based on your GPU:
#   "flash_attention_2"  → A100, H100 (most common Colab GPU)
#   "sdpa"               → any GPU, safest fallback
#   "flash-attn-3"       → Blackwell only (not available on Colab)
ATTENTION_MECHANISM = "flash_attention_2"

# Paths
DATASET_ZIP_DRIVE_PATH = "MyDrive/lora-trainer/input/batch1.zip"
OUTPUT_DRIVE_DIR       = f"MyDrive/lora-trainer/output/{LORA_OUTPUT_NAME}"
# ──────────────────────────────────────────────────────────────────────────────
```

### 2. Section 0: Runtime check

- Run `nvidia-smi` and print GPU model + VRAM
- Assert CUDA is available via `torch.cuda.is_available()`
- Print a warning if GPU is not A100/H100

### 3. Section 1: Mount Google Drive

- Call `google.colab.drive.mount('/content/drive')`
- Idempotent: skip if `/content/drive/MyDrive` already exists
- Verify `DATASET_ZIP_DRIVE_PATH` exists; print a clear error if not found

### 4. Section 2: Install dependencies

- Install SimpleTuner from git main: `pip install 'simpletuner[cuda] @ git+https://github.com/bghira/SimpleTuner.git@main'`
  - Use `[cuda]` extra (not `[cuda13]`; that is Blackwell-only)
  - Pin to `@main` to include the batch>1 Qwen-Image attention fix (commit a47da0d)
- Apply the `safety_check.py` patch: locate the installed file at `$(python -c "import simpletuner; import os; print(os.path.dirname(simpletuner.__file__))")/helpers/training/default_settings/safety_check.py` and use a Python `str.replace` (or `sed -i`) to replace:
  ```
  total_memory = int(output.decode().strip()) / 1024
  ```
  with:
  ```
  raw = output.decode().strip(); total_memory = (int(raw) if raw.lstrip("-").isdigit() else 131072) / 1024
  ```
  This handles `nvidia-smi` returning `[N/A]` for unified-memory GPUs (e.g. GB10); on Colab discrete GPUs `nvidia-smi` returns a valid integer so the patch is a no-op functionally but keeps parity with the Dockerfile.
- Idempotent: check `importlib.util.find_spec("simpletuner")` before reinstalling

### 5. Section 3: Prepare dataset

**Subject images:**
- Unzip `batch1.zip` → `/content/data/dataset/images/`
- Idempotent: skip if dir is non-empty

**Regularization images:**
- `huggingface_hub.snapshot_download("bghira/pseudo-camera-10k", repo_type="dataset", local_dir=...)`
- Idempotent: skip if dir is non-empty

### 6. Section 4: Download base model

- Read `HF_TOKEN` from `google.colab.userdata.get("HF_TOKEN")`
- `huggingface_hub.snapshot_download("Qwen/Qwen-Image-2512", local_dir="/content/data/models/Qwen-Image-2512", token=HF_TOKEN)`
- Idempotent: skip if `/content/data/models/Qwen-Image-2512/config.json` already exists
- Print size of download and estimated time (~10-15 min on Colab)

### 7. Section 5: Write config files

Generate all 4 SimpleTuner config files at `/content/data/config/`. Always regenerates (fast, keeps config in sync with user variables).

**`config.json`** — built from the K8s configmap values with these substitutions:
- `pretrained_model_name_or_path`: `/content/data/models/Qwen-Image-2512`
- `data_backend_config`: `/content/data/config/multidatabackend.json`
- `output_dir`: `/content/drive/MyDrive/<OUTPUT_DRIVE_DIR>` (writes directly to Drive so checkpoints persist)
- `user_prompt_library`: `/content/data/config/user_prompt_library.json`
- `attention_mechanism`: value of `ATTENTION_MECHANISM` config variable
- `lora_rank`, `lora_alpha`, `learning_rate`, `max_train_steps`, `train_batch_size`: from config cell variables
- All other values (e.g. `model_family`, `mixed_precision`, `base_model_precision`, `gradient_checkpointing`, `max_grad_norm`, `optimizer`, `resume_from_checkpoint`, etc.) copied verbatim from the K8s configmap

**`multidatabackend.json`** — copied verbatim from the K8s configmap with only path substitutions:
- `instance_data_dir` for subject images: `/content/data/dataset/images`
- `instance_data_dir` for regularization images: `/content/data/dataset/regularization/images`
- `cache_dir_vae` for subject: `/content/data/cache/vae/subject`
- `cache_dir_vae` for regularization: `/content/data/cache/vae/regularization`
- `cache_dir` for text embeds: `/content/data/cache/text/qwen_image`
- All other fields (ids, repeats, resolution, caption strategies, etc.) unchanged

**`user_prompt_library.json`** — copied verbatim from the K8s configmap, no substitutions needed (prompts contain no paths)

**`config.env`** — written from the K8s configmap values verbatim:
```
export TRAINING_NUM_PROCESSES=1
export TRAINING_NUM_MACHINES=1
export TRAINING_DYNAMO_BACKEND=no
export MIXED_PRECISION=bf16
export SIMPLETUNER_LOG_LEVEL=INFO
```

### 8. Section 6: Train

- Set environment variables from `config.env` values (`TRAINING_NUM_PROCESSES=1`, `MIXED_PRECISION=bf16`, etc.)
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Set `HF_TOKEN` in env for SimpleTuner's hub operations
- Run training via subprocess: `simpletuner train` with `cwd=/content/data`
- Stream stdout/stderr live to the notebook cell output
- SimpleTuner's `resume_from_checkpoint: latest` handles reconnection automatically — re-running this cell resumes from the latest checkpoint

### 9. Section 7: Save outputs

- **Drive copy**: Not needed as a separate step — `output_dir` in `config.json` is set to `OUTPUT_DRIVE_DIR` directly, so SimpleTuner writes all checkpoints and weights to Drive throughout training.
- **Browser download**: zip the final `.safetensors` files from `OUTPUT_DRIVE_DIR` and call `google.colab.files.download()`
- Both operations guarded: only run if output dir exists and contains `.safetensors` files
- Print the final LoRA path so user knows where to find it

## Key differences from K8s pipeline

| Aspect | K8s Job | Colab Notebook |
|---|---|---|
| Dataset source | Internal filebrowser service | Google Drive zip |
| Model source | `hf download` in initContainer | `snapshot_download` in notebook |
| Config delivery | ConfigMap mounted as files | Generated in-notebook from Python vars |
| Output destination | `/shared/comfyui/models/loras/` on PVC | Google Drive + browser download |
| Attention mechanism | `flash-attn-3` (Blackwell) | User-selected via config cell |
| CUDA extra | `[cuda13]` (CUDA 13/Blackwell) | `[cuda]` (standard CUDA 12.x) |
| Security context | Non-root, restricted PSS | Default Colab (root) |

## Constraints and notes

- The `[cuda]` SimpleTuner extra is correct for Colab A100/H100 (CUDA 12.x). Do NOT use `[cuda13]`.
- `flash-attn-3` is not available on A100/H100; use `flash_attention_2` or `sdpa`.
- `train_batch_size: 1` is the safe default. A100 80 GB can handle 2 with `gradient_checkpointing: true`.
- The model download (~40 GB) will need to be repeated on fresh runtimes unless the user optionally caches to Drive.
- Colab Pro+ with A100 or H100 is strongly recommended. T4 (16 GB) is insufficient for this model.
