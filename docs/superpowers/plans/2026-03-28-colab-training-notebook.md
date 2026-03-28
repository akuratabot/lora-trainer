# Colab Training Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `colab/train_lora.ipynb` — a Google Colab-compatible Jupyter notebook that replicates the K3s LoRA training pipeline end-to-end, using Google Drive for input/output and designed to survive runtime disconnections.

**Architecture:** A single `.ipynb` with idempotent, checkpoint-aware sections. Each section checks whether its work is already done before repeating it. Config values live in one user-editable Python cell at the top. The notebook mirrors the K8s Job exactly: download model + dataset, generate config files from the same values as the configmap, run SimpleTuner, save outputs to Drive and offer a browser download.

**Tech Stack:** Python 3, google-colab, huggingface_hub, SimpleTuner (git main, `[cuda]` extra), subprocess, shutil, json, pathlib

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `colab/train_lora.ipynb` | **Create** | The notebook itself |

That is the only file. All config content is inlined into the notebook cells from the K8s configmap values.

---

### Task 1: Create the `colab/` directory and notebook skeleton

**Files:**
- Create: `colab/train_lora.ipynb`

- [ ] **Step 1: Verify the repo root**

  Run:
  ```bash
  ls /home/node/workspace/lora-trainer
  ```
  Confirm there is no `colab/` directory yet.

- [ ] **Step 2: Create the directory**

  ```bash
  mkdir colab
  ```

- [ ] **Step 3: Create the notebook skeleton**

  Create `colab/train_lora.ipynb` as a valid `.ipynb` JSON file. The skeleton should have the correct nbformat metadata and an empty `cells` array. Use nbformat 4, nbformat_minor 5.

  ```json
  {
   "nbformat": 4,
   "nbformat_minor": 5,
   "metadata": {
    "kernelspec": {
     "display_name": "Python 3",
     "language": "python",
     "name": "python3"
    },
    "language_info": {
     "name": "python",
     "version": "3.10.0"
    },
    "colab": {
     "provenance": []
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
   },
   "cells": []
  }
  ```

- [ ] **Step 4: Validate JSON parses**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```
  Expected: `OK`

- [ ] **Step 5: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "chore: add colab notebook skeleton"
  ```

---

### Task 2: Add title, prerequisites, and directory-layout markdown cells

**Files:**
- Modify: `colab/train_lora.ipynb`

These are pure markdown (documentation) cells. No code runs here.

- [ ] **Step 1: Add the title cell**

  Append to `cells` array:

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-title",
   "metadata": {},
   "source": [
    "# Qwen-Image-2512 Character LoRA — Google Colab Training\n",
    "\n",
    "Train a character-specific LoRA on Qwen-Image-2512 (20B MMDiT) directly in Google Colab.\n",
    "\n",
    "**Recommended runtime:** A100 (80 GB) or H100 — minimum 40 GB VRAM required. T4 is insufficient.\n",
    "\n",
    "Run cells top-to-bottom. Every section is idempotent — if your runtime disconnects, reconnect, re-run all cells, and training resumes from the latest checkpoint automatically."
   ]
  }
  ```

- [ ] **Step 2: Add the prerequisites cell**

  Append to `cells` array:

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-prereqs",
   "metadata": {},
   "source": [
    "## Before You Start\n",
    "\n",
    "### 1. Colab Secret: `HF_TOKEN`\n",
    "Go to **Secrets** (key icon in the left sidebar) and add:\n",
    "- Name: `HF_TOKEN`\n",
    "- Value: your HuggingFace access token with read access to `Qwen/Qwen-Image-2512`\n",
    "- Enable **Notebook access**\n",
    "\n",
    "### 2. Google Drive: training dataset\n",
    "Put your captioned subject images zip at:\n",
    "```\n",
    "MyDrive/lora-trainer/input/batch1.zip\n",
    "```\n",
    "The zip should contain images + matching `.txt` caption files (one `.txt` per image, same filename).\n",
    "Use `scripts/caption.py` from the repo to generate captions if you haven't yet.\n",
    "\n",
    "### Directory layout (for reference)\n",
    "```\n",
    "/content/\n",
    "├── drive/MyDrive/lora-trainer/\n",
    "│   ├── input/\n",
    "│   │   └── batch1.zip          ← YOUR DATASET (put here before running)\n",
    "│   └── output/\n",
    "│       └── qwen-batch1/        ← LoRA weights saved here (persists after disconnect)\n",
    "│\n",
    "└── data/                       ← ephemeral working dir (lost on runtime reset)\n",
    "    ├── models/Qwen-Image-2512/ ← ~40 GB base model\n",
    "    ├── dataset/\n",
    "    │   ├── images/             ← your subject images + .txt captions\n",
    "    │   └── regularization/\n",
    "    │       └── images/         ← regularization dataset (auto-downloaded)\n",
    "    ├── cache/                  ← SimpleTuner VAE + text embed cache\n",
    "    └── config/                 ← generated config files\n",
    "```"
   ]
  }
  ```

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```
  Expected: `OK`

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "docs(colab): add title, prerequisites, and directory layout cells"
  ```

---

### Task 3: Add the user config cell

**Files:**
- Modify: `colab/train_lora.ipynb`

This is the only cell users need to edit. All subsequent cells read from these variables.

- [ ] **Step 1: Append the config code cell**

  Append to `cells`:

  ```json
  {
   "cell_type": "code",
   "id": "cell-config",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── User Config ────────────────────────────────────────────────────────────────\n",
    "# Edit these values before running the notebook.\n",
    "\n",
    "TRIGGER_WORD        = \"ohwx\"          # Trigger word embedded in all captions\n",
    "LORA_OUTPUT_NAME    = \"qwen-batch1\"   # Subfolder name under Drive output dir\n",
    "MAX_TRAIN_STEPS     = 3000\n",
    "TRAIN_BATCH_SIZE    = 1               # A100 80 GB can handle 2; H100 can handle 2-4\n",
    "LORA_RANK           = 64\n",
    "LORA_ALPHA          = 64\n",
    "LEARNING_RATE       = 3e-5\n",
    "\n",
    "# Attention mechanism — choose based on your GPU:\n",
    "#   \"flash_attention_2\"  → A100, H100 (recommended for Colab)\n",
    "#   \"sdpa\"               → any GPU, safe fallback with lower throughput\n",
    "#   \"flash-attn-3\"       → Blackwell only (GB10/GB200), NOT available on Colab\n",
    "ATTENTION_MECHANISM = \"flash_attention_2\"\n",
    "\n",
    "# Paths on Google Drive\n",
    "DATASET_ZIP_DRIVE_PATH = f\"/content/drive/MyDrive/lora-trainer/input/batch1.zip\"\n",
    "OUTPUT_DRIVE_DIR       = f\"/content/drive/MyDrive/lora-trainer/output/{LORA_OUTPUT_NAME}\"\n",
    "\n",
    "# Internal working paths (ephemeral, under /content/data)\n",
    "DATA_ROOT            = \"/content/data\"\n",
    "MODEL_DIR            = f\"{DATA_ROOT}/models/Qwen-Image-2512\"\n",
    "SUBJECT_IMAGES_DIR   = f\"{DATA_ROOT}/dataset/images\"\n",
    "REG_IMAGES_DIR       = f\"{DATA_ROOT}/dataset/regularization/images\"\n",
    "CONFIG_DIR           = f\"{DATA_ROOT}/config\"\n",
    "VAE_CACHE_SUBJECT    = f\"{DATA_ROOT}/cache/vae/subject\"\n",
    "VAE_CACHE_REG        = f\"{DATA_ROOT}/cache/vae/regularization\"\n",
    "TEXT_CACHE_DIR       = f\"{DATA_ROOT}/cache/text/qwen_image\"\n",
    "# ──────────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "print(\"Config loaded.\")\n",
    "print(f\"  Output name : {LORA_OUTPUT_NAME}\")\n",
    "print(f\"  Train steps : {MAX_TRAIN_STEPS}\")\n",
    "print(f\"  Batch size  : {TRAIN_BATCH_SIZE}\")\n",
    "print(f\"  Attention   : {ATTENTION_MECHANISM}\")"
   ]
  }
  ```

- [ ] **Step 2: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add user config cell"
  ```

---

### Task 4: Add Section 0 — Runtime check cell

**Files:**
- Modify: `colab/train_lora.ipynb`

- [ ] **Step 1: Append section header markdown cell**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s0-header",
   "metadata": {},
   "source": ["## Section 0: Runtime Check\n", "Verify a GPU is attached and print its specs."]
  }
  ```

- [ ] **Step 2: Append runtime check code cell**

  ```json
  {
   "cell_type": "code",
   "id": "cell-s0-check",
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess, sys\n",
    "\n",
    "result = subprocess.run([\"nvidia-smi\", \"--query-gpu=name,memory.total\", \"--format=csv,noheader\"],\n",
    "                        capture_output=True, text=True)\n",
    "if result.returncode != 0:\n",
    "    raise RuntimeError(\"No GPU found. Change runtime type to GPU (A100 or H100).\")\n",
    "\n",
    "gpu_info = result.stdout.strip()\n",
    "print(f\"GPU: {gpu_info}\")\n",
    "\n",
    "import torch\n",
    "assert torch.cuda.is_available(), \"CUDA not available — check runtime type\"\n",
    "print(f\"CUDA: {torch.version.cuda}\")\n",
    "print(f\"PyTorch: {torch.__version__}\")\n",
    "\n",
    "gpu_name = gpu_info.split(',')[0].strip().lower()\n",
    "if not any(g in gpu_name for g in ['a100', 'h100', 'h200']):\n",
    "    print(f\"\\nWARNING: GPU '{gpu_info.split(chr(44))[0].strip()}' may not have enough VRAM.\")\n",
    "    print(\"Recommended: A100 (80 GB) or H100. Minimum 40 GB VRAM required.\")\n",
    "else:\n",
    "    print(\"GPU OK.\")"
   ]
  }
  ```

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add runtime check cell"
  ```

---

### Task 5: Add Section 1 — Mount Google Drive

**Files:**
- Modify: `colab/train_lora.ipynb`

- [ ] **Step 1: Append section header markdown**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s1-header",
   "metadata": {},
   "source": [
    "## Section 1: Mount Google Drive\n",
    "Mounts Drive and verifies your dataset zip is present."
   ]
  }
  ```

- [ ] **Step 2: Append Drive mount code cell**

  ```json
  {
   "cell_type": "code",
   "id": "cell-s1-mount",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# Mount Drive (idempotent)\n",
    "if not Path('/content/drive/MyDrive').exists():\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "else:\n",
    "    print(\"Drive already mounted.\")\n",
    "\n",
    "# Verify dataset zip exists\n",
    "zip_path = Path(DATASET_ZIP_DRIVE_PATH)\n",
    "if not zip_path.exists():\n",
    "    raise FileNotFoundError(\n",
    "        f\"Dataset zip not found at {zip_path}\\n\"\n",
    "        \"Please upload batch1.zip to MyDrive/lora-trainer/input/ and re-run.\"\n",
    "    )\n",
    "print(f\"Dataset zip found: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)\")\n",
    "\n",
    "# Create output directory on Drive\n",
    "Path(OUTPUT_DRIVE_DIR).mkdir(parents=True, exist_ok=True)\n",
    "print(f\"Output directory ready: {OUTPUT_DRIVE_DIR}\")"
   ]
  }
  ```

- [ ] **Step 3: Append optional model cache markdown cell**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s1-cache-note",
   "metadata": {},
   "source": [
    "### Optional: Cache model weights on Drive\n",
    "\n",
    "The base model (~40 GB) is re-downloaded every time you get a fresh runtime. If you want to avoid this, run the cell below **once** to symlink the model directory to Google Drive. Subsequent runtimes will find the weights already there.\n",
    "\n",
    "**Trade-off:** Drive I/O is slower than Colab's local SSD. Training startup will take a few extra minutes.\n",
    "\n",
    "**Skip this cell if you prefer a fresh download each time (faster during training).**"
   ]
  }
  ```

- [ ] **Step 4: Append optional model cache code cell**

  ```json
  {
   "cell_type": "code",
   "id": "cell-s1-cache",
   "metadata": {},
   "outputs": [],
   "source": [
    "# OPTIONAL — run this cell to cache the model on Drive\n",
    "# Skip entirely if you want the model on fast local SSD instead.\n",
    "\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "drive_model_cache = \"/content/drive/MyDrive/lora-trainer/model-cache/Qwen-Image-2512\"\n",
    "Path(drive_model_cache).mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "local_model_parent = Path(MODEL_DIR).parent\n",
    "local_model_parent.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "if Path(MODEL_DIR).is_symlink():\n",
    "    print(f\"Symlink already exists: {MODEL_DIR} -> {os.readlink(MODEL_DIR)}\")\n",
    "elif Path(MODEL_DIR).exists():\n",
    "    print(f\"{MODEL_DIR} already exists as a real directory (not symlinking).\")\n",
    "else:\n",
    "    os.symlink(drive_model_cache, MODEL_DIR)\n",
    "    print(f\"Symlinked {MODEL_DIR} -> {drive_model_cache}\")\n",
    "    print(\"Model weights will be cached to Drive and reused across runtimes.\")"
   ]
  }
  ```

- [ ] **Step 5: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add Drive mount and optional model cache cells"
  ```

---

### Task 6: Add Section 2 — Install dependencies

**Files:**
- Modify: `colab/train_lora.ipynb`

- [ ] **Step 1: Append section header markdown**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s2-header",
   "metadata": {},
   "source": [
    "## Section 2: Install Dependencies\n",
    "Installs SimpleTuner from git main (required for Qwen-Image batch>1 fix). Skips if already installed."
   ]
  }
  ```

- [ ] **Step 2: Append install code cell**

  ```json
  {
   "cell_type": "code",
   "id": "cell-s2-install",
   "metadata": {},
   "outputs": [],
   "source": [
    "import importlib.util, subprocess, sys\n",
    "\n",
    "def run(cmd, **kwargs):\n",
    "    \"\"\"Run a shell command, streaming output, raising on failure.\"\"\"\n",
    "    print(f\"$ {cmd}\")\n",
    "    result = subprocess.run(cmd, shell=True, **kwargs)\n",
    "    if result.returncode != 0:\n",
    "        raise RuntimeError(f\"Command failed: {cmd}\")\n",
    "\n",
    "# Install SimpleTuner if not already present\n",
    "if importlib.util.find_spec(\"simpletuner\") is None:\n",
    "    print(\"Installing SimpleTuner from git main...\")\n",
    "    run(\"pip install -q 'simpletuner[cuda] @ git+https://github.com/bghira/SimpleTuner.git@main'\")\n",
    "    print(\"SimpleTuner installed.\")\n",
    "else:\n",
    "    print(\"SimpleTuner already installed, skipping.\")\n",
    "\n",
    "# Apply safety_check.py patch (handles nvidia-smi returning [N/A] on unified-memory GPUs).\n",
    "# On Colab discrete GPUs this is a no-op functionally, but keeps parity with the Dockerfile.\n",
    "import simpletuner, os\n",
    "safety_check_path = os.path.join(\n",
    "    os.path.dirname(simpletuner.__file__),\n",
    "    \"helpers\", \"training\", \"default_settings\", \"safety_check.py\"\n",
    ")\n",
    "old_line = \"total_memory = int(output.decode().strip()) / 1024\"\n",
    "new_line  = \"raw = output.decode().strip(); total_memory = (int(raw) if raw.lstrip('-').isdigit() else 131072) / 1024\"\n",
    "\n",
    "with open(safety_check_path, 'r') as f:\n",
    "    content = f.read()\n",
    "\n",
    "if old_line in content:\n",
    "    with open(safety_check_path, 'w') as f:\n",
    "        f.write(content.replace(old_line, new_line))\n",
    "    print(\"safety_check.py patched.\")\n",
    "else:\n",
    "    print(\"safety_check.py already patched or line not found (skipping).\")\n",
    "\n",
    "print(\"Dependencies ready.\")"
   ]
  }
  ```

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add dependency install cell"
  ```

---

### Task 7: Add Section 3 — Prepare dataset

**Files:**
- Modify: `colab/train_lora.ipynb`

- [ ] **Step 1: Append section header markdown**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s3-header",
   "metadata": {},
   "source": [
    "## Section 3: Prepare Dataset\n",
    "Extracts subject images from Drive zip and downloads regularization images from HuggingFace. Both steps are idempotent."
   ]
  }
  ```

- [ ] **Step 2: Append dataset prep code cell**

  ```json
  {
   "cell_type": "code",
   "id": "cell-s3-dataset",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, zipfile\n",
    "from pathlib import Path\n",
    "from huggingface_hub import snapshot_download\n",
    "\n",
    "# ── Subject images ──────────────────────────────────────────────────────────\n",
    "subject_dir = Path(SUBJECT_IMAGES_DIR)\n",
    "subject_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "existing_images = list(subject_dir.glob('*.jpg')) + list(subject_dir.glob('*.png')) + list(subject_dir.glob('*.jpeg'))\n",
    "if existing_images:\n",
    "    print(f\"Subject images already extracted ({len(existing_images)} images). Skipping unzip.\")\n",
    "else:\n",
    "    print(f\"Extracting {DATASET_ZIP_DRIVE_PATH} ...\")\n",
    "    with zipfile.ZipFile(DATASET_ZIP_DRIVE_PATH, 'r') as zf:\n",
    "        zf.extractall(str(subject_dir))\n",
    "    images_after = list(subject_dir.rglob('*.jpg')) + list(subject_dir.rglob('*.png')) + list(subject_dir.rglob('*.jpeg'))\n",
    "    print(f\"Extracted {len(images_after)} images to {subject_dir}\")\n",
    "\n",
    "# Verify captions exist\n",
    "all_images = list(subject_dir.rglob('*.jpg')) + list(subject_dir.rglob('*.png')) + list(subject_dir.rglob('*.jpeg'))\n",
    "missing_captions = [img for img in all_images if not img.with_suffix('.txt').exists()]\n",
    "if missing_captions:\n",
    "    print(f\"WARNING: {len(missing_captions)} images have no .txt caption file:\")\n",
    "    for p in missing_captions[:5]:\n",
    "        print(f\"  {p}\")\n",
    "    if len(missing_captions) > 5:\n",
    "        print(f\"  ... and {len(missing_captions) - 5} more\")\n",
    "else:\n",
    "    print(f\"All {len(all_images)} images have captions. Subject dataset ready.\")\n",
    "\n",
    "# ── Regularization images ───────────────────────────────────────────────────\n",
    "reg_dir = Path(REG_IMAGES_DIR)\n",
    "reg_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "reg_images = list(reg_dir.rglob('*.jpg')) + list(reg_dir.rglob('*.png'))\n",
    "if reg_images:\n",
    "    print(f\"Regularization images already present ({len(reg_images)} images). Skipping download.\")\n",
    "else:\n",
    "    print(\"Downloading bghira/pseudo-camera-10k regularization dataset...\")\n",
    "    print(\"This may take several minutes.\")\n",
    "    snapshot_download(\n",
    "        repo_id=\"bghira/pseudo-camera-10k\",\n",
    "        repo_type=\"dataset\",\n",
    "        local_dir=str(reg_dir),\n",
    "    )\n",
    "    reg_images_after = list(reg_dir.rglob('*.jpg')) + list(reg_dir.rglob('*.png'))\n",
    "    print(f\"Downloaded {len(reg_images_after)} regularization images.\")\n",
    "\n",
    "print(\"Dataset preparation complete.\")"
   ]
  }
  ```

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add dataset preparation cell"
  ```

---

### Task 8: Add Section 4 — Download base model

**Files:**
- Modify: `colab/train_lora.ipynb`

- [ ] **Step 1: Append section header markdown**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s4-header",
   "metadata": {},
   "source": [
    "## Section 4: Download Base Model\n",
    "Downloads Qwen/Qwen-Image-2512 (~40 GB) from HuggingFace. Uses your `HF_TOKEN` Colab Secret. Idempotent — skips if already downloaded."
   ]
  }
  ```

- [ ] **Step 2: Append model download code cell**

  ```json
  {
   "cell_type": "code",
   "id": "cell-s4-model",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from pathlib import Path\n",
    "from huggingface_hub import snapshot_download\n",
    "from google.colab import userdata\n",
    "\n",
    "# Read HF token from Colab Secrets\n",
    "try:\n",
    "    HF_TOKEN = userdata.get('HF_TOKEN')\n",
    "    print(\"HF_TOKEN loaded from Colab Secrets.\")\n",
    "except Exception:\n",
    "    raise RuntimeError(\n",
    "        \"HF_TOKEN not found in Colab Secrets.\\n\"\n",
    "        \"Go to the key icon in the left sidebar, add HF_TOKEN, and enable Notebook access.\"\n",
    "    )\n",
    "\n",
    "model_dir = Path(MODEL_DIR)\n",
    "\n",
    "# Idempotency check: presence of config.json in the model directory\n",
    "if (model_dir / 'config.json').exists():\n",
    "    print(f\"Model already present at {model_dir}. Skipping download.\")\n",
    "else:\n",
    "    model_dir.mkdir(parents=True, exist_ok=True)\n",
    "    print(f\"Downloading Qwen/Qwen-Image-2512 to {model_dir} ...\")\n",
    "    print(\"This takes ~10-15 minutes on a fresh Colab runtime (~40 GB).\")\n",
    "    snapshot_download(\n",
    "        repo_id=\"Qwen/Qwen-Image-2512\",\n",
    "        local_dir=str(model_dir),\n",
    "        token=HF_TOKEN,\n",
    "    )\n",
    "    print(\"Model download complete.\")\n",
    "\n",
    "# Store token in env for SimpleTuner's hub calls\n",
    "os.environ['HF_TOKEN'] = HF_TOKEN\n",
    "print(\"Base model ready.\")"
   ]
  }
  ```

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add base model download cell"
  ```

---

### Task 9: Add Section 5 — Write config files

**Files:**
- Modify: `colab/train_lora.ipynb`

Config values are inlined verbatim from the K8s configmap with path substitutions. See the spec for the full mapping.

- [ ] **Step 1: Append section header markdown**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s5-header",
   "metadata": {},
   "source": [
    "## Section 5: Write Config Files\n",
    "Generates all 4 SimpleTuner config files from your config variables. Always regenerates to stay in sync."
   ]
  }
  ```

- [ ] **Step 2: Append config generation code cell**

  The cell builds all 4 files using Python dicts/json.dumps (not f-strings into JSON, which is error-prone):

  ```json
  {
   "cell_type": "code",
   "id": "cell-s5-config",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json, os\n",
    "from pathlib import Path\n",
    "\n",
    "config_dir = Path(CONFIG_DIR)\n",
    "config_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# ── config.json ─────────────────────────────────────────────────────────────\n",
    "config_json = {\n",
    "    \"model_family\": \"qwen_image\",\n",
    "    \"model_flavour\": \"v1.0\",\n",
    "    \"model_type\": \"lora\",\n",
    "    \"pretrained_model_name_or_path\": MODEL_DIR,\n",
    "    \"base_model_precision\": \"no_change\",\n",
    "    \"mixed_precision\": \"bf16\",\n",
    "    \"lora_type\": \"standard\",\n",
    "    \"lora_rank\": LORA_RANK,\n",
    "    \"lora_alpha\": LORA_ALPHA,\n",
    "    \"learning_rate\": LEARNING_RATE,\n",
    "    \"lr_scheduler\": \"constant_with_warmup\",\n",
    "    \"lr_warmup_steps\": 100,\n",
    "    \"optimizer\": \"optimi-lion\",\n",
    "    \"train_batch_size\": TRAIN_BATCH_SIZE,\n",
    "    \"gradient_checkpointing\": True,\n",
    "    \"max_train_steps\": MAX_TRAIN_STEPS,\n",
    "    \"num_train_epochs\": 0,\n",
    "    \"max_grad_norm\": 0.01,\n",
    "    \"seed\": 42,\n",
    "    \"flow_schedule_shift\": 1.73,\n",
    "    \"resolution\": 1024,\n",
    "    \"resolution_type\": \"pixel_area\",\n",
    "    \"minimum_image_size\": 0,\n",
    "    \"caption_dropout_probability\": 0.0,\n",
    "    \"data_backend_config\": f\"{CONFIG_DIR}/multidatabackend.json\",\n",
    "    \"output_dir\": OUTPUT_DRIVE_DIR,\n",
    "    \"checkpoint_step_interval\": 500,\n",
    "    \"checkpoints_total_limit\": 5,\n",
    "    \"resume_from_checkpoint\": \"latest\",\n",
    "    \"ignore_final_epochs\": True,\n",
    "    \"disable_bucket_pruning\": True,\n",
    "    \"validation_steps\": 1000,\n",
    "    \"validation_guidance\": 4.0,\n",
    "    \"validation_guidance_rescale\": 0.0,\n",
    "    \"validation_lycoris_strength\": 1.0,\n",
    "    \"validation_num_inference_steps\": 20,\n",
    "    \"validation_prompt\": f\"A photo of {TRIGGER_WORD} standing against a white background, natural lighting\",\n",
    "    \"validation_prompt_library\": False,\n",
    "    \"user_prompt_library\": f\"{CONFIG_DIR}/user_prompt_library.json\",\n",
    "    \"validation_negative_prompt\": \"ugly, cropped, blurry, low-quality, deformed, disfigured\",\n",
    "    \"validation_resolution\": \"1024x1024\",\n",
    "    \"validation_seed\": 42,\n",
    "    \"num_eval_images\": 1,\n",
    "    \"report_to\": \"none\",\n",
    "    \"tracker_project_name\": \"qwen-character-lora\",\n",
    "    \"tracker_run_name\": f\"{TRIGGER_WORD}-training\",\n",
    "    \"push_to_hub\": False,\n",
    "    \"push_checkpoints_to_hub\": False,\n",
    "    \"use_ema\": False,\n",
    "    \"vae_batch_size\": 1,\n",
    "    \"compress_disk_cache\": False,\n",
    "    \"attention_mechanism\": ATTENTION_MECHANISM,\n",
    "    \"disable_benchmark\": False,\n",
    "    \"skip_file_discovery\": False\n",
    "}\n",
    "\n",
    "(config_dir / 'config.json').write_text(json.dumps(config_json, indent=4))\n",
    "print(\"Wrote config.json\")\n",
    "\n",
    "# ── multidatabackend.json ────────────────────────────────────────────────────\n",
    "multidatabackend = [\n",
    "    {\n",
    "        \"id\": \"subject-images\",\n",
    "        \"type\": \"local\",\n",
    "        \"instance_data_dir\": SUBJECT_IMAGES_DIR,\n",
    "        \"caption_strategy\": \"textfile\",\n",
    "        \"crop\": False,\n",
    "        \"resolution\": 1024,\n",
    "        \"resolution_type\": \"pixel_area\",\n",
    "        \"minimum_image_size\": 512,\n",
    "        \"maximum_image_size\": 1328,\n",
    "        \"target_downsample_size\": 1024,\n",
    "        \"cache_dir_vae\": VAE_CACHE_SUBJECT,\n",
    "        \"metadata_backend\": \"discovery\",\n",
    "        \"repeats\": 20,\n",
    "        \"is_regularisation_data\": False,\n",
    "        \"disabled\": False,\n",
    "        \"skip_file_discovery\": \"\"\n",
    "    },\n",
    "    {\n",
    "        \"id\": \"regularization-images\",\n",
    "        \"type\": \"local\",\n",
    "        \"instance_data_dir\": REG_IMAGES_DIR,\n",
    "        \"caption_strategy\": \"instanceprompt\",\n",
    "        \"instance_prompt\": \"a photo of a person\",\n",
    "        \"crop\": True,\n",
    "        \"crop_style\": \"center\",\n",
    "        \"crop_aspect\": \"square\",\n",
    "        \"resolution\": 1024,\n",
    "        \"resolution_type\": \"pixel_area\",\n",
    "        \"minimum_image_size\": 512,\n",
    "        \"maximum_image_size\": 1328,\n",
    "        \"target_downsample_size\": 1024,\n",
    "        \"cache_dir_vae\": VAE_CACHE_REG,\n",
    "        \"metadata_backend\": \"discovery\",\n",
    "        \"repeats\": 0,\n",
    "        \"is_regularisation_data\": True,\n",
    "        \"disabled\": False,\n",
    "        \"skip_file_discovery\": \"\"\n",
    "    },\n",
    "    {\n",
    "        \"id\": \"text-embeds\",\n",
    "        \"type\": \"local\",\n",
    "        \"dataset_type\": \"text_embeds\",\n",
    "        \"default\": True,\n",
    "        \"cache_dir\": TEXT_CACHE_DIR,\n",
    "        \"write_batch_size\": 16,\n",
    "        \"disabled\": False\n",
    "    }\n",
    "]\n",
    "\n",
    "(config_dir / 'multidatabackend.json').write_text(json.dumps(multidatabackend, indent=4))\n",
    "print(\"Wrote multidatabackend.json\")\n",
    "\n",
    "# ── user_prompt_library.json ─────────────────────────────────────────────────\n",
    "# Prompts reference TRIGGER_WORD; substituted from configmap values.\n",
    "user_prompts = {\n",
    "    \"identity_check\": f\"A photo of {TRIGGER_WORD} standing against a plain white background, arms at sides, looking directly at the camera, neutral expression, soft even studio lighting\",\n",
    "    \"pose_sitting\": f\"A photo of {TRIGGER_WORD} sitting cross-legged on a wooden floor, hands resting on knees, relaxed expression, warm indoor lighting\",\n",
    "    \"pose_action\": f\"A photo of {TRIGGER_WORD} running through a grassy park, mid-stride, energetic expression, bright daylight with dappled shadows\",\n",
    "    \"setting_urban\": f\"A photo of {TRIGGER_WORD} standing on a busy city sidewalk at night, neon signs reflected on wet pavement, wearing a dark overcoat\",\n",
    "    \"setting_nature\": f\"A photo of {TRIGGER_WORD} walking through a snowy pine forest at golden hour, breath visible in cold air, wearing a red winter jacket\",\n",
    "    \"outfit_formal\": f\"A photo of {TRIGGER_WORD} wearing a tailored navy blue suit and white dress shirt, standing in a modern office with floor-to-ceiling windows\",\n",
    "    \"outfit_casual\": f\"A photo of {TRIGGER_WORD} wearing a plain white t-shirt and jeans, sitting on a park bench, relaxed posture, sunny day\",\n",
    "    \"close_up\": f\"A close-up photo of {TRIGGER_WORD} from the shoulders up, looking slightly to the left, gentle smile, soft natural window light\"\n",
    "}\n",
    "\n",
    "(config_dir / 'user_prompt_library.json').write_text(json.dumps(user_prompts, indent=4))\n",
    "print(\"Wrote user_prompt_library.json\")\n",
    "\n",
    "# ── config.env ───────────────────────────────────────────────────────────────\n",
    "config_env = \"\"\"export TRAINING_NUM_PROCESSES=1\n",
    "export TRAINING_NUM_MACHINES=1\n",
    "export TRAINING_DYNAMO_BACKEND=no\n",
    "export MIXED_PRECISION=bf16\n",
    "export SIMPLETUNER_LOG_LEVEL=INFO\n",
    "\"\"\"\n",
    "(config_dir / 'config.env').write_text(config_env)\n",
    "print(\"Wrote config.env\")\n",
    "\n",
    "print(f\"\\nAll config files written to {config_dir}\")"
   ]
  }
  ```

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add config file generation cell"
  ```

---

### Task 10: Add Section 6 — Train

**Files:**
- Modify: `colab/train_lora.ipynb`

- [ ] **Step 1: Append section header markdown**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s6-header",
   "metadata": {},
   "source": [
    "## Section 6: Train\n",
    "\n",
    "Runs SimpleTuner training. Output streams live to this cell.\n",
    "\n",
    "**If your runtime disconnects:** reconnect, re-run all cells from the top, then re-run this cell. Training will resume automatically from the latest checkpoint (via `resume_from_checkpoint: latest`).\n",
    "\n",
    "Expected duration: ~2-4 hours for 3000 steps with 40-80 subject images on an A100."
   ]
  }
  ```

- [ ] **Step 2: Append training code cell**

  The cell sets env vars from `config.env` values, then runs `simpletuner train` via subprocess with live output streaming:

  ```json
  {
   "cell_type": "code",
   "id": "cell-s6-train",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, subprocess, sys\n",
    "from pathlib import Path\n",
    "\n",
    "# Set environment variables (mirrors config.env and train-job.yaml)\n",
    "os.environ.update({\n",
    "    'TRAINING_NUM_PROCESSES': '1',\n",
    "    'TRAINING_NUM_MACHINES': '1',\n",
    "    'TRAINING_DYNAMO_BACKEND': 'no',\n",
    "    'MIXED_PRECISION': 'bf16',\n",
    "    'SIMPLETUNER_LOG_LEVEL': 'INFO',\n",
    "    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',\n",
    "})\n",
    "\n",
    "# SimpleTuner discovers config at ./config/config.json relative to cwd\n",
    "# So we run with cwd=/content/data (same as workingDir in train-job.yaml)\n",
    "print(f\"Working directory: {DATA_ROOT}\")\n",
    "print(f\"Config: {CONFIG_DIR}/config.json\")\n",
    "print(f\"Output: {OUTPUT_DRIVE_DIR}\")\n",
    "print(\"Starting training...\\n\")\n",
    "\n",
    "process = subprocess.Popen(\n",
    "    [sys.executable, '-m', 'simpletuner.train'],\n",
    "    cwd=DATA_ROOT,\n",
    "    stdout=subprocess.PIPE,\n",
    "    stderr=subprocess.STDOUT,\n",
    "    text=True,\n",
    "    bufsize=1,\n",
    ")\n",
    "\n",
    "for line in process.stdout:\n",
    "    print(line, end='', flush=True)\n",
    "\n",
    "process.wait()\n",
    "if process.returncode != 0:\n",
    "    raise RuntimeError(f\"Training failed with exit code {process.returncode}\")\n",
    "\n",
    "print(\"\\nTraining complete!\")"
   ]
  }
  ```

  > **Note on the CLI entry point:** The spec reviewer flagged verifying `simpletuner train` vs `python -m simpletuner.train`. The cell uses `python -m simpletuner.train` via `sys.executable` which is unambiguous. If SimpleTuner exposes a `simpletuner` console script entry point, it can be used equivalently; `python -m simpletuner.train` is the safe fallback.

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add training cell"
  ```

---

### Task 11: Add Section 7 — Save outputs

**Files:**
- Modify: `colab/train_lora.ipynb`

- [ ] **Step 1: Append section header markdown**

  ```json
  {
   "cell_type": "markdown",
   "id": "cell-s7-header",
   "metadata": {},
   "source": [
    "## Section 7: Save Outputs\n",
    "\n",
    "Saves the trained LoRA weights to Google Drive and offers a browser download.\n",
    "\n",
    "The weights are already being written directly to Drive during training (via `output_dir`). This cell copies them to a dated subfolder for archiving and triggers a browser download as backup."
   ]
  }
  ```

- [ ] **Step 2: Append save outputs code cell**

  > **Why no `shutil.copytree`:** `output_dir` in `config.json` points directly to `OUTPUT_DRIVE_DIR` on Drive, so SimpleTuner writes checkpoints and weights there throughout training. No copy step is needed. The cell only zips the final `.safetensors` files for a browser download as backup.

  ```json
  {
   "cell_type": "code",
   "id": "cell-s7-save",
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil, zipfile, os\n",
    "from pathlib import Path\n",
    "from google.colab import files\n",
    "\n",
    "output_dir = Path(OUTPUT_DRIVE_DIR)\n",
    "\n",
    "# Check training produced output\n",
    "safetensors = list(output_dir.rglob('*.safetensors'))\n",
    "if not safetensors:\n",
    "    print(f\"No .safetensors files found in {output_dir}.\")\n",
    "    print(\"Training may not have completed. Check the training cell output.\")\n",
    "else:\n",
    "    print(f\"Found {len(safetensors)} .safetensors file(s):\")\n",
    "    for f_path in safetensors:\n",
    "        size_mb = f_path.stat().st_size / 1e6\n",
    "        print(f\"  {f_path.relative_to(output_dir)} ({size_mb:.1f} MB)\")\n",
    "\n",
    "    print(f\"\\nOutputs already saved to Drive: {output_dir}\")\n",
    "\n",
    "    # Zip for browser download\n",
    "    zip_path = Path(f\"/content/{LORA_OUTPUT_NAME}.zip\")\n",
    "    print(f\"\\nCreating zip for download: {zip_path}\")\n",
    "    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:\n",
    "        for f_path in safetensors:\n",
    "            zf.write(str(f_path), f_path.relative_to(output_dir))\n",
    "    print(f\"Zip created ({zip_path.stat().st_size / 1e6:.1f} MB). Starting download...\")\n",
    "    files.download(str(zip_path))\n",
    "\n",
    "    print(\"\\nDone! Your LoRA weights are at:\")\n",
    "    print(f\"  Drive: {output_dir}\")\n",
    "    print(f\"  Local zip: {zip_path}\")"
   ]
  }
  ```

- [ ] **Step 3: Validate JSON**

  ```bash
  python -c "import json; json.load(open('colab/train_lora.ipynb')); print('OK')"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): add output save and download cell"
  ```

---

### Task 12: Final validation and cleanup commit

**Files:**
- Modify: `colab/train_lora.ipynb` (no content changes — just verify)

- [ ] **Step 1: Validate final notebook JSON**

  ```bash
  python -c "import json; nb = json.load(open('colab/train_lora.ipynb')); print(f'OK — {len(nb[\"cells\"])} cells')"
  ```
  Expected: `OK — 18 cells` (2 title/prereq + 1 config + 2×5 sections with headers + 1 optional cache)

  Actual cell count may vary slightly — what matters is that the JSON parses and the count matches the number of cells added across all tasks.

- [ ] **Step 2: Verify cell IDs are unique**

  ```bash
  python -c "
  import json
  nb = json.load(open('colab/train_lora.ipynb'))
  ids = [c['id'] for c in nb['cells']]
  dupes = [i for i in ids if ids.count(i) > 1]
  print('Duplicate IDs:', dupes if dupes else 'None')
  "
  ```
  Expected: `Duplicate IDs: None`

- [ ] **Step 3: Verify all code cells have valid Python syntax**

  ```bash
  python -c "
  import json, ast
  nb = json.load(open('colab/train_lora.ipynb'))
  for i, cell in enumerate(nb['cells']):
      if cell['cell_type'] == 'code':
          src = ''.join(cell['source'])
          try:
              ast.parse(src)
          except SyntaxError as e:
              print(f'Cell {i} ({cell[\"id\"]}): SYNTAX ERROR: {e}')
  print('All code cells parse OK.')
  "
  ```
  Expected: `All code cells parse OK.`

- [ ] **Step 4: Final commit**

  ```bash
  git add colab/train_lora.ipynb
  git commit -m "feat(colab): complete Qwen-Image-2512 LoRA training notebook"
  ```
