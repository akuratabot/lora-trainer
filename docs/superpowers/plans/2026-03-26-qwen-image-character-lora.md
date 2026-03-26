# Qwen-Image-2512 Character LoRA Training Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a K3s-deployable training pipeline that takes photos of a person and produces a LoRA for Qwen-Image-2512 that preserves their full-body identity while allowing text-prompt control of pose, setting, and outfit.

**Architecture:** A single container image (NGC PyTorch 25.11 + SimpleTuner) used by two K3s Jobs: a captioning Job that calls an external vLLM API to generate per-image captions, and a training Job that runs SimpleTuner Dreambooth-LoRA training. All data lives on a single PVC.

**Tech Stack:** Docker (NGC PyTorch 25.11-py3 base), SimpleTuner (pip, cuda13), Python 3 (captioning script), K3s (PVC + Jobs), OpenAI-compatible vLLM API.

**Spec:** `docs/superpowers/specs/2026-03-26-qwen-image-character-lora-design.md`

---

## File Structure

```
qwen-lora-training/
├── Dockerfile
├── scripts/
│   └── caption.py                    # Captioning script (calls vLLM API)
├── config/
│   ├── config.json                   # SimpleTuner training hyperparameters
│   ├── config.env                    # SimpleTuner accelerate/env settings
│   ├── multidatabackend.json         # SimpleTuner dataset config
│   └── user_prompt_library.json      # Validation prompts
├── k8s/
│   ├── pvc.yaml                      # PersistentVolumeClaim (130Gi)
│   ├── caption-job.yaml              # K3s Job: captioning
│   └── train-job.yaml                # K3s Job: training
└── GUIDE.md                          # Step-by-step usage guide
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `qwen-lora-training/` (project root)

- [ ] **Step 1: Create the project directory structure**

```bash
mkdir -p qwen-lora-training/{scripts,config,k8s}
```

- [ ] **Step 2: Initialize git repo**

```bash
cd qwen-lora-training
git init
```

- [ ] **Step 3: Commit empty structure**

```bash
touch scripts/.gitkeep config/.gitkeep k8s/.gitkeep
git add .
git commit -m "chore: scaffold project structure"
```

---

### Task 2: Captioning Script

**Files:**
- Create: `qwen-lora-training/scripts/caption.py`

- [ ] **Step 1: Write `scripts/caption.py`**

This script iterates over images in a directory, sends each to a vLLM endpoint (OpenAI-compatible vision API), and writes a `.txt` caption file per image.

```python
#!/usr/bin/env python3
"""
Auto-caption images for LoRA training via an OpenAI-compatible VLM API.

Produces one .txt caption file per image with structured descriptions
that separate identity (trigger word) from variable attributes.

Environment variables:
  VLM_API_URL   - Base URL of the vLLM endpoint (required)
                  e.g. http://vlm-service.default.svc:8000/v1
  VLM_MODEL_NAME - Model name for the API (default: auto-detect from /v1/models)
  TRIGGER_WORD   - Trigger word for the subject (default: ohwx)
  IMAGE_DIR      - Path to input images (default: /data/dataset/images)
  CAPTION_DIR    - Path to write captions (default: /data/dataset/captions)
"""

import base64
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

SYSTEM_PROMPT = """You are a precise image captioning assistant for AI training data.

Your task: describe the person in the photo in detail, following this exact structure:

1. Begin with: "A photo of {trigger_word}"
2. Describe the person's FIXED physical traits: face shape, eye color, skin tone, hair color/style/length, distinguishing marks (tattoos, scars, birthmarks, moles, piercings). Be specific and consistent.
3. Describe the POSE and EXPRESSION: body position, hand placement, head tilt, facial expression, eye direction.
4. Describe CLOTHING and ACCESSORIES: garments, colors, fabrics, jewelry, bags, hats, glasses.
5. Describe the SETTING and LIGHTING: location, background elements, time of day, lighting direction and quality.

Rules:
- Write in natural flowing English, not tags or bullet points.
- Be factual and precise. Do not invent details you cannot see.
- Do NOT mention camera settings, lens type, or photography terminology.
- Do NOT use words like "portrait", "photograph", "image", or "picture".
- Keep the description between 50 and 150 words.
- The physical trait description should be nearly identical across all images of the same person. Only pose, clothing, and setting should vary."""


def encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return mime_map.get(ext, "image/jpeg")


def detect_model_name(api_url: str) -> str:
    """Auto-detect the model name from the /v1/models endpoint."""
    url = f"{api_url.rstrip('/')}/models"
    try:
        req = Request(url, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except (URLError, HTTPError, KeyError, IndexError) as e:
        print(f"Warning: could not auto-detect model name from {url}: {e}")
    return "default"


def caption_image(api_url: str, model_name: str, image_path: Path, trigger_word: str) -> str:
    """Send an image to the VLM API and return the caption."""
    b64 = encode_image_base64(image_path)
    mime = get_mime_type(image_path)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.replace("{trigger_word}", trigger_word)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Describe this person in detail. Begin with 'A photo of {trigger_word}'.",
                    },
                ],
            },
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }

    url = f"{api_url.rstrip('/')}/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    with urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"].strip()


def main():
    api_url = os.environ.get("VLM_API_URL")
    if not api_url:
        print("ERROR: VLM_API_URL environment variable is required", file=sys.stderr)
        sys.exit(1)

    model_name = os.environ.get("VLM_MODEL_NAME", "")
    trigger_word = os.environ.get("TRIGGER_WORD", "ohwx")
    image_dir = Path(os.environ.get("IMAGE_DIR", "/data/dataset/images"))
    caption_dir = Path(os.environ.get("CAPTION_DIR", "/data/dataset/images"))

    if not image_dir.exists():
        print(f"ERROR: Image directory does not exist: {image_dir}", file=sys.stderr)
        sys.exit(1)

    caption_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"ERROR: No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images in {image_dir}")

    # Auto-detect model name if not set
    if not model_name:
        model_name = detect_model_name(api_url)
        print(f"Auto-detected model: {model_name}")

    # Caption each image
    success = 0
    failed = 0
    for i, img_path in enumerate(images, 1):
        caption_path = caption_dir / f"{img_path.stem}.txt"

        # Skip if caption already exists
        if caption_path.exists():
            print(f"[{i}/{len(images)}] SKIP (exists): {img_path.name}")
            success += 1
            continue

        try:
            print(f"[{i}/{len(images)}] Captioning: {img_path.name} ... ", end="", flush=True)
            caption = caption_image(api_url, model_name, img_path, trigger_word)
            caption_path.write_text(caption, encoding="utf-8")
            print(f"OK ({len(caption)} chars)")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone: {success} captioned, {failed} failed, out of {len(images)} total")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x scripts/caption.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/caption.py
git commit -m "feat: add captioning script with vLLM API integration"
```

---

### Task 3: SimpleTuner Configuration Files

**Files:**
- Create: `qwen-lora-training/config/config.json`
- Create: `qwen-lora-training/config/multidatabackend.json`
- Create: `qwen-lora-training/config/user_prompt_library.json`

- [ ] **Step 1: Write `config/config.json`**

SimpleTuner primary training config, optimized for GB10 128GB unified memory at bf16.

```json
{
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "model_type": "lora",
    "pretrained_model_name_or_path": "/data/models/Qwen-Image-2512",
    "base_model_precision": "bf16",
    "mixed_precision": "bf16",
    "lora_type": "standard",
    "lora_rank": 64,
    "lora_alpha": 64,
    "learning_rate": 3e-5,
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 100,
    "optimizer": "adamw-bf16",
    "train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": true,
    "max_train_steps": 3000,
    "max_grad_norm": 0.01,
    "seed": 42,
    "noise_scheduler": "flowmatch",
    "flow_schedule_shift": 1.73,
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "minimum_image_size": 512,
    "caption_dropout_probability": 0.0,
    "data_backend_config": "/data/config/multidatabackend.json",
    "output_dir": "/data/output",
    "checkpointing_steps": 500,
    "checkpoints_total_limit": 5,
    "resume_from_checkpoint": "latest",
    "validation_steps": 250,
    "validation_guidance": 3.5,
    "validation_guidance_rescale": 0.0,
    "validation_num_inference_steps": 50,
    "num_eval_images": 4,
    "validation_prompt": "A photo of ohwx standing against a white background, natural lighting",
    "validation_prompt_library": "/data/config/user_prompt_library.json",
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, deformed, disfigured",
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "report_to": "tensorboard",
    "tracker_project_name": "qwen-character-lora",
    "tracker_run_name": "ohwx-training",
    "push_to_hub": false,
    "hub_model_id": "",
    "use_ema": false,
    "vae_batch_size": 1,
    "compress_disk_cache": false,
    "disable_benchmark": false,
    "skip_file_discovery": ""
}
```

- [ ] **Step 2: Write `config/multidatabackend.json`**

Dataset backend configuration with subject images + regularization data + text embed cache.

```json
[
    {
        "id": "subject-images",
        "type": "local",
        "instance_data_dir": "/data/dataset/images",
        "caption_strategy": "textfile",
        "crop": false,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 512,
        "maximum_image_size": 1328,
        "target_downsample_size": 1024,
        "cache_dir_vae": "/data/cache/vae/subject",
        "metadata_backend": "discovery",
        "repeats": 20,
        "is_regularisation_data": false,
        "disabled": false,
        "skip_file_discovery": ""
    },
    {
        "id": "regularization-images",
        "type": "local",
        "instance_data_dir": "/data/dataset/regularization/images",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a photo of a person",
        "crop": true,
        "crop_style": "center",
        "crop_aspect": "square",
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 512,
        "maximum_image_size": 1328,
        "target_downsample_size": 1024,
        "cache_dir_vae": "/data/cache/vae/regularization",
        "metadata_backend": "discovery",
        "repeats": 0,
        "is_regularisation_data": true,
        "disabled": false,
        "skip_file_discovery": ""
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/data/cache/text/qwen_image",
        "write_batch_size": 16,
        "disabled": false
    }
]
```

**Note on `caption_strategy: "textfile"`**: SimpleTuner looks for a `.txt` file with the same stem as each image. The captioning script (Task 2) writes `.txt` files directly into `/data/dataset/images/` by default, so `photo_001.jpg` and `photo_001.txt` live side by side.

- [ ] **Step 3: Write `config/user_prompt_library.json`**

Validation prompts testing identity, pose, setting, and outfit control.

```json
{
    "identity_check": "A photo of ohwx standing against a plain white background, arms at sides, looking directly at the camera, neutral expression, soft even studio lighting",
    "pose_sitting": "A photo of ohwx sitting cross-legged on a wooden floor, hands resting on knees, relaxed expression, warm indoor lighting",
    "pose_action": "A photo of ohwx running through a grassy park, mid-stride, energetic expression, bright daylight with dappled shadows",
    "setting_urban": "A photo of ohwx standing on a busy city sidewalk at night, neon signs reflected on wet pavement, wearing a dark overcoat",
    "setting_nature": "A photo of ohwx walking through a snowy pine forest at golden hour, breath visible in cold air, wearing a red winter jacket",
    "outfit_formal": "A photo of ohwx wearing a tailored navy blue suit and white dress shirt, standing in a modern office with floor-to-ceiling windows",
    "outfit_casual": "A photo of ohwx wearing a plain white t-shirt and jeans, sitting on a park bench, relaxed posture, sunny day",
    "close_up": "A close-up photo of ohwx from the shoulders up, looking slightly to the left, gentle smile, soft natural window light"
}
```

- [ ] **Step 4: Write `config/config.env`**

SimpleTuner optionally reads this file for accelerate/environment settings. It is loaded before `config.json`. The training Job's env vars in the K3s manifest also set these, but having the file ensures SimpleTuner's config discovery works cleanly.

```bash
export TRAINING_NUM_PROCESSES=1
export TRAINING_NUM_MACHINES=1
export TRAINING_DYNAMO_BACKEND=no
export MIXED_PRECISION=bf16
export SIMPLETUNER_LOG_LEVEL=INFO
```

- [ ] **Step 5: Commit**

```bash
git add config/config.json config/multidatabackend.json config/user_prompt_library.json config/config.env
git commit -m "feat: add SimpleTuner training configs for Qwen-Image character LoRA"
```

---

### Task 4: Dockerfile and .dockerignore

**Files:**
- Create: `qwen-lora-training/Dockerfile`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
# Qwen-Image-2512 LoRA Training Container
# Base: NGC PyTorch 25.11 (CUDA 13.0.2, PyTorch 2.10, ARM64 SBSA + Blackwell)
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Blackwell sm_100 kernel compilation target
ENV TORCH_CUDA_ARCH_LIST="10.0"
ENV DEBIAN_FRONTEND=noninteractive

# Install SimpleTuner with CUDA 13 support
RUN pip install --no-cache-dir 'simpletuner[cuda13]' \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Install minimal dependencies for captioning script
# (uses only stdlib: urllib, json, base64, pathlib -- no extra deps needed)

# Copy captioning script
COPY scripts/caption.py /app/caption.py
RUN chmod +x /app/caption.py

# Copy config files
COPY config/ /app/config/

# Working directory for SimpleTuner
WORKDIR /workspace

# Default entrypoint: can be overridden per Job
ENTRYPOINT ["python"]
CMD ["/app/caption.py"]
```

- [ ] **Step 2: Create `.dockerignore`**

```
.git
k8s/
GUIDE.md
docs/
*.md
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add Dockerfile based on NGC PyTorch 25.11 + SimpleTuner"
```

---

### Task 5: K3s Manifests

**Files:**
- Create: `qwen-lora-training/k8s/pvc.yaml`
- Create: `qwen-lora-training/k8s/caption-job.yaml`
- Create: `qwen-lora-training/k8s/train-job.yaml`

- [ ] **Step 1: Write `k8s/pvc.yaml`**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: lora-training-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-path
  resources:
    requests:
      storage: 130Gi
```

- [ ] **Step 2: Write `k8s/caption-job.yaml`**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: lora-caption
  namespace: default
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: caption
          image: YOUR_REGISTRY/qwen-lora-training:latest  # <-- replace
          command: ["python", "/app/caption.py"]
          env:
            - name: VLM_API_URL
              value: "http://YOUR_VLLM_SERVICE:8000/v1"   # <-- replace
            # - name: VLM_MODEL_NAME
            #   value: ""  # auto-detect if unset
            - name: TRIGGER_WORD
              value: "ohwx"
            - name: IMAGE_DIR
              value: "/data/dataset/images"
            - name: CAPTION_DIR
              value: "/data/dataset/images"
          volumeMounts:
            - name: training-data
              mountPath: /data
          resources:
            requests:
              memory: "1Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
      volumes:
        - name: training-data
          persistentVolumeClaim:
            claimName: lora-training-pvc
```

- [ ] **Step 3: Write `k8s/train-job.yaml`**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: lora-train
  namespace: default
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: OnFailure
      # Copy default configs from image to PVC if not already present.
      # User can override by pre-populating /data/config/ on the PVC.
      initContainers:
        - name: init-config
          image: YOUR_REGISTRY/qwen-lora-training:latest  # <-- replace
          command:
            - sh
            - -c
            - |
              mkdir -p /data/config
              for f in config.json multidatabackend.json user_prompt_library.json config.env; do
                if [ ! -f "/data/config/$f" ]; then
                  echo "Copying default $f to /data/config/"
                  cp "/app/config/$f" "/data/config/$f"
                else
                  echo "Using existing /data/config/$f"
                fi
              done
          volumeMounts:
            - name: training-data
              mountPath: /data
      containers:
        - name: train
          image: YOUR_REGISTRY/qwen-lora-training:latest  # <-- replace
          command:
            - simpletuner
            - train
          # SimpleTuner discovers config at ./config/config.json relative to workingDir
          workingDir: /data
          env:
            - name: TRAINING_NUM_PROCESSES
              value: "1"
            - name: TRAINING_NUM_MACHINES
              value: "1"
            - name: TRAINING_DYNAMO_BACKEND
              value: "no"
            - name: MIXED_PRECISION
              value: "bf16"
            - name: SIMPLETUNER_LOG_LEVEL
              value: "INFO"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"
            # Uncomment if using HuggingFace Hub for model download:
            # - name: HF_TOKEN
            #   valueFrom:
            #     secretKeyRef:
            #       name: hf-token
            #       key: token
          volumeMounts:
            - name: training-data
              mountPath: /data
          resources:
            requests:
              memory: "120Gi"
              nvidia.com/gpu: "1"
            limits:
              memory: "128Gi"
              nvidia.com/gpu: "1"
      volumes:
        - name: training-data
          persistentVolumeClaim:
            claimName: lora-training-pvc
```

- [ ] **Step 4: Commit**

```bash
git add k8s/
git commit -m "feat: add K3s manifests for PVC, captioning Job, and training Job"
```

---

### Task 6: Usage Guide

**Files:**
- Create: `qwen-lora-training/GUIDE.md`

- [ ] **Step 1: Write `GUIDE.md`**

```markdown
# Qwen-Image-2512 Character LoRA Training Guide

Train a character-specific LoRA on Qwen-Image-2512 for photorealistic full-body
generation with text-prompt control over pose, setting, and outfit.

## Prerequisites

- K3s cluster with NVIDIA GPU node (GB10 / Blackwell)
- NVIDIA device plugin and GPU drivers installed
- Container registry accessible from the cluster
- Self-hosted vLLM endpoint with a vision-language model

## 1. Build the Container Image

```bash
# Clone this repo
git clone <this-repo-url>
cd qwen-lora-training

# Build for ARM64 (run on the GX10 or use buildx)
docker build -t your-registry/qwen-lora-training:latest .
docker push your-registry/qwen-lora-training:latest
```

Update the `image:` field in both `k8s/caption-job.yaml` and `k8s/train-job.yaml`.

## 2. Create the PVC

```bash
kubectl apply -f k8s/pvc.yaml
```

## 3. Populate the PVC

Use a temporary pod or direct host access to populate the PVC:

```bash
# Start a helper pod
kubectl run pvc-helper --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"lora-training-pvc"}}]}}'

kubectl wait --for=condition=Ready pod/pvc-helper

# Create directory structure
kubectl exec pvc-helper -- mkdir -p \
  /data/models \
  /data/dataset/images \
  /data/dataset/regularization/images \
  /data/config \
  /data/output \
  /data/cache

# Copy your photos
kubectl cp /path/to/your/photos/ pvc-helper:/data/dataset/images/

# Copy config files (optional -- the train Job's initContainer copies defaults
# from the image if these don't exist on the PVC, but you can override here)
kubectl cp config/config.json pvc-helper:/data/config/config.json
kubectl cp config/config.env pvc-helper:/data/config/config.env
kubectl cp config/multidatabackend.json pvc-helper:/data/config/multidatabackend.json
kubectl cp config/user_prompt_library.json pvc-helper:/data/config/user_prompt_library.json
```

### Download the base model

```bash
kubectl exec pvc-helper -- sh -c '
  pip install huggingface_hub[cli] &&
  huggingface-cli download Qwen/Qwen-Image-2512 \
    --local-dir /data/models/Qwen-Image-2512
'
```

### Download regularization images

```bash
kubectl exec pvc-helper -- sh -c '
  pip install huggingface_hub[cli] &&
  huggingface-cli download bghira/pseudo-camera-10k \
    --repo-type dataset \
    --local-dir /data/dataset/regularization/images
'
```

### Clean up helper

```bash
kubectl delete pod pvc-helper
```

## 4. Run Captioning

Make sure your vLLM pod is running, then update `VLM_API_URL` in
`k8s/caption-job.yaml` and apply:

```bash
kubectl apply -f k8s/caption-job.yaml
kubectl logs -f job/lora-caption
```

Wait for completion. Review a sample of generated captions:

```bash
kubectl run pvc-helper --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox","command":["sleep","300"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"lora-training-pvc"}}]}}'

kubectl exec pvc-helper -- ls /data/dataset/images/*.txt
kubectl exec pvc-helper -- cat /data/dataset/images/photo_001.txt
kubectl delete pod pvc-helper
```

Verify:
- Every image has a corresponding `.txt` caption
- Captions begin with "A photo of ohwx"
- Physical descriptions are accurate
- Pose/clothing/setting vary between images

Edit any captions that need correction before proceeding.

## 5. Run Training

**Scale down vLLM first** to free GPU memory:

```bash
kubectl scale deployment <your-vllm-deployment> --replicas=0
```

Then start training:

```bash
kubectl apply -f k8s/train-job.yaml
kubectl logs -f job/lora-train
```

Training takes approximately 2-4 hours for 3000 steps with 40-80 images.

Monitor validation samples in `/data/output/` (check every ~250 steps).

## 6. Retrieve Results

After training completes:

```bash
kubectl scale deployment <your-vllm-deployment> --replicas=1

kubectl run pvc-helper --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox","command":["sleep","300"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"lora-training-pvc"}}]}}'

# Copy the final LoRA weights
kubectl cp pvc-helper:/data/output/ ./output/

kubectl delete pod pvc-helper
```

The LoRA weights are in `output/` as safetensors files.

## 7. Inference

### With diffusers

```python
from diffusers import QwenImagePipeline
import torch

pipe = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16
).to("cuda")

pipe.load_lora_weights("./output/")

image = pipe(
    prompt="A photo of ohwx wearing a leather jacket, standing on a rooftop at sunset",
    negative_prompt="ugly, blurry, deformed, low quality",
    num_inference_steps=50,
    true_cfg_scale=3.5,
    width=1024,
    height=1024,
).images[0]

image.save("result.png")
```

## Dataset Guidelines

### Training photos (40-80 images)

| Category | Count | Description |
|----------|-------|-------------|
| Face closeups | 10-15 | Head and shoulders, various angles |
| Upper body | 10-15 | Waist up, various poses |
| Full body | 15-25 | Head to toe, standing/sitting/walking |
| Various angles | 5-10 | Profile, 3/4, back |

**Quality rules:**
- Minimum 512x512, ideally 1024+ longest edge
- Diverse lighting (indoor, outdoor, overcast, direct sun)
- Diverse backgrounds (prevents background leaking into identity)
- Diverse clothing (prevents outfit baking into identity)
- No other people in frame
- No heavy filters, watermarks, or text overlays

### Tuning Parameters

If results are unsatisfactory, try:

| Issue | Adjustment |
|-------|------------|
| Character looks different each time | Increase `lora_rank` (try 96 or 128), add more face closeups |
| Character ignores pose prompts | Add more regularization images, reduce `repeats` to 15 |
| Training diverges / artifacts | Reduce `learning_rate` to 1e-5, reduce `train_batch_size` to 2 |
| Overfitting (identical outputs) | Reduce `max_train_steps`, increase caption diversity |
| Background leaks into identity | Ensure training photos have diverse backgrounds |
```

- [ ] **Step 2: Commit**

```bash
git add GUIDE.md
git commit -m "docs: add step-by-step usage guide"
```

---

### Task 7: Final Review and Tag

- [ ] **Step 1: Review all files are present**

```bash
find . -type f -not -path './.git/*' | sort
```

Expected output:
```
./.dockerignore
./Dockerfile
./GUIDE.md
./config/config.env
./config/config.json
./config/multidatabackend.json
./config/user_prompt_library.json
./k8s/caption-job.yaml
./k8s/pvc.yaml
./k8s/train-job.yaml
./scripts/caption.py
```

- [ ] **Step 2: Verify Dockerfile syntax**

```bash
docker build --check .
```

Or just verify it parses:
```bash
cat Dockerfile
```

- [ ] **Step 3: Verify JSON configs are valid**

```bash
python3 -c "import json; json.load(open('config/config.json'))"
python3 -c "import json; json.load(open('config/multidatabackend.json'))"
python3 -c "import json; json.load(open('config/user_prompt_library.json'))"
```

Expected: no output (no errors).

- [ ] **Step 4: Verify YAML manifests are valid**

```bash
kubectl apply --dry-run=client -f k8s/pvc.yaml
kubectl apply --dry-run=client -f k8s/caption-job.yaml
kubectl apply --dry-run=client -f k8s/train-job.yaml
```

Expected: `configured (dry run)` for each.

- [ ] **Step 5: Tag release**

```bash
git tag v0.1.0
git log --oneline
```
