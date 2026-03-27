# Qwen-Image-2512 Character LoRA Training Guide

Train a character-specific LoRA on Qwen-Image-2512 for photorealistic full-body
generation with text-prompt control over pose, setting, and outfit.

## Prerequisites

- Python 3.8+ with `openai` package (`pip install openai`)
- Access to an OpenAI-compatible VLM API (e.g. self-hosted vLLM)
- K3s cluster with NVIDIA GPU node (GB10 / Blackwell)
- NVIDIA device plugin and GPU drivers installed
- Container registry accessible from the cluster

## Overview

The workflow has two phases:

1. **Local** -- Prepare your dataset: curate photos, auto-caption them, review/edit captions
2. **Cluster** -- Push dataset to PVC, run SimpleTuner training as a K3s Job

---

## Phase 1: Local Dataset Preparation

### 1.1 Collect Training Photos (40-80 images)

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

Put all photos in a local directory (e.g. `./dataset/images/`).

### 1.2 Auto-Caption Images

The captioning script calls your VLM API to generate a `.txt` caption file
alongside each image.

```bash
pip install openai  # one-time setup

python scripts/caption.py \
  --api-url http://your-vlm-server:8000/v1 \
  --image-dir ./dataset/images \
  --trigger ohwx
```

If you're using a reasoning model (QwQ, Qwen3, etc.), add `--no-think` to
disable the thinking/reasoning step and get direct captions:

```bash
python scripts/caption.py \
  --api-url http://your-vlm-server:8000/v1 \
  --image-dir ./dataset/images \
  --trigger ohwx \
  --no-think
```

Options:
- `--api-url` -- Your vLLM endpoint (required)
- `--image-dir` -- Directory containing your photos (default: `./dataset/images`)
- `--trigger` -- Trigger word for the subject (default: `ohwx`)
- `--model` -- Model name (auto-detected if not set)
- `--no-think` -- Disable reasoning for thinking models (QwQ, Qwen3)
- `--force` -- Overwrite existing captions

After captioning, each image will have a matching `.txt` file:
```
dataset/images/
  photo_001.jpg
  photo_001.txt   <-- auto-generated caption
  photo_002.jpg
  photo_002.txt
  ...
```

### 1.3 Review and Edit Captions

Open the `.txt` files and verify:
- Every caption begins with "A photo of ohwx"
- Physical descriptions are accurate and consistent across images
- Pose, clothing, and setting descriptions vary appropriately

Edit any captions that are inaccurate. This is the most important quality step.

### 1.4 Download Regularization Images

Download generic person photos to prevent the model from associating all
person-related prompts with your subject:

```bash
pip install huggingface_hub[cli]
huggingface-cli download bghira/pseudo-camera-10k \
  --repo-type dataset \
  --local-dir ./dataset/regularization/images
```

---

## Phase 2: K3s Training

### 2.1 Build the Container Image

```bash
docker build -t your-registry/qwen-lora-training:latest .
docker push your-registry/qwen-lora-training:latest
```

Update the `image:` field in `k8s/train-job.yaml`.

### 2.2 Apply K3s Resources

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
```

To edit training config, update `k8s/configmap.yaml` and re-apply. No image rebuild needed.

### 2.3 Populate the PVC

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
  /data/output \
  /data/cache

# Copy your captioned photos (images + .txt captions)
kubectl cp ./dataset/images/ pvc-helper:/data/dataset/images/

# Copy regularization images
kubectl cp ./dataset/regularization/images/ pvc-helper:/data/dataset/regularization/images/
```

### Download the base model

```bash
kubectl exec pvc-helper -- sh -c '
  pip install huggingface_hub[cli] &&
  huggingface-cli download Qwen/Qwen-Image-2512 \
    --local-dir /data/models/Qwen-Image-2512
'
```

### Clean up helper

```bash
kubectl delete pod pvc-helper
```

### 2.4 Run Training

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

### 2.5 Retrieve Results

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

---

## Inference

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

---

## Tuning Parameters

If results are unsatisfactory, edit `config/config.json` and re-run training:

| Issue | Adjustment |
|-------|------------|
| Character looks different each time | Increase `lora_rank` (try 96 or 128), add more face closeups |
| Character ignores pose prompts | Add more regularization images, reduce `repeats` to 15 |
| Training diverges / artifacts | Reduce `learning_rate` to 1e-5, reduce `train_batch_size` to 2 |
| Overfitting (identical outputs) | Reduce `max_train_steps`, increase caption diversity |
| Background leaks into identity | Ensure training photos have diverse backgrounds |
