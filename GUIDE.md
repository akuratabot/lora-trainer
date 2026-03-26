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
cd lora-trainer

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
