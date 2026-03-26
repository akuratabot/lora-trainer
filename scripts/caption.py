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
  CAPTION_DIR    - Path to write captions (default: /data/dataset/images)
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
