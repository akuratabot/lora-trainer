#!/usr/bin/env python3
"""
Auto-caption images for LoRA training via an OpenAI-compatible VLM API.

Run locally to generate .txt caption files alongside your training images.
Captions use a structured format that separates identity (trigger word)
from variable attributes (pose, clothing, setting).

Usage:
  python scripts/caption.py --api-url http://localhost:8000/v1 --image-dir ./dataset/images
  python scripts/caption.py --api-url http://localhost:8000/v1 --image-dir ./dataset/images --trigger ohwx

Environment variables (fallbacks for CLI args):
  VLM_API_URL    - Base URL of the vLLM endpoint
  VLM_MODEL_NAME - Model name for the API (default: auto-detect)
  TRIGGER_WORD   - Trigger word for the subject (default: ohwx)

No dependencies beyond Python 3.8+ stdlib.
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-caption images for LoRA training via a VLM API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --api-url http://localhost:8000/v1 --image-dir ./photos
  %(prog)s --api-url http://my-server:8000/v1 --image-dir ./photos --trigger sks
  %(prog)s --api-url http://my-server:8000/v1 --image-dir ./photos --force""",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("VLM_API_URL", ""),
        help="Base URL of the vLLM endpoint (e.g. http://localhost:8000/v1). "
        "Also reads VLM_API_URL env var.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLM_MODEL_NAME", ""),
        help="Model name for the API request. Auto-detected if not set. "
        "Also reads VLM_MODEL_NAME env var.",
    )
    parser.add_argument(
        "--trigger",
        default=os.environ.get("TRIGGER_WORD", "ohwx"),
        help="Trigger word for the subject (default: ohwx). "
        "Also reads TRIGGER_WORD env var.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(os.environ.get("IMAGE_DIR", "./dataset/images")),
        help="Path to input images (default: ./dataset/images). "
        "Captions are written as .txt files in the same directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing caption files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_url:
        print("ERROR: --api-url is required (or set VLM_API_URL env var)", file=sys.stderr)
        parser_help = "Run with --help for usage information."
        print(parser_help, file=sys.stderr)
        sys.exit(1)

    image_dir = args.image_dir.resolve()

    if not image_dir.exists():
        print(f"ERROR: Image directory does not exist: {image_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect images
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"ERROR: No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Trigger word: {args.trigger}")
    print(f"API: {args.api_url}")

    # Auto-detect model name if not set
    model_name = args.model
    if not model_name:
        model_name = detect_model_name(args.api_url)
        print(f"Auto-detected model: {model_name}")
    else:
        print(f"Model: {model_name}")

    print()

    # Caption each image -- writes .txt alongside the image
    success = 0
    skipped = 0
    failed = 0
    for i, img_path in enumerate(images, 1):
        caption_path = img_path.with_suffix(".txt")

        # Skip if caption already exists (unless --force)
        if caption_path.exists() and not args.force:
            print(f"[{i}/{len(images)}] SKIP (exists): {img_path.name}")
            skipped += 1
            continue

        try:
            print(f"[{i}/{len(images)}] Captioning: {img_path.name} ... ", end="", flush=True)
            caption = caption_image(args.api_url, model_name, img_path, args.trigger)
            caption_path.write_text(caption, encoding="utf-8")
            print(f"OK ({len(caption)} chars)")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone: {success} captioned, {skipped} skipped, {failed} failed, out of {len(images)} total")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
