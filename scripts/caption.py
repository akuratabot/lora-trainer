#!/usr/bin/env python3
"""
Auto-caption images for LoRA training via an OpenAI-compatible VLM API.

Run locally to generate .txt caption files alongside your training images.
Captions use a structured format that separates identity (trigger word)
from variable attributes (pose, clothing, setting).

Requires: pip install openai

Usage:
  python scripts/caption.py --api-url http://localhost:8000/v1 --image-dir ./dataset/images
  python scripts/caption.py --api-url http://localhost:8000/v1 --image-dir ./dataset/images --trigger ohwx

Environment variables (fallbacks for CLI args):
  VLM_API_URL    - Base URL of the vLLM endpoint
  VLM_MODEL_NAME - Model name for the API (default: auto-detect)
  TRIGGER_WORD   - Trigger word for the subject (default: ohwx)
"""

import argparse
import base64
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)

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


MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}


def encode_image_url(image_path: Path) -> str:
    """Encode image as a base64 data URL."""
    mime = MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def detect_model_name(client: OpenAI) -> str:
    """Auto-detect the first available model from the API."""
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception as e:
        print(f"Warning: could not auto-detect model: {e}")
    return "default"


def caption_image(
    client: OpenAI,
    model_name: str,
    image_path: Path,
    trigger_word: str,
    no_think: bool,
) -> str:
    """Send an image to the VLM API and return the caption."""
    image_url = encode_image_url(image_path)

    # For reasoning models, set a generous budget so thinking doesn't consume
    # all tokens. For non-reasoning models, max_completion_tokens works the
    # same as max_tokens.
    kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.replace("{trigger_word}", trigger_word),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": f"Describe this person in detail. Begin with 'A photo of {trigger_word}'.",
                    },
                ],
            },
        ],
        "max_completion_tokens": 2048,
        "temperature": 0.3,
    }

    # If --no-think is set, ask the API to suppress reasoning output.
    # vLLM supports this via chat_template_kwargs; other providers may use
    # different mechanisms (e.g. reasoning_effort, thinking parameter).
    if no_think:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    content = choice.message.content

    # If content is still null (reasoning model consumed everything, or refusal),
    # check if there's usable text in the reasoning field.
    if content is None:
        # Try to access reasoning content if available
        reasoning = getattr(choice.message, "reasoning", None) or getattr(
            choice.message, "reasoning_content", None
        )
        if reasoning:
            raise ValueError(
                f"Model produced reasoning but no final content. "
                f"The model may need --no-think, or a higher token budget. "
                f"finish_reason={choice.finish_reason}"
            )
        raise ValueError(
            f"API returned null content. finish_reason={choice.finish_reason}"
        )

    return content.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-caption images for LoRA training via a VLM API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --api-url http://localhost:8000/v1 --image-dir ./photos
  %(prog)s --api-url http://my-server:8000/v1 --image-dir ./photos --trigger sks
  %(prog)s --api-url http://my-server:8000/v1 --image-dir ./photos --no-think
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
        "--no-think",
        action="store_true",
        help="Disable thinking/reasoning for reasoning models (e.g. QwQ, Qwen3). "
        "Passes enable_thinking=false via chat_template_kwargs.",
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
        print("Run with --help for usage information.", file=sys.stderr)
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

    # Create OpenAI client pointed at the vLLM endpoint
    client = OpenAI(base_url=args.api_url, api_key="not-needed")

    # Auto-detect model name if not set
    model_name = args.model
    if not model_name:
        model_name = detect_model_name(client)

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Trigger word: {args.trigger}")
    print(f"API: {args.api_url}")
    print(f"Model: {model_name}")
    if args.no_think:
        print("Thinking: disabled")
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
            caption = caption_image(client, model_name, img_path, args.trigger, args.no_think)
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
