"""
vlm_verifier.py

After YOLO detection + keyframe extraction, sends entry/middle/exit frames
to the local VLM (LLaVA via Ollama) to independently verify whether each
tracked object is a person or vehicle.  This corrects YOLO misclassifications
(e.g. person labeled as vehicle) before the final result is built.

The VLM receives ONLY the cropped frame images with no class hint, so it
must decide purely from visual content.
"""

import base64
import io
import json
import logging
import os
from typing import List, Optional

from PIL import Image
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")


def _encode_image(image_path: str, max_dim: int = 512) -> Optional[str]:
    """Read an image file, resize to stay within VLM context budget, and return base64 encoding."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_dim, max_dim))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.error("Failed to encode image %s: %s", image_path, exc)
        return None


def verify_object_class(
    keyframe_paths: List[str],
    target_id: str,
    fallback_class: str = "person",
    allowed_classes: Optional[List[str]] = None,
) -> str:
    """
    Ask the VLM to look at one or more keyframes of a tracked object and
    decide its class, strictly constrained to the model's supported classes.
    """
    # Prefer close-up target crop if available, supplemented by representative keyframe
    candidate_images = []
    if keyframe_paths:
        target_dir = os.path.dirname(keyframe_paths[0])
        for pref in ["best_detection.jpg", "middle_detection.jpg", "first_detection.jpg"]:
            cand = os.path.join(target_dir, pref)
            if os.path.exists(cand) and cand not in candidate_images:
                candidate_images.append(cand)

    for kp in keyframe_paths:
        if kp not in candidate_images:
            candidate_images.append(kp)
        if len(candidate_images) >= 2:
            break

    images_b64: List[str] = []
    for kp in candidate_images[:2]:
        enc = _encode_image(kp, max_dim=512)
        if enc:
            images_b64.append(enc)

    if not images_b64:
        logger.warning(
            "[VLM VERIFY] Target %s: no usable keyframes, keeping '%s'",
            target_id,
            fallback_class,
        )
        return fallback_class

    clean_fallback = str(fallback_class).lower().strip() if fallback_class else "person"

    # Build exact allowed set from model classes or default
    if allowed_classes:
        valid_set = set(str(c).lower().strip() for c in allowed_classes)
    else:
        valid_set = {"person", "vehicle", "drone"}

    if clean_fallback:
        valid_set.add(clean_fallback)

    valid_list_str = ", ".join([f'"{c}"' for c in sorted(valid_set)])

    prompt = (
        "You are an aerial surveillance image classifier and verifier.\n"
        "Look closely at the keyframe image(s) below showing a detected subject from an aerial video.\n\n"
        "YOUR CRITICAL MISSION:\n"
        f"Verify the visual classification of the object (detector labeled it as '{clean_fallback}').\n"
        f"You MUST classify the object into EXACTLY ONE of the model's supported classes: [{valid_list_str}].\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "  - Examine the visual shape and features of the target object in the frame.\n"
        "  - Base your classification strictly on the true physical shape of the subject.\n"
        f"  - Do NOT invent outside classes. Your response MUST be one of: [{valid_list_str}].\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"classification": "<one_of_the_supported_classes>"}'
    )

    payload = {
        "model": "qwen2.5vl:7b",
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}

        cls = str(parsed.get("classification", "")).lower().strip()

        if cls in valid_set:
            if cls != clean_fallback:
                logger.info(
                    "[VLM VERIFY] Target %s: class corrected '%s' -> '%s'",
                    target_id,
                    clean_fallback,
                    cls,
                )
                print(
                    f"[VLM VERIFY] Target {target_id}: "
                    f"class corrected '{clean_fallback}' -> '{cls}'"
                )
            else:
                logger.info(
                    "[VLM VERIFY] Target %s: class confirmed '%s'",
                    target_id,
                    cls,
                )
            return cls

        logger.warning(
            "[VLM VERIFY] Target %s: VLM returned unrecognised class %r (allowed: %s), keeping '%s'",
            target_id,
            cls,
            valid_list_str,
            clean_fallback,
        )
        return clean_fallback

    except Exception as exc:
        logger.error(
            "[VLM VERIFY] Target %s: VLM call failed (%s), keeping '%s'",
            target_id,
            exc,
            fallback_class,
        )
        print(
            f"[VLM VERIFY] Target {target_id}: VLM unavailable "
            f"({exc}), keeping '{fallback_class}'"
        )
        return fallback_class
