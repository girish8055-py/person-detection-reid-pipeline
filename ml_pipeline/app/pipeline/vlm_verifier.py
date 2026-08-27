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
import json
import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


def _encode_image(image_path: str) -> Optional[str]:
    """Read an image file and return its base64 encoding."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logger.error("Failed to encode image %s: %s", image_path, exc)
        return None


def verify_object_class(
    keyframe_paths: List[str],
    target_id: str,
    fallback_class: str = "person",
) -> str:
    """
    Ask the VLM to look at one or more keyframes of a tracked object and
    decide whether it is a **person** or a **vehicle**.

    Args:
        keyframe_paths: list of absolute paths to keyframe JPGs (entry,
            middle, exit).  At least one must exist.
        target_id: human-readable ID used only for logging.
        fallback_class: returned when every image fails to load or the VLM
            is unreachable.

    Returns:
        ``"person"`` or ``"vehicle"``
    """
    images_b64: List[str] = []
    for kp in keyframe_paths:
        enc = _encode_image(kp)
        if enc:
            images_b64.append(enc)

    if not images_b64:
        logger.warning(
            "[VLM VERIFY] Target %s: no usable keyframes, keeping '%s'",
            target_id,
            fallback_class,
        )
        return fallback_class

    prompt = (
        "You are an aerial drone surveillance image classifier and verifier.\n"
        "Look closely at the image(s) below. Each image shows a cropped region "
        "highlighted inside a red bounding box from an aerial video.\n\n"
        "YOUR CRITICAL MISSION:\n"
        "Verify whether the object inside the red box is ACTUALLY a person or a vehicle.\n"
        "YOLO object detectors frequently misclassify objects (e.g. detecting a vehicle/car/truck as a person, or detecting a person as a vehicle).\n"
        "You must examine the visual shape and features inside the red box to correct any false detection:\n"
        '  - "person": human being, walking, standing, sitting, soldier, infantryman, head/shoulders from above.\n'
        '  - "vehicle": car, truck, van, bus, motorcycle, bicycle, tractor, tank, or any wheeled/motorized machinery.\n\n'
        "CRITICAL INSTRUCTIONS:\n"
        '  - If the detector labeled it as a person but visually it is a vehicle/car/truck, you MUST classify it as "vehicle".\n'
        '  - If the detector labeled it as a vehicle but visually it is a person/human, you MUST classify it as "person".\n'
        "  - Base your classification strictly on the true physical shape inside the red bounding box.\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"classification": "person"} OR {"classification": "vehicle"}'
    )

    payload = {
        "model": "llava:latest",
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
        if cls in ("person", "vehicle"):
            if cls != fallback_class:
                logger.info(
                    "[VLM VERIFY] Target %s: class corrected '%s' -> '%s'",
                    target_id,
                    fallback_class,
                    cls,
                )
                print(
                    f"[VLM VERIFY] Target {target_id}: "
                    f"class corrected '{fallback_class}' -> '{cls}'"
                )
            else:
                logger.info(
                    "[VLM VERIFY] Target %s: class confirmed '%s'",
                    target_id,
                    cls,
                )
            return cls

        logger.warning(
            "[VLM VERIFY] Target %s: VLM returned unrecognised class %r, "
            "keeping '%s'",
            target_id,
            cls,
            fallback_class,
        )
        return fallback_class

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
