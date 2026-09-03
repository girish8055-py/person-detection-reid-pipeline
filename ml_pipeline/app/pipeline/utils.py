def format_time(seconds):
    """
    Convert seconds to HH:MM:SS
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02}:{minutes:02}:{secs:02}"


def get_cardinal_direction(yaw):
    """
    Convert yaw degrees (0-360) to a cardinal direction (N, NE, E, etc.)
    """
    if yaw is None:
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((yaw + 22.5) % 360 / 45)
    return directions[idx]


def get_severity_level(reid_count):
    """
    Map a detected person's re-identification count to a severity level.

        Level 1 (LOW)     -> re-ID count  1-3
        Level 2 (MEDIUM)  -> re-ID count  3-6
        Level 3 (HIGH)    -> re-ID count  6-8
        Level 4 (CRITICAL)-> re-ID count  8-10+ (e.g. 10 re-IDs => level 4)

    Returns:
        (severity_level:int, severity_label:str)
    """
    if reid_count is None:
        reid_count = 0

    if reid_count <= 3:
        return 1, "LOW"
    elif reid_count <= 6:
        return 2, "MEDIUM"
    elif reid_count <= 8:
        return 3, "HIGH"
    else:
        return 4, "CRITICAL"


_SEVERITY_LABELS = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}


def get_severity_from_label(label):
    """
    Normalize a free-form severity label coming from the VLM
    (e.g. "LOW", "Medium", "3", "critical") into the standard
    (severity_level:int, severity_label:str) tuple.

    Returns None when the label cannot be interpreted.
    """
    if label is None:
        return None

    norm = str(label).strip().lower()
    if not norm:
        return None

    # Substring matching handles messy VLM output like
    # "HIGH THREAT", "medium-high", "Severity: Critical", "level 3 - high".
    level = None
    if "critical" in norm or "extreme" in norm:
        level = 4
    elif "high" in norm or "severe" in norm:
        level = 3
    elif "med" in norm or "moderate" in norm:  # matches medium/med/moderate
        level = 2
    elif "low" in norm or "minimal" in norm:
        level = 1

    if level is not None:
        return level, _SEVERITY_LABELS[level]

    # Allow numeric severities like "2" or "2.0"
    try:
        level = int(float(norm))
    except ValueError:
        return None

    if level < 1:
        level = 1
    elif level > 4:
        level = 4
    return level, _SEVERITY_LABELS[level]


def resolve_severity(vlm_severity, reid_count=0):
    """
    Resolves a severity tuple (severity_level: int, severity_label: str).
    Tries to parse vlm_severity first; if invalid/None/empty, falls back to
    get_severity_level(reid_count), ensuring severity is NEVER null.
    """
    sev = get_severity_from_label(vlm_severity)
    if sev is not None:
        return sev
    return get_severity_level(reid_count)


import re


def clean_vlm_description(text: str) -> str:
    """
    Strips out any accidental VLM meta-references to 'red box', 'bounding box',
    'cropped region', 'red outline', etc. from description text.
    """
    if not text:
        return ""
    cleaned = str(text)
    patterns = [
        r'\b(?:inside|in|within|highlighted by|surrounded by|bounded by)\s+(?:the|a)\s+(?:red\s+)?(?:bounding\s+)?box\b,?\s*',
        r'\b(?:the\s+)?red\s+(?:bounding\s+)?box\s+(?:shows|contains|highlights|indicates|marks)\s*(?:a|an|the)?\s*',
        r'\s*(?:inside|in|within)\s+(?:the|a)\s+(?:red\s+)?(?:bounding\s+)?box\.?\s*',
        r'\s*(?:inside|in|within)\s+(?:the|a)\s+(?:red\s+)?(?:outline|crop|cropped\s+region)\.?\s*',
        r'\b(?:the\s+)?red\s+(?:bounding\s+)?box\b,?\s*',
        r'\b(?:the\s+)?bounding\s+box\b,?\s*',
        r'\b(?:the\s+)?red\s+outline\b,?\s*',
        r'\b(?:in|inside)\s+the\s+cropped\s+image\b,?\s*',
        r'\b(?:the\s+)?cropped\s+image\b,?\s*',
    ]
    for pat in patterns:
        cleaned = re.sub(pat, ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'^(?:,\s*|;\s*|\.\s*)+', '', cleaned)
    cleaned = re.sub(r'^(?:a|an|the)\s+(a|an|the)\s+', r'\1 ', cleaned, flags=re.IGNORECASE)
    if cleaned and not cleaned[0].isupper():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def limit_text_words(text, max_words=30):
    """
    Ensures text is at most max_words long. If longer, truncates to max_words
    and formats cleanly with a period.
    Strips out any meta-references to 'red box' or 'bounding box'.
    """
    if not text:
        return ""
    text = clean_vlm_description(text)
    words = str(text).strip().split()
    if len(words) <= max_words:
        return str(text).strip()
    truncated = " ".join(words[:max_words]).rstrip(",;-—:")
    if not truncated.endswith("."):
        truncated += "."
    return truncated


def ensure_detailed_suggested_action(act, severity_label="LOW", target_id="target", duration=0, reid_count=0, obj_class=None):
    """
    Ensures suggested_action is never just a single word (e.g. 'MONITOR', 'TRACK', 'ALERT', 'DETAIN').
    If act is a single word or brief phrase, expands it dynamically based on the situation context.
    Varies based on target class (vehicle vs person), duration, re-identification count, and severity.
    Enforces a strict 30-word limit on the output.
    """
    if not act:
        act = "MONITOR"

    clean_act = str(act).strip()
    words = clean_act.split()

    # If it's already a full sentence (> 3 words), cap at 30 words and return
    if len(words) > 3:
        return limit_text_words(clean_act, 30)

    first_word = words[0].strip(" -—:,").upper()
    sev_upper = str(severity_label).upper() if severity_label else "LOW"

    # Context-aware target description
    target_type = "vehicle" if ("vehicle" in str(target_id).lower() or (obj_class and "vehicle" in str(obj_class).lower())) else "target"

    if first_word in ("MONITOR", "TRACK", "ALERT", "DETAIN"):
        verb = first_word
    elif sev_upper == "CRITICAL":
        verb = "DETAIN"
    elif sev_upper == "HIGH":
        verb = "ALERT"
    elif sev_upper == "MEDIUM":
        verb = "TRACK"
    else:
        verb = "MONITOR"

    if target_type == "vehicle":
        if verb == "DETAIN" or sev_upper == "CRITICAL":
            res = f"DETAIN — High-threat vehicle {target_id} identified. Deploy perimeter barricades and QRF team to intercept vehicle immediately."
        elif verb == "ALERT" or sev_upper == "HIGH":
            res = f"ALERT — Vehicle {target_id} loitering in sector for {int(duration)}s with {reid_count} re-appearances. Dispatch mobile patrol to inspect."
        elif verb == "TRACK" or sev_upper == "MEDIUM":
            res = f"TRACK — Vehicle {target_id} moving through operational sector. Maintain continuous sensor tracking and log route."
        else:
            res = f"MONITOR — Standard vehicle {target_id} passage ({int(duration)}s). Log keyframes and maintain routine visual observation."
    else:
        if verb == "DETAIN" or sev_upper == "CRITICAL":
            res = f"DETAIN — Hostile target {target_id} identified in sector. Dispatch armed reaction force to challenge and apprehend immediately."
        elif verb == "ALERT" or sev_upper == "HIGH":
            res = f"ALERT — Target {target_id} re-identified {reid_count} times over {int(duration)}s. Alert perimeter guards and maintain tight tracking."
        elif verb == "TRACK" or sev_upper == "MEDIUM":
            res = f"TRACK — Target {target_id} exhibits persistent movement in sector. Maintain automated sensor tracking and record appearance log."
        else:
            res = f"MONITOR — Routine target {target_id} observed for {int(duration)}s. Maintain passive surveillance and log movement."

    return limit_text_words(res, 30)