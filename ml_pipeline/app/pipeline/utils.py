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


def ensure_detailed_suggested_action(act, severity_label="LOW", target_id="target", duration=0, reid_count=0):
    """
    Ensures suggested_action is never just a single word (e.g. 'monitor', 'track', 'alert', 'detain').
    If act is a single word or brief phrase, expands it into a complete tactical recommendation.
    """
    if not act:
        act = "MONITOR"

    clean_act = str(act).strip()

    # If it's already a full sentence/description (> 3 words), return it as is
    words = clean_act.split()
    if len(words) > 3:
        return clean_act

    first_word = words[0].strip(" -—:,").upper()

    action_templates = {
        "MONITOR": f"MONITOR — Maintain passive surveillance on Target {target_id} and log movement patterns across area.",
        "TRACK": f"TRACK — Actively track Target {target_id} with drone sensors; observe for loitering or hostile activity.",
        "ALERT": f"ALERT — Raise force readiness level and notify perimeter security units regarding Target {target_id}.",
        "DETAIN": f"DETAIN — Dispatch Quick Reaction Force (QRF) to intercept and challenge Target {target_id} immediately."
    }

    if first_word in action_templates:
        return action_templates[first_word]

    # Fallback template based on severity label
    sev_upper = str(severity_label).upper() if severity_label else "LOW"
    if sev_upper == "CRITICAL":
        return f"DETAIN — Critical threat level assessed for Target {target_id}. Dispatch reaction force to intercept immediately."
    elif sev_upper == "HIGH":
        return f"ALERT — High priority observation on Target {target_id}. Alert field units and maintain continuous track."
    elif sev_upper == "MEDIUM":
        return f"TRACK — Target {target_id} exhibits moderate activity. Continue automated tracking and log keyframes."
    else:
        return f"MONITOR — Routine surveillance on Target {target_id}. Maintain passive observation."