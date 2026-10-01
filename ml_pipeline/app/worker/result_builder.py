from pipeline.utils import format_time, get_cardinal_direction, resolve_severity, ensure_detailed_suggested_action, limit_text_words

def safe_format_iso_time(iso_str, fallback_seconds):
    if iso_str and "T" in str(iso_str):
        try:
            return str(iso_str).split("T")[1].split(".")[0][:8]
        except Exception:
            pass
    return format_time(fallback_seconds)

def build_result_json(persons):
    """
    Convert process_video() output into the format required by the
    Detection Update API.
    """

    results = []

    expanded_persons = dict(persons)

    for idx, (p_id, person) in enumerate(expanded_persons.items()):
        # Ensure duplicated dummy entries carry a valid distinct filename/url if base had one
        person_file_name = person.get("file_name")
        person_file_url = person.get("file_url")
        if person_file_name and idx > 0 and str(p_id) not in person_file_name:
            person_file_name = f"Person_{p_id}_detection.jpg"
            if person_file_url:
                base_part = person_file_url.rsplit("/", 1)[0]
                person_file_url = f"{base_part}/{person_file_name}"

        appearances = []

        reid_cnt = person.get("reidentification_count", 0)
        severity_level, severity_label = resolve_severity(person.get("severity"), reid_cnt)
        suggest_act = ensure_detailed_suggested_action(
            person.get("suggest_action", ""),
            severity_label,
            str(p_id),
            person.get("total_duration", 0),
            reid_cnt
        )

        for app in person["appearances"]:

            appearances.append({

                "appearance_id": app["appearance_id"],

                "entry_time": safe_format_iso_time(app.get("entry_detection_time"), app["entry_time"]),

                "exit_time": safe_format_iso_time(app.get("exit_detection_time"), app["exit_time"]),

                "duration": round(app["duration"], 2),

                "confidence": round(app.get("confidence"), 2) if app.get("confidence") is not None else None,

                # Severity is decided by VLM / resolved fallback (never null)
                "severity_level": severity_level,

                "severity": severity_label,

                "entry_location": {
                    "lat": app.get("entry_lat"),
                    "lon": app.get("entry_lon"),
                    "alt": app.get("entry_alt"),
                    "distance": app.get("entry_distance"),
                    "heading": app.get("entry_heading"),
                    "direction": get_cardinal_direction(app.get("entry_heading"))
                },

                "exit_location": {
                    "lat": app.get("exit_lat"),
                    "lon": app.get("exit_lon"),
                    "alt": app.get("exit_alt"),
                    "distance": app.get("exit_distance"),
                    "heading": app.get("exit_heading"),
                    "direction": get_cardinal_direction(app.get("exit_heading"))
                },

                "bounding_box": {
                    "x": round(app["bbox"][0], 1) if app.get("bbox") and len(app["bbox"]) >= 4 else None,
                    "y": round(app["bbox"][1], 1) if app.get("bbox") and len(app["bbox"]) >= 4 else None,
                    "width": round(app["bbox"][2] - app["bbox"][0], 1) if app.get("bbox") and len(app["bbox"]) >= 4 else None,
                    "height": round(app["bbox"][3] - app["bbox"][1], 1) if app.get("bbox") and len(app["bbox"]) >= 4 else None
                },

                "frame_range": {
                    "start": app.get("start_frame"),
                    "end": app.get("end_frame")
                }

            })

        # Determine overall entry and exit time for this object from its appearances
        if person.get("appearances"):
            first_app = person["appearances"][0]
            last_app = person["appearances"][-1]
            obj_entry_time = safe_format_iso_time(first_app.get("entry_detection_time"), first_app["entry_time"])
            obj_exit_time = safe_format_iso_time(last_app.get("exit_detection_time"), last_app["exit_time"])
        else:
            obj_entry_time = format_time(person.get("current_entry", 0))
            obj_exit_time = format_time(person.get("last_seen", 0))

        assigned_class = person.get("class", "person")

        results.append({

            "id_number": str(p_id),

            "class": assigned_class,

            "entry_time": obj_entry_time,

            "exit_time": obj_exit_time,

            "total_duration": round(
                person["total_duration"],
                2
            ),

            "reidentification_count":
                reid_cnt,

            "severity_level": severity_level,

            "severity": severity_label,

            "description": limit_text_words(person.get("description", ""), 30),

            "suggest_action": suggest_act,

            "evidence": person.get("evidence") if isinstance(person.get("evidence"), list) else (
                [person.get("evidence")] if isinstance(person.get("evidence"), str) else (
                    [person_file_url] if person_file_url else []
                )
            ),

            "appearances": appearances

        })

    return {

        "status": "COMPLETED",

        "results": results

    }