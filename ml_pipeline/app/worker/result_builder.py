from pipeline.utils import format_time, get_cardinal_direction


def build_result_json(persons):
    """
    Convert process_video() output into the format required by the
    Detection Update API.
    """

    results = []

    # <-- FOR DUMMY DATA: List of distinct classes for different IDs
    dummy_classes = ["person", "vehicle", "car", "truck", "motorbike"]

    # Generate at least 3 distinct objects with different IDs if fewer detections exist
    expanded_persons = dict(persons)
    if len(expanded_persons) < 3 and expanded_persons:
        base_id = list(expanded_persons.keys())[0]
        base_data = expanded_persons[base_id]
        for extra_id in range(1, 4):
            if extra_id not in expanded_persons:
                import copy
                new_person = copy.deepcopy(base_data)
                new_person["person_id"] = extra_id
                expanded_persons[extra_id] = new_person

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

        for app in person["appearances"]:

            appearances.append({

                "appearance_id": app["appearance_id"],

                "entry_time": format_time(app["entry_time"]),

                "exit_time": format_time(app["exit_time"]),

                "duration": round(app["duration"], 2),

                "confidence": app.get("confidence"),

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

                # Add these later when PersonManager stores them
                "bounding_box": {
                    "x": None,
                    "y": None,
                    "width": None,
                    "height": None
                },

                "frame_range": {
                    "start": None,
                    "end": None
                }

            })

        # Assign a distinct dummy class for each distinct ID
        assigned_class = dummy_classes[idx % len(dummy_classes)]

        results.append({

            "id_number": str(p_id),

            # <-- USE THIS LINE FOR DUMMY DATA (Different class for each different ID)
            "class": assigned_class,

            # <-- USE THIS LINE FOR ORIGINAL REAL DATA (Uncomment line below & comment line above)
            # "class": person.get("class", "person"),

            "total_duration": round(
                person["total_duration"],
                2
            ),

            "reidentification_count":
                person["reidentification_count"],

            "evidence": {
                "file_name": person_file_name,
                "file_size": person.get("file_size"),
                "file_url": person_file_url
            },

            "appearances": appearances

        })

    return {

        "status": "COMPLETED",

        "results": results

    }