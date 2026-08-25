import os
import json

from pipeline.utils import format_time, get_cardinal_direction, resolve_severity, ensure_detailed_suggested_action, limit_text_words


class JsonWriter:

    def __init__(self, output_folder="outputs"):

        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    def write(self, persons):

        for idx, (person_id, person_data) in enumerate(persons.items()):

            assigned_class = person_data.get("class", "person")

            # -------------------------------------------------
            # Create Person Folder
            # -------------------------------------------------

            person_folder = os.path.join(
                self.output_folder,
                f"detection_{person_id}"
            )

            os.makedirs(person_folder, exist_ok=True)

            reid_count = person_data.get("reidentification_count", 0)
            sev_level, sev_label = resolve_severity(person_data.get("severity"), reid_count)
            suggest_act = ensure_detailed_suggested_action(
                person_data.get("suggest_action", ""),
                sev_label,
                str(person_id),
                person_data.get("total_duration", 0),
                reid_count
            )

            # -------------------------------------------------
            # Convert all appearances
            # -------------------------------------------------

            appearances = []

            for appearance in person_data["appearances"]:

                appearances.append({

                    "appearance_id":
                        appearance["appearance_id"],

                    "entry_time":
                        format_time(
                            appearance["entry_time"]
                        ),

                    "exit_time":
                        format_time(
                            appearance["exit_time"]
                        ),

                    "duration":
                        round(
                            appearance["duration"],
                            2
                        ),

                    # Severity is decided by VLM / resolved fallback
                    "severity_level":
                        sev_level,

                    "severity":
                        sev_label,

                    # ---------------------------------------
                    # GPS -- present only if telemetry was
                    # available for this video; otherwise null
                    # ---------------------------------------

                    "entry_location": {
                        "lat": appearance.get("entry_lat"),
                        "lon": appearance.get("entry_lon"),
                        "alt": appearance.get("entry_alt"),
                        "distance": appearance.get("entry_distance"),
                        "heading": appearance.get("entry_heading"),
                        "direction": get_cardinal_direction(appearance.get("entry_heading")),
                        "detection_time": appearance.get("entry_detection_time"),
                    },

                    "exit_location": {
                        "lat": appearance.get("exit_lat"),
                        "lon": appearance.get("exit_lon"),
                        "alt": appearance.get("exit_alt"),
                        "distance": appearance.get("exit_distance"),
                        "heading": appearance.get("exit_heading"),
                        "direction": get_cardinal_direction(appearance.get("exit_heading")),
                        "detection_time": appearance.get("exit_detection_time"),
                    },

                })

            # -------------------------------------------------
            # Final JSON
            # -------------------------------------------------

            metadata = {

                "id_number":
                    str(person_data["person_id"]),

                # <-- USE THIS LINE FOR DUMMY DATA (Different class for each different ID)
                "class":
                    assigned_class,

                # <-- USE THIS LINE FOR ORIGINAL REAL DATA (Uncomment line below & comment line above)
                # "class":
                #     person_data.get("class", "person"),

                "total_duration":
                    round(
                        person_data["total_duration"],
                        2
                    ),

                "reidentification_count":
                    reid_count,

                "severity_level":
                    sev_level,

                "severity":
                    sev_label,

                "description":
                    limit_text_words(person_data.get("description", ""), 30),

                "suggest_action":
                    suggest_act,

                "appearances":
                    appearances

            }

            # -------------------------------------------------
            # Write JSON
            # -------------------------------------------------

            json_path = os.path.join(
                person_folder,
                "metadata.json"
            )

            with open(json_path, "w") as file:

                json.dump(
                    metadata,
                    file,
                    indent=4
                )

        print("\nJSON files created successfully.")