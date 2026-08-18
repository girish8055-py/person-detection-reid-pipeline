import os
import json

from pipeline.utils import format_time, get_cardinal_direction


class JsonWriter:

    def __init__(self, output_folder="outputs"):

        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    def write(self, persons):

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

        for idx, (person_id, person_data) in enumerate(expanded_persons.items()):

            # DUMMY CLASS TOGGLE: Cycle through dummy classes per ID
            assigned_class = dummy_classes[idx % len(dummy_classes)]

            # -------------------------------------------------
            # Create Person Folder
            # -------------------------------------------------

            person_folder = os.path.join(
                self.output_folder,
                f"detection_{person_id}"
            )

            os.makedirs(person_folder, exist_ok=True)

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
                    person_data["reidentification_count"],

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