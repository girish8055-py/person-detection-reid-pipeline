import os
import json

from pipeline.utils import format_time


class JsonWriter:

    def __init__(self, output_folder="outputs"):

        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    def write(self, persons):

        # <-- FOR DUMMY DATA: Dummy classes list (vehicle, person, vehicle)
        dummy_classes = ["vehicle", "person", "vehicle"]

        for idx, (person_id, person_data) in enumerate(persons.items()):

            # DUMMY CLASS TOGGLE: Cycle through dummy classes
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
                        )

                })

            # -------------------------------------------------
            # Final JSON
            # -------------------------------------------------

            metadata = {

                "id_number":
                    str(person_data["person_id"]),

                # <-- USE THIS LINE FOR DUMMY DATA (vehicle, person, vehicle)
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