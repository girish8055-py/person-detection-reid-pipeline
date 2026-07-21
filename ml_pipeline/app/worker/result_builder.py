from pipeline.utils import format_time


def build_result_json(persons):
    """
    Convert process_video() output into the format required by the
    Detection Update API.
    """

    results = []

    for _, person in persons.items():

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
                    "alt": app.get("entry_alt")
                },

                "exit_location": {
                    "lat": app.get("exit_lat"),
                    "lon": app.get("exit_lon"),
                    "alt": app.get("exit_alt")
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

        results.append({

            "person_id": f"Person_{person['person_id']}" if not str(person['person_id']).startswith("Person_") else person['person_id'],

            "total_duration": round(
                person["total_duration"],
                2
            ),

            "reidentification_count":
                person["reidentification_count"],

            "evidence": {
                "file_name": person.get("file_name"),
                "file_size": person.get("file_size"),
                "file_url": person.get("file_url")
            },

            "appearances": appearances

        })

    return {

        "status": "COMPLETED",

        "results": results

    }