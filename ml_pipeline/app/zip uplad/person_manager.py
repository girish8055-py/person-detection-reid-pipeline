"""
person_manager.py

Maintains the lifecycle of every detected person.

Stores:
- Permanent Person ID
- Multiple appearances
- Entry / Exit time
- Total duration
"""
print("NEW PERSON MANAGER LOADED")
from typing import Dict


class PersonManager:

    def __init__(self):

        # person_id -> metadata
        self.persons: Dict[int, dict] = {}

        # track_id -> person_id
        

    def update(self, detections, current_time):

        active_persons = set()

        updated_detections = []

        for det in detections:
            person_id = det["person_id"]

            det["person_id"] = person_id

            updated_detections.append(det)

            active_persons.add(person_id)

            # -------------------------------------------------
            # First time seeing this person
            # -------------------------------------------------

            if person_id not in self.persons:

                self.persons[person_id] = {

                    "person_id": person_id,

                    "active": True,

                    "current_entry": current_time,

                    "last_seen": current_time,

                    "total_duration": 0,

                    "reidentification_count": 0,

                    "appearances": []

                }

            else:

                person = self.persons[person_id]

                # Person re-entered
                if not person["active"]:

                    person["active"] = True

                    person["current_entry"] = current_time

                    person["reidentification_count"] += 1

                person["last_seen"] = current_time

        # -------------------------------------------------
        # Close appearance if person disappeared
        # -------------------------------------------------

        for person_id, person in self.persons.items():

            if person["active"] and person_id not in active_persons:

                entry = person["current_entry"]

                exit_time = person["last_seen"]

                duration = round(exit_time - entry, 2)

                person["appearances"].append({

                    "appearance_id": len(person["appearances"]) + 1,

                    "entry_time": entry,

                    "exit_time": exit_time,

                    "duration": duration

                })

                person["total_duration"] += duration

                person["active"] = False

        return updated_detections

    def finalize(self, final_time):

        for person in self.persons.values():

            if person["active"]:

                entry = person["current_entry"]

                duration = round(final_time - entry, 2)

                person["appearances"].append({

                    "appearance_id": len(person["appearances"]) + 1,

                    "entry_time": entry,

                    "exit_time": final_time,

                    "duration": duration

                })

                person["total_duration"] += duration

                person["active"] = False

    def get_persons(self):

        return self.persons

