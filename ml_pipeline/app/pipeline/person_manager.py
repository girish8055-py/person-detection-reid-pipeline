"""
person_manager.py

Maintains the lifecycle of every detected person.

Stores:
- Permanent Person ID
- Multiple appearances
- Entry / Exit time
- Total duration
- Entry / Exit GPS position (lat, lon, alt) and real-world detection time,
  when telemetry is available for this video.

GPS fields are OPTIONAL: if a detection doesn't have "lat"/"lon" set (no
telemetry loaded for this video), everything works exactly as before --
those fields just come back None.
"""


from typing import Dict


class PersonManager:

    def __init__(self):

        # person_id -> metadata
        self.persons: Dict[int, dict] = {}

    def update(self, detections, current_time):

        active_persons = set()

        updated_detections = []

        for det in detections:
            person_id = det["person_id"]

            det["person_id"] = person_id

            updated_detections.append(det)

            active_persons.add(person_id)

            lat = det.get("lat")
            lon = det.get("lon")
            alt = det.get("alt")
            distance = det.get("distance")
            heading = det.get("heading")
            detection_time = det.get("detection_time")  # ISO string, if telemetry available
            detected_class = det.get("class", "person")
            confidence = det.get("confidence")
            bbox = det.get("bbox")
            frame_num = det.get("frame_num")

            # -------------------------------------------------
            # First time seeing this person
            # -------------------------------------------------

            if person_id not in self.persons:

                self.persons[person_id] = {

                    "person_id": person_id,

                    "class": detected_class,

                    "active": True,

                    "current_entry": current_time,

                    "last_seen": current_time,

                    "total_duration": 0,

                    "reidentification_count": 0,

                    "appearances": [],

                    # GPS at the moment this appearance started
                    "entry_lat": lat,
                    "entry_lon": lon,
                    "entry_alt": alt,
                    "entry_distance": distance,
                    "entry_heading": heading,
                    "entry_detection_time": detection_time,
                    "entry_confidence": confidence,
                    "entry_bbox": bbox,
                    "start_frame": frame_num,
                }

            else:

                person = self.persons[person_id]

                # Person re-entered
                if not person["active"]:

                    person["active"] = True

                    person["current_entry"] = current_time

                    person["reidentification_count"] += 1

                    # New appearance starting -> reset entry GPS & info
                    person["entry_lat"] = lat
                    person["entry_lon"] = lon
                    person["entry_alt"] = alt
                    person["entry_distance"] = distance
                    person["entry_heading"] = heading
                    person["entry_detection_time"] = detection_time
                    person["entry_confidence"] = confidence
                    person["entry_bbox"] = bbox
                    person["start_frame"] = frame_num

                # Update class if a specific object class (e.g. drone, vehicle) is detected over default person
                if detected_class != "person" and person.get("class") == "person":
                    person["class"] = detected_class

                person["last_seen"] = current_time

            # Always keep the latest position & metadata
            self.persons[person_id]["last_lat"] = lat
            self.persons[person_id]["last_lon"] = lon
            self.persons[person_id]["last_alt"] = alt
            self.persons[person_id]["last_distance"] = distance
            self.persons[person_id]["last_heading"] = heading
            self.persons[person_id]["last_detection_time"] = detection_time
            self.persons[person_id]["last_confidence"] = confidence
            self.persons[person_id]["last_bbox"] = bbox
            self.persons[person_id]["end_frame"] = frame_num

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

                    "duration": duration,

                    "confidence": person.get("entry_confidence") or person.get("last_confidence"),

                    "bbox": person.get("entry_bbox") or person.get("last_bbox"),

                    "start_frame": person.get("start_frame"),

                    "end_frame": person.get("end_frame"),

                    "entry_lat": person.get("entry_lat"),
                    "entry_lon": person.get("entry_lon"),
                    "entry_alt": person.get("entry_alt"),
                    "entry_distance": person.get("entry_distance"),
                    "entry_heading": person.get("entry_heading"),
                    "entry_detection_time": person.get("entry_detection_time"),

                    "exit_lat": person.get("last_lat"),
                    "exit_lon": person.get("last_lon"),
                    "exit_alt": person.get("last_alt"),
                    "exit_distance": person.get("last_distance"),
                    "exit_heading": person.get("last_heading"),
                    "exit_detection_time": person.get("last_detection_time"),

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

                    "duration": duration,

                    "confidence": person.get("entry_confidence") or person.get("last_confidence"),

                    "bbox": person.get("entry_bbox") or person.get("last_bbox"),

                    "start_frame": person.get("start_frame"),

                    "end_frame": person.get("end_frame"),

                    "entry_lat": person.get("entry_lat"),
                    "entry_lon": person.get("entry_lon"),
                    "entry_alt": person.get("entry_alt"),
                    "entry_distance": person.get("entry_distance"),
                    "entry_heading": person.get("entry_heading"),
                    "entry_detection_time": person.get("entry_detection_time"),

                    "exit_lat": person.get("last_lat"),
                    "exit_lon": person.get("last_lon"),
                    "exit_alt": person.get("last_alt"),
                    "exit_distance": person.get("last_distance"),
                    "exit_heading": person.get("last_heading"),
                    "exit_detection_time": person.get("last_detection_time"),

                })

                person["total_duration"] += duration

                person["active"] = False

    def get_persons(self):

        return self.persons