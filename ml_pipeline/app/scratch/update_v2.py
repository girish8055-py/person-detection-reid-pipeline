import docx

# Open the master document
doc_path = "d:\\deployment_model\\Person_Detection_ReID_Pipeline_Documentation_v2.docx"
doc = docx.Document(doc_path)

# Update Paragraph 27 (under Telemetry & Spatiotemporal Fusion)
for idx, para in enumerate(doc.paragraphs):
    if "Parses paired drone telemetry CSV log files" in para.text:
        # Update paragraph text to include the new features
        para.text = (
            "Parses paired drone telemetry CSV log files containing GPS coordinates (latitude, longitude, altitude), "
            "distance metrics (obsDist, liveObstacleDistance), drone heading/yaw, and ISO timestamps. "
            "It computes the approximate distance between the drone and the person, drone heading, and cardinal direction."
        )
        print("Updated paragraph text under Telemetry Fusion.")
        break

# New JSON schema representation for Table 1
new_json = """{
  "total_persons": 1,
  "persons": {
    "Person_001": {
      "person_id": "Person_001",
      "total_duration": 14.5,
      "reidentification_count": 1,
      "entry_lat": 18.52043,
      "entry_lon": 73.85674,
      "entry_alt": 45.2,
      "entry_distance": 8.45,
      "entry_heading": 82.77,
      "entry_direction": "NE",
      "entry_detection_time": "2026-07-21T11:00:00Z",
      "exit_lat": 18.52049,
      "exit_lon": 73.85680,
      "exit_alt": 45.1,
      "exit_distance": 8.50,
      "exit_heading": 82.79,
      "exit_direction": "NE",
      "exit_detection_time": "2026-07-21T11:00:15Z",
      "file_name": "Person_001_sec_10",
      "file_size": 245120,
      "appearances": [
        {
          "entry": 0.0,
          "exit": 8.0,
          "duration": 8.0,
          "entry_location": {
            "lat": 18.52043,
            "lon": 73.85674,
            "alt": 45.2,
            "distance": 8.45,
            "heading": 82.77,
            "direction": "NE",
            "detection_time": "2026-07-21T11:00:00Z"
          },
          "exit_location": {
            "lat": 18.52046,
            "lon": 73.85677,
            "alt": 45.2,
            "distance": 8.47,
            "heading": 82.78,
            "direction": "NE",
            "detection_time": "2026-07-21T11:00:08Z"
          }
        },
        {
          "entry": 11.5,
          "exit": 18.0,
          "duration": 6.5,
          "entry_location": {
            "lat": 18.52047,
            "lon": 73.85678,
            "alt": 45.1,
            "distance": 8.49,
            "heading": 82.79,
            "direction": "NE",
            "detection_time": "2026-07-21T11:00:11Z"
          },
          "exit_location": {
            "lat": 18.52049,
            "lon": 73.85680,
            "alt": 45.1,
            "distance": 8.50,
            "heading": 82.79,
            "direction": "NE",
            "detection_time": "2026-07-21T11:00:15Z"
          }
        }
      ]
    }
  }
}"""

# Update Table 1
table = doc.tables[1]
cell = table.cell(0, 0)
cell.text = new_json

# Format font to Consolas/Pt 9.5 for the JSON code
for paragraph in cell.paragraphs:
    for run in paragraph.runs:
        run.font.name = "Consolas"
        run.font.size = docx.shared.Pt(9.5)

print("Updated Table 1 with new JSON output schema.")

# Save modifications
doc.save(doc_path)
print("Changes successfully saved to Person_Detection_ReID_Pipeline_Documentation_v2.docx")
