import os
import sys

# Ensure python-docx is installed
try:
    import docx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    import docx

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_element(name):
    return OxmlElement(name)

def create_attribute(element, name, value):
    element.set(qn(name), value)

def add_page_number(run):
    fldChar1 = create_element('w:fldChar')
    create_attribute(fldChar1, 'w:fldCharType', 'begin')
    instrText = create_element('w:instrText')
    create_attribute(instrText, 'xml:space', 'preserve')
    instrText.text = "PAGE"
    fldChar2 = create_element('w:fldChar')
    create_attribute(fldChar2, 'w:fldCharType', 'separate')
    fldChar3 = create_element('w:fldChar')
    create_attribute(fldChar3, 'w:fldCharType', 'end')
    
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def add_total_pages(run):
    fldChar1 = create_element('w:fldChar')
    create_attribute(fldChar1, 'w:fldCharType', 'begin')
    instrText = create_element('w:instrText')
    create_attribute(instrText, 'xml:space', 'preserve')
    instrText.text = "NUMPAGES"
    fldChar2 = create_element('w:fldChar')
    create_attribute(fldChar2, 'w:fldCharType', 'separate')
    fldChar3 = create_element('w:fldChar')
    create_attribute(fldChar3, 'w:fldCharType', 'end')
    
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def build_docx(output_path):
    doc = Document()
    
    # Configure page size and margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Configure Header & Footer
        section.different_first_page_header_footer = True
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("AeroGCS IDMS — Threat Detection API Documentation")
        hrun.font.name = 'Calibri'
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(128, 128, 128)
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun1 = fp.add_run("Page ")
        frun1.font.name = 'Calibri'
        frun1.font.size = Pt(9.5)
        frun1.font.color.rgb = RGBColor(128, 128, 128)
        
        add_page_number(frun1)
        
        frun2 = fp.add_run(" of ")
        frun2.font.name = 'Calibri'
        frun2.font.size = Pt(9.5)
        frun2.font.color.rgb = RGBColor(128, 128, 128)
        
        add_total_pages(frun2)

    # Styling defaults
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(51, 51, 51)
    
    # Title Page
    p_title_space = doc.add_paragraph()
    p_title_space.paragraph_format.space_before = Pt(72)
    
    p_subtitle = doc.add_paragraph()
    p_subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    srun = p_subtitle.add_run("AeroGCS IDMS")
    srun.bold = True
    srun.font.size = Pt(16)
    srun.font.color.rgb = RGBColor(28, 115, 232)
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    trun = p_title.add_run("AI Threat Detection & Threat Management")
    trun.bold = True
    trun.font.size = Pt(26)
    trun.font.color.rgb = RGBColor(30, 41, 59)
    p_title.paragraph_format.space_after = Pt(6)
    
    p_title2 = doc.add_paragraph()
    p_title2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t2run = p_title2.add_run("Backend API & Workflow Documentation")
    t2run.font.size = Pt(16)
    t2run.font.color.rgb = RGBColor(100, 116, 139)
    p_title2.paragraph_format.space_after = Pt(48)
    
    p_desc = doc.add_paragraph()
    p_desc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    drun = p_desc.add_run(
        "Covers: flight video capture → detection task creation → Python AI service polling → "
        "object detection → automatic threat creation → threat lifecycle management"
    )
    drun.font.italic = True
    drun.font.size = Pt(11)
    drun.font.color.rgb = RGBColor(71, 85, 105)
    p_desc.paragraph_format.space_after = Pt(144)
    
    p_ver = doc.add_paragraph()
    p_ver.alignment = WD_ALIGN_PARAGRAPH.CENTER
    vrun = p_ver.add_run("Version 1.0 • 2026-07-22")
    vrun.font.size = Pt(10)
    vrun.font.color.rgb = RGBColor(148, 163, 184)
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 1
    # ----------------------------------------------------
    h1 = doc.add_heading("1. System Overview", level=1)
    h1.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(12)
    
    p = doc.add_paragraph(
        "This module automates threat detection on recorded drone flight videos. When a flight ends and its "
        "video is uploaded to Azure Blob Storage, a detection task is created and queued. A separate Python AI "
        "service polls for pending tasks, downloads the video, runs object detection, uploads evidence images, "
        "and reports results back. The backend then converts each detection result into a Threat record with "
        "evidence and a full status history, which operators can view, comment on, assign, and manage through "
        "the Threats APIs."
    )
    p.paragraph_format.space_after = Pt(12)
    
    h1_1 = doc.add_heading("1.1 End-to-End Flow", level=2)
    h1_1.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h1_1.paragraph_format.space_before = Pt(12)
    h1_1.paragraph_format.space_after = Pt(6)
    
    steps = [
        ("Drone flight ends and the recorded video finishes uploading to Azure Blob Storage under the user's container (DroneStreams/{flightId}/{streamkey}.mp4).", False),
        ("The client calls the Stream Key Lookup API to confirm the upload. This API records the final file size, generates a short-lived read SAS URL for the video and a short-lived write SAS URL (upload container) for evidence images, and creates a Detection Task with status PENDING.", False),
        ("The Python AI/object-detection service periodically calls GET /AeroMegh/idms/detection/pending to fetch all tasks currently in PENDING status, ordered by newest first.", False),
        ("The Python service downloads the video using the task's video_url, runs object/person detection, tracking, Re-ID, and telemetry interpolation, and uploads any evidence images (detected frames/snapshots) directly to Azure Blob Storage using the task's upload_url (SAS token).", True),
        ("The Python service reports results back by calling PUT /AeroMegh/idms/detection/update/:taskid with the new status (COMPLETED or FAILED), an array of detection results (containing coordinates, distance, heading, and cardinal direction), and an optional error_message.", True),
        ("For every object in the results array, the backend creates a new Threat record (status NEW, source_type AI_DETECTION), links it to the flight/drone/detection task, stores its evidence image, and writes an initial entry into the threat status history — all inside a single database transaction.", True),
        ("The detection task row is updated with its final status and error_message (if any). Raw AI results are not persisted on the task; they only exist as the Threats created from them.", False),
        ("Operators then use the Threats APIs to list, filter, review, comment on, assign, update, or delete the generated threats.", False),
    ]
    
    for i, (step, is_updated) in enumerate(steps, 1):
        sp = doc.add_paragraph()
        sp.paragraph_format.left_indent = Inches(0.25)
        sp.paragraph_format.space_after = Pt(6)
        r_num = sp.add_run(f"{i}. ")
        r_num.bold = True
        r_text = sp.add_run(step)
        if is_updated:
            r_text.font.color.rgb = RGBColor(16, 124, 65)  # Highlight updated logic in green
            
    # Tables for status values
    h1_2 = doc.add_heading("1.2 Detection Task Status Values", level=2)
    h1_2.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h1_2.paragraph_format.space_before = Pt(12)
    h1_2.paragraph_format.space_after = Pt(6)
    
    t1 = doc.add_table(rows=4, cols=2)
    t1.autofit = True
    headers1 = ["Status", "Meaning"]
    for j, text in enumerate(headers1):
        cell = t1.cell(0, j)
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, "1E293B")
        
    t1_data = [
        ("PENDING", "Task created after video upload; waiting to be picked up by the AI service."),
        ("COMPLETED", "AI service finished processing and reported results (threats created if any objects were detected)."),
        ("FAILED", "AI service could not process the video; error_message explains the failure.")
    ]
    for row_idx, (status, meaning) in enumerate(t1_data, 1):
        t1.cell(row_idx, 0).text = status
        t1.cell(row_idx, 0).paragraphs[0].runs[0].bold = True
        t1.cell(row_idx, 1).text = meaning
        set_cell_background(t1.cell(row_idx, 0), "F1F5F9")
        set_cell_background(t1.cell(row_idx, 1), "FFFFFF")

    h1_3 = doc.add_heading("1.3 Threat Status Values", level=2)
    h1_3.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h1_3.paragraph_format.space_before = Pt(12)
    h1_3.paragraph_format.space_after = Pt(6)
    
    t2 = doc.add_table(rows=4, cols=2)
    t2.autofit = True
    headers2 = ["Status", "Typical Meaning"]
    for j, text in enumerate(headers2):
        cell = t2.cell(0, j)
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, "1E293B")
        
    t2_data = [
        ("NEW", "Default status when a threat is first created (manually or by AI detection)."),
        ("IN_PROGRESS / REVIEW", "Assigned and being investigated by a user or team (project-specific values)."),
        ("RESOLVED / CLOSED", "Investigation complete; no further action needed.")
    ]
    for row_idx, (status, meaning) in enumerate(t2_data, 1):
        t2.cell(row_idx, 0).text = status
        t2.cell(row_idx, 0).paragraphs[0].runs[0].bold = True
        t2.cell(row_idx, 1).text = meaning
        set_cell_background(t2.cell(row_idx, 0), "F1F5F9")
        set_cell_background(t2.cell(row_idx, 1), "FFFFFF")
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 2
    # ----------------------------------------------------
    h2 = doc.add_heading("2. Node.js Backend ↔ Python AI Service Interaction Diagram", level=1)
    h2.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(12)
    
    p2 = doc.add_paragraph("Sequence of calls between the Drone/Client app, the Node.js backend, Azure Blob Storage, and the Python AI detection service:")
    p2.paragraph_format.space_after = Pt(12)
    
    # Text-based sequence diagram
    diag_text = (
        "1. Video finishes uploading -> Client app calls GET /flight/stream/:streamKey/:fileSize\n"
        "2. Node.js backend verifies blob size, generates SAS read/write tokens\n"
        "3. Node.js backend creates PENDING task in 'threat-detection'\n"
        "4. Python AI service polls GET /detection/pending on an interval\n"
        "5. Python AI service downloads video via read SAS video_url\n"
        "6. Python AI service runs YOLO detection + ByteTrack + Re-ID + Telemetry interpolation\n"
        "7. Python AI service uploads evidence images to Azure via SAS write upload_url\n"
        "8. Python AI service calls PUT /detection/update/:taskid (marks task COMPLETED, sends results array)\n"
        "9. Node.js backend creates Threat records (status NEW, source_type AI_DETECTION) and audits status history"
    )
    dp = doc.add_paragraph()
    drun = dp.add_run(diag_text)
    drun.font.name = 'Consolas'
    drun.font.size = Pt(9.5)
    drun.font.color.rgb = RGBColor(51, 65, 85)
    dp.paragraph_format.left_indent = Inches(0.25)
    dp.paragraph_format.space_after = Pt(12)
    
    h2_1 = doc.add_heading("2.1 Phase Summary", level=2)
    h2_1.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h2_1.paragraph_format.space_before = Pt(12)
    
    phases = [
        ("Phase A. Flight ends & video upload", "Video lands in Blob Storage; Stream Key Lookup confirms the upload, mints SAS URLs, and inserts a PENDING detection task."),
        ("Phase B. AI service polls for pending work", "The Python service repeatedly calls GET /detection/pending on its own schedule until it receives tasks to process."),
        ("Phase C. Detection & evidence capture", "The Python service downloads the video with the SAS video_url, runs detection/re-identification, and uploads evidence images with the SAS upload_url."),
        ("Phase D. Report results & auto-create threats", "The Python service calls PUT /detection/update/:taskid; the backend creates a Threat (with evidence and status history) per result and marks the task COMPLETED or FAILED."),
    ]
    for name, desc in phases:
        ph_p = doc.add_paragraph()
        ph_p.paragraph_format.left_indent = Inches(0.2)
        ph_p.paragraph_format.space_after = Pt(6)
        r_name = ph_p.add_run(f"{name}: ")
        r_name.bold = True
        ph_p.add_run(desc)

    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 3
    # ----------------------------------------------------
    h3 = doc.add_heading("3. Referenced Tables", level=1)
    h3.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(12)
    
    t3 = doc.add_table(rows=8, cols=2)
    t3.autofit = True
    headers3 = ["Table", "Purpose"]
    for j, text in enumerate(headers3):
        cell = t3.cell(0, j)
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, "1E293B")
        
    t3_data = [
        ("flights", "Flight records; streamkey, blob_container, size, status, end_time, drone_id."),
        ("threat-detection", "One row per detection task: flight_id, user_id, video_url, upload_url, status, error_message, timestamps."),
        ("threats", "Core threat records: threat_code, title, description, priority, severity, status, assignment, location, links to flight/drone/detection task."),
        ("threat_evidence_images", "Evidence image(s) attached to a threat (image_url, file_name, file_size, uploaded_by/at)."),
        ("threat_status_history", "Audit trail of every status change on a threat (old_status → new_status, changed_by, reason, changed_at)."),
        ("threat_comments", "Free-text comments/notes added to a threat by users."),
        ("threat_assignments", "History of assignment changes (assigned_user_id / assigned_team_id, assigned_by, note, assigned_at).")
    ]
    for row_idx, (tbl, purp) in enumerate(t3_data, 1):
        t3.cell(row_idx, 0).text = tbl
        t3.cell(row_idx, 0).paragraphs[0].runs[0].bold = True
        t3.cell(row_idx, 1).text = purp
        set_cell_background(t3.cell(row_idx, 0), "F1F5F9")
        set_cell_background(t3.cell(row_idx, 1), "FFFFFF")

    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 4
    # ----------------------------------------------------
    h4 = doc.add_heading("4. API Reference", level=1)
    h4.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h4.paragraph_format.space_before = Pt(12)
    h4.paragraph_format.space_after = Pt(12)
    
    p_api = doc.add_paragraph("Base path: /AeroMegh/idms — all endpoints require a valid Bearer token unless noted otherwise.")
    p_api.paragraph_format.space_after = Pt(12)
    
    h4_1 = doc.add_heading("4.1 Flight Upload & Detection Task Creation", level=2)
    h4_1.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h4_1.paragraph_format.space_before = Pt(12)
    
    # 4.1.1
    h4_1_1 = doc.add_heading("4.1.1 Stream Key Lookup (creates the Detection Task)", level=3)
    h4_1_1.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    
    def add_endpoint_table(doc, method, endpoint, auth, purpose):
        t = doc.add_table(rows=4, cols=2)
        t.cell(0, 0).text = "Method"
        t.cell(0, 1).text = method
        t.cell(1, 0).text = "Endpoint"
        t.cell(1, 1).text = endpoint
        t.cell(2, 0).text = "Auth"
        t.cell(2, 1).text = auth
        t.cell(3, 0).text = "Purpose"
        t.cell(3, 1).text = purpose
        for row in t.rows:
            row.cells[0].paragraphs[0].runs[0].bold = True
            set_cell_background(row.cells[0], "F1F5F9")
            set_cell_background(row.cells[1], "FFFFFF")
            
    add_endpoint_table(
        doc,
        "GET",
        "/AeroMegh/idms/flight/stream/:streamKey/:fileSize",
        "Bearer token (headers.authorization)",
        "Called once the flight's recorded video has finished uploading to Blob Storage. Confirms the flight by streamkey, records the final video size, updates storage usage, generates SAS URLs, and triggers creation of a PENDING detection task for the AI service."
    )
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 4.1.2 Get Pending Tasks
    h4_1_2 = doc.add_heading("4.1.2 Get Pending Detection Tasks", level=3)
    h4_1_2.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    
    add_endpoint_table(
        doc,
        "GET",
        "/AeroMegh/idms/detection/pending",
        "Bearer token (headers.authorization)",
        "Polled by the Python AI service to retrieve all detection tasks currently awaiting processing."
    )
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # 4.1.3 Update Tasks
    h4_1_3 = doc.add_heading("4.1.3 Update Detection Task (reports AI results, auto-creates Threats)", level=3)
    h4_1_3.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    
    add_endpoint_table(
        doc,
        "PUT",
        "/AeroMegh/idms/detection/update/:taskid",
        "Bearer token (headers.authorization)",
        "Called by the Python AI service once processing finishes, to mark the task COMPLETED or FAILED and report detected objects. Every entry in results is converted into a new Threat with evidence, inside a single transaction. Includes approx distance, heading angle, and cardinal direction values."
    )
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Example JSON update schema
    p_ex_title = doc.add_paragraph()
    p_ex_title.add_run("Example Request Payload:").bold = True
    
    ex_json = (
        "{\n"
        '  "status": "COMPLETED",\n'
        '  "results": [\n'
        "    {\n"
        '      "person_id": "Person_1",\n'
        '      "total_duration": 14.50,\n'
        '      "reidentification_count": 1,\n'
        '      "appearances": [\n'
        "        {\n"
        '          "appearance_id": 1,\n'
        '          "entry_time": "00:00:00",\n'
        '          "exit_time": "00:00:08",\n'
        '          "duration": 8.0,\n'
        '          "confidence": 0.88,\n'
        '          "entry_location": {\n'
        '            "lat": 19.9451248,\n'
        '            "lon": 73.6006139,\n'
        '            "alt": 45.2,\n'
        '            "distance": 8.45,\n'
        '            "heading": 82.77,\n'
        '            "direction": "NE"\n'
        "          },\n"
        '          "exit_location": {\n'
        '            "lat": 19.9451251,\n'
        '            "lon": 73.6006142,\n'
        '            "alt": 45.2,\n'
        '            "distance": 8.50,\n'
        '            "heading": 82.79,\n'
        '            "direction": "NE"\n'
        "          }\n"
        "        }\n"
        "      ],\n"
        '      "evidence": {\n'
        '        "file_url": "https://pdrlimage.blob.core.windows.net/.../Person_1_sec_10.jpg",\n'
        '        "file_name": "Person_1_sec_10",\n'
        '        "file_size": 234011\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    
    ep = doc.add_paragraph()
    erun = ep.add_run(ex_json)
    erun.font.name = 'Consolas'
    erun.font.size = Pt(9.0)
    erun.font.color.rgb = RGBColor(16, 124, 65)
    ep.paragraph_format.left_indent = Inches(0.25)
    ep.paragraph_format.space_after = Pt(12)
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # SECTION 5: Route Summary
    # ----------------------------------------------------
    h5 = doc.add_heading("5. Route Summary", level=1)
    h5.runs[0].font.color.rgb = RGBColor(30, 41, 59)
    h5.paragraph_format.space_before = Pt(12)
    h5.paragraph_format.space_after = Pt(12)
    
    t5 = doc.add_table(rows=11, cols=3)
    t5.autofit = True
    headers5 = ["Method", "Route", "Short Description"]
    for j, text in enumerate(headers5):
        cell = t5.cell(0, j)
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, "1E293B")
        
    t5_data = [
        ("GET", "/AeroMegh/idms/flight/stream/:streamKey/:fileSize", "Confirms uploaded flight video and triggers the PENDING task."),
        ("GET", "/AeroMegh/idms/detection/pending", "Returns all pending detection tasks for the AI service."),
        ("PUT", "/AeroMegh/idms/detection/update/:taskid", "Reports AI results (includes distance and direction) and auto-creates Threats."),
        ("POST", "/AeroMegh/idms/threats/new", "Manually creates a new threat."),
        ("GET", "/AeroMegh/idms/threats", "List/filter/paginate threats."),
        ("GET", "/AeroMegh/idms/threats/:id", "Get full details (evidence, comments, history) of a threat."),
        ("DELETE", "/AeroMegh/idms/threats/delete/:id", "Permanently deletes a threat."),
        ("PUT", "/AeroMegh/idms/threats/update/:id", "Updates threat fields and audits status changes."),
        ("POST", "/AeroMegh/idms/threats/comment/:id", "Adds a comment to a threat."),
        ("POST", "/AeroMegh/idms/threats/assign/:id", "Assigns/reassigns a threat to a user or team.")
    ]
    for row_idx, (method, route, desc) in enumerate(t5_data, 1):
        t5.cell(row_idx, 0).text = method
        t5.cell(row_idx, 0).paragraphs[0].runs[0].bold = True
        t5.cell(row_idx, 1).text = route
        t5.cell(row_idx, 1).paragraphs[0].runs[0].font.name = 'Consolas'
        t5.cell(row_idx, 1).paragraphs[0].runs[0].font.size = Pt(9.5)
        t5.cell(row_idx, 2).text = desc
        set_cell_background(t5.cell(row_idx, 0), "F1F5F9")
        set_cell_background(t5.cell(row_idx, 1), "FFFFFF")
        set_cell_background(t5.cell(row_idx, 2), "FFFFFF")

    doc.save(output_path)
    print(f"Docx generated successfully at {output_path}")

if __name__ == "__main__":
    out = os.path.join("d:\\deployment_model", "AeroGCS_IDMS_Threat_Detection_API_Documentation.docx")
    build_docx(out)
