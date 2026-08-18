"""
Professional DOCX Generator — AeroGCS IDMS System Documentation v1.0

Generates two enterprise-grade Word documents:
  Document 1: Detection Pipeline System Documentation
  Document 2: API Integration & Communication Documentation
"""

import os
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# ── Color Palette ───────────────────────────────────────────
NAVY       = RGBColor(0x0D, 0x1B, 0x2A)
DARK_BLUE  = RGBColor(0x1B, 0x36, 0x5D)
MID_BLUE   = RGBColor(0x2B, 0x54, 0x8A)
ACCENT     = RGBColor(0x41, 0x5A, 0x77)
LIGHT_BLUE = RGBColor(0x3A, 0x7C, 0xA5)
BODY_TEXT  = RGBColor(0x2D, 0x2D, 0x2D)
SUBTLE     = RGBColor(0x66, 0x66, 0x66)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
CODE_BG    = "EEF1F5"
TABLE_HEAD = "0D1B2A"
TABLE_ALT  = "F4F6F9"
BORDER_CLR = "B0BEC5"

TODAY = datetime.now().strftime("%B %d, %Y")


# ── Utility Helpers ─────────────────────────────────────────

def _set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}" w:val="clear"/>')
    tcPr.append(shd)

def _set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="{top}" w:type="dxa"/>'
        f'  <w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'  <w:left w:w="{left}" w:type="dxa"/>'
        f'  <w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def _set_cell_border(cell, color="B0BEC5", sz=2):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="{sz}" w:color="{color}" w:space="0"/>'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:color="{color}" w:space="0"/>'
        f'  <w:left w:val="single" w:sz="{sz}" w:color="{color}" w:space="0"/>'
        f'  <w:right w:val="single" w:sz="{sz}" w:color="{color}" w:space="0"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)

def _set_paragraph_spacing(p, before=0, after=0, line=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if line:
        pf.line_spacing = Pt(line)

def _add_run(paragraph, text, font_name="Calibri", size=11, color=BODY_TEXT, bold=False, italic=False):
    run = paragraph.add_run(text)
    run.font.name = font_name
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    return run

def _add_bottom_border(p, color="3A7CA5", sz=6):
    pPr = p._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:color="{color}" w:space="2"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)

def _remove_table_borders(tbl_element):
    tblPr = tbl_element.tblPr
    if tblPr is None:
        tblPr = parse_xml(f'<w:tblPr {nsdecls("w")}/>')
        tbl_element.insert(0, tblPr)
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="none" w:sz="0" w:color="auto"/>'
        f'  <w:left w:val="none" w:sz="0" w:color="auto"/>'
        f'  <w:bottom w:val="none" w:sz="0" w:color="auto"/>'
        f'  <w:right w:val="none" w:sz="0" w:color="auto"/>'
        f'  <w:insideH w:val="none" w:sz="0" w:color="auto"/>'
        f'  <w:insideV w:val="none" w:sz="0" w:color="auto"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


# ── Cover Page ──────────────────────────────────────────────

def _add_cover_page(doc, title, subtitle, version="1.0"):
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1.15)
    section.right_margin = Inches(1.15)

    for _ in range(6):
        p = doc.add_paragraph()
        _set_paragraph_spacing(p, after=0)

    # Accent bar
    bar = doc.add_table(rows=1, cols=1)
    bar.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = bar.cell(0, 0)
    _set_cell_shading(cell, "3A7CA5")
    p = cell.paragraphs[0]
    _add_run(p, "  ", size=4, color=WHITE)
    _remove_table_borders(bar._tbl)
    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _add_run(p, title, size=32, color=NAVY, bold=True)
    _set_paragraph_spacing(p, after=4)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _add_run(p, subtitle, size=14, color=ACCENT, italic=True)
    _set_paragraph_spacing(p, after=20)

    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.LEFT
    meta_data = [
        ("Document Version", version),
        ("Date", TODAY),
        ("Classification", "CONFIDENTIAL - Internal Use Only"),
        ("Organization", "AeroGCS - IDMS Threat Detection Division"),
    ]
    for i, (label, value) in enumerate(meta_data):
        lc = meta_table.cell(i, 0)
        lc.width = Inches(1.8)
        p = lc.paragraphs[0]
        _add_run(p, label, size=10, color=SUBTLE, bold=True)
        _set_cell_margins(lc, top=40, bottom=40)

        vc = meta_table.cell(i, 1)
        vc.width = Inches(4)
        p = vc.paragraphs[0]
        _add_run(p, value, size=10, color=BODY_TEXT)
        _set_cell_margins(vc, top=40, bottom=40)

    _remove_table_borders(meta_table._tbl)
    doc.add_page_break()


# ── Headers / Footers ───────────────────────────────────────

def _add_header_footer(doc, header_text):
    for section in doc.sections:
        section.different_first_page_header_footer = True

        header = section.header
        header.is_linked_to_previous = False
        p = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _add_run(p, header_text, size=8, color=SUBTLE, italic=True)
        _set_paragraph_spacing(p, after=0)
        _add_bottom_border(p, color="B0BEC5", sz=4)

        footer = section.footer
        footer.is_linked_to_previous = False
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pPr = p._p.get_or_add_pPr()
        pBdr = parse_xml(
            f'<w:pBdr {nsdecls("w")}>'
            f'  <w:top w:val="single" w:sz="4" w:color="B0BEC5" w:space="1"/>'
            f'</w:pBdr>'
        )
        pPr.append(pBdr)
        _add_run(p, "AeroGCS IDMS  |  ", size=8, color=SUBTLE)
        _add_run(p, "CONFIDENTIAL", size=8, color=SUBTLE, bold=True)
        _add_run(p, "  |  Page ", size=8, color=SUBTLE)
        fld_begin = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
        fld_code = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
        fld_end = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
        r1 = p.add_run()._r
        r1.append(fld_begin)
        r2 = p.add_run()
        r2.font.size = Pt(8)
        r2.font.color.rgb = SUBTLE
        r2._r.append(fld_code)
        r3 = p.add_run()._r
        r3.append(fld_end)


# ── Table of Contents ───────────────────────────────────────

def _add_toc(doc, entries):
    p = doc.add_paragraph()
    _add_run(p, "TABLE OF CONTENTS", size=16, color=DARK_BLUE, bold=True)
    _set_paragraph_spacing(p, before=12, after=18)

    for level, title, page_str in entries:
        pe = doc.add_paragraph()
        if level == 2:
            pe.paragraph_format.left_indent = Inches(0.4)
            _set_paragraph_spacing(pe, before=2, after=2)
        elif level == 3:
            pe.paragraph_format.left_indent = Inches(0.8)
            _set_paragraph_spacing(pe, before=1, after=1)
        else:
            _set_paragraph_spacing(pe, before=4, after=4)

        pe.paragraph_format.tab_stops.add_tab_stop(Inches(6.2), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        
        run_title = pe.add_run(title)
        run_title.font.name = "Calibri"
        if level == 1:
            run_title.font.bold = True
            run_title.font.size = Pt(11)
            run_title.font.color.rgb = NAVY
        else:
            run_title.font.size = Pt(10)
            run_title.font.color.rgb = BODY_TEXT

        run_tab = pe.add_run("\t")
        run_tab.font.name = "Calibri"

        run_page = pe.add_run(page_str)
        run_page.font.name = "Calibri"
        if level == 1:
            run_page.font.bold = True
            run_page.font.size = Pt(11)
            run_page.font.color.rgb = NAVY
        else:
            run_page.font.size = Pt(10)
            run_page.font.color.rgb = BODY_TEXT

    doc.add_page_break()


# ── Content Builders ────────────────────────────────────────

def _heading(doc, text, level=1):
    color_map = {1: NAVY, 2: DARK_BLUE, 3: MID_BLUE, 4: ACCENT}
    size_map = {1: 20, 2: 16, 3: 13, 4: 11}
    sp = {1: 22, 2: 16, 3: 12, 4: 8}
    h = doc.add_heading(text, level=level)
    _set_paragraph_spacing(h, before=sp.get(level, 8), after=6)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.color.rgb = color_map.get(level, BODY_TEXT)
        run.font.size = Pt(size_map.get(level, 11))
    if level == 1:
        _add_bottom_border(h, color="3A7CA5", sz=6)
    return h

def _body(doc, text):
    import re
    p = doc.add_paragraph()
    _set_paragraph_spacing(p, after=6, line=16)
    parts = re.split(r'(\*\*.*?\*\*|`[^`]+`)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            _add_run(p, part[2:-2], bold=True)
        elif part.startswith("`") and part.endswith("`"):
            _add_run(p, part[1:-1], font_name="Consolas", size=10, color=MID_BLUE)
        else:
            _add_run(p, part)
    return p

def _bullet(doc, text, level=0):
    import re
    p = doc.add_paragraph(style="List Bullet")
    _set_paragraph_spacing(p, after=3, line=15)
    if level > 0:
        p.paragraph_format.left_indent = Inches(0.5 * level)
    parts = re.split(r'(\*\*.*?\*\*|`[^`]+`)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            _add_run(p, part[2:-2], bold=True)
        elif part.startswith("`") and part.endswith("`"):
            _add_run(p, part[1:-1], font_name="Consolas", size=10, color=MID_BLUE)
        else:
            _add_run(p, part)
    return p

def _code(doc, text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = tbl.cell(0, 0)
    _set_cell_shading(cell, CODE_BG)
    _set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
    _set_cell_border(cell, color="CFD8DC", sz=4)
    p = cell.paragraphs[0]
    _set_paragraph_spacing(p, before=2, after=2)
    _add_run(p, text.strip(), font_name="Consolas", size=9, color=RGBColor(0x1A, 0x23, 0x2F))
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def _table(doc, headers, rows):
    num_cols = len(headers)
    tbl = doc.add_table(rows=1 + len(rows), cols=num_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl.autofit = True
    for ci, h in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.text = ""
        _set_cell_shading(cell, TABLE_HEAD)
        _set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
        _set_cell_border(cell, color="0D1B2A", sz=0)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        _add_run(p, h, size=10, color=WHITE, bold=True)
        _set_paragraph_spacing(p, before=2, after=2)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            if ci >= num_cols:
                break
            cell = tbl.cell(ri + 1, ci)
            cell.text = ""
            bg = TABLE_ALT if ri % 2 == 0 else "FFFFFF"
            _set_cell_shading(cell, bg)
            _set_cell_margins(cell, top=50, bottom=50, left=100, right=100)
            _set_cell_border(cell, color=BORDER_CLR, sz=2)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            _add_run(p, val.strip() if val else "", size=10)
            _set_paragraph_spacing(p, before=1, after=1)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

def _separator(doc):
    p = doc.add_paragraph()
    _add_bottom_border(p, color="CFD8DC", sz=4)
    _set_paragraph_spacing(p, before=6, after=10)


# ══════════════════════════════════════════════════════════════
# DOCUMENT 1: DETECTION PIPELINE SYSTEM DOCUMENTATION
# ══════════════════════════════════════════════════════════════

def build_detection_pipeline_doc(output_path):
    doc = Document()

    _add_cover_page(
        doc,
        "Drone Threat & Object\nDetection Core Pipeline",
        "Technical System Specification -- Architecture, Algorithms & Geodetic Mathematics",
        version="1.0"
    )
    
    toc_entries = [
        (1, "1. Executive Overview", "Page 3"),
        (1, "2. High-Level Architecture & Data Flow", "Page 4"),
        (1, "3. Core Components & Processing Subsystems", "Page 5"),
        (2, "Ingestion & Pre-processing Engine", "Page 5"),
        (2, "Dynamic Model Management", "Page 5"),
        (2, "Object Detection Engine", "Page 6"),
        (2, "OSNet Deep Re-Identification Subsystem", "Page 6"),
        (2, "Person Lifecycle Manager (person_manager.py)", "Page 7"),
        (2, "Evidence Screenshot Manager (screenshot_manager.py)", "Page 7"),
        (1, "4. Telemetry Processing & Geolocation Engine", "Page 8"),
        (2, "Telemetry Data Sources", "Page 8"),
        (2, "Temporal Interpolation", "Page 8"),
        (2, "Distance Selection Priority", "Page 9"),
        (2, "Spherical Geodetic Forward Projection", "Page 9"),
        (2, "Cardinal Direction Mapping", "Page 10"),
        (1, "5. Result Builder & Output Schema", "Page 11"),
        (1, "6. Azure Blob Storage -- Evidence Upload", "Page 12"),
        (1, "7. System Hyperparameters & Configuration", "Page 13"),
        (1, "8. Flask Web Application & Management Interface", "Page 14"),
        (1, "9. File System & Directory Structure", "Page 15"),
        (1, "10. File-by-File Responsibility Matrix", "Page 16")
    ]
    _add_toc(doc, toc_entries)
    _add_header_footer(doc, "Detection Pipeline System Documentation  |  v1.0")

    # ── 1. EXECUTIVE OVERVIEW ──────────────────────────────────
    _heading(doc, "Executive Overview", 1)
    _body(doc,
        "The **Drone Threat & Object Detection Pipeline** is an enterprise-grade computer "
        "vision and telemetry fusion system engineered for autonomous aerial surveillance. "
        "The pipeline processes multi-modal inputs -- combining high-resolution aerial video "
        "streams with synchronous flight telemetry data -- to detect, track, classify, "
        "re-identify, and geo-locate ground-level objects of interest in real time."
    )
    _body(doc,
        "The architecture decouples heavy machine learning inference from background "
        "orchestration workers, ensuring resilient processing, high throughput, and "
        "seamless telemetry alignment. Detection models are dynamically fetched from "
        "the central server's approved model repository, with intelligent local caching "
        "to avoid redundant downloads."
    )
    _body(doc,
        "The system supports multiple object classes determined by the active detection model. "
        "When detections occur, each detected object's class is resolved directly from the "
        "model's trained class set (e.g., person, vehicle, car, truck, motorbike) and sent in "
        "the API output payload. Unique numerical identifiers (`id_number`) are assigned to "
        "each detected entity for tracking and re-identification across the video."
    )

    _separator(doc)

    # ── 2. HIGH-LEVEL ARCHITECTURE ─────────────────────────────
    _heading(doc, "High-Level Architecture & Data Flow", 1)
    _body(doc,
        "The system is structured as a multi-stage pipeline with six primary subsystems "
        "that execute sequentially for each surveillance task:"
    )
    _code(doc,
        "+--------------------------+     +---------------------------+     +----------------------------+\n"
        "|   Media & Telemetry      |     |   Custom YOLO Model       |     |   OSNet Deep Feature       |\n"
        "|   Ingestion Engine       +---->|   Detection Engine         +---->|   Re-ID & Tracking         |\n"
        "+--------------------------+     +---------------------------+     +----------------------------+\n"
        "                                                                                |\n"
        "                                                                                v\n"
        "+--------------------------+     +---------------------------+     +----------------------------+\n"
        "|  REST API Notification   |     |  Azure Blob Storage       |     | Dynamic Telemetry Fusion   |\n"
        "|  & Payload Dispatch      |<----+  Snapshot Upload          |<----+ & Geolocation Engine       |\n"
        "+--------------------------+     +---------------------------+     +----------------------------+"
    )
    _body(doc,
        "Each stage is modular: the ingestion engine downloads media and telemetry, the "
        "detection engine runs the approved YOLO model with ByteTrack, the Re-ID subsystem "
        "assigns stable identities using deep embeddings, the telemetry fusion engine computes "
        "real-world GPS coordinates via spherical geodetic math, evidence snapshots are uploaded "
        "to Azure Blob Storage, and the final structured JSON payload is dispatched to the "
        "backend API."
    )

    _separator(doc)

    # ── 3. CORE COMPONENTS ─────────────────────────────────────
    _heading(doc, "Core Components & Processing Subsystems", 1)

    # 3.1 Ingestion
    _heading(doc, "Ingestion & Pre-processing Engine", 2)
    _body(doc, "The ingestion layer handles all input acquisition and normalization:")
    _bullet(doc, "**Video Download**: Downloads the surveillance video file from the URL provided in the task payload. Saved locally as `{task_id}.mp4` in the `upload/` directory.")
    _bullet(doc, "**Telemetry Acquisition**: Supports two telemetry ingestion modes:")
    _bullet(doc, "**CSV File Download**: Downloads `offline_teledata_` and `offline_flightdata_` CSV files from URLs provided in the task. The flight data CSV is derived by replacing `offline_teledata_` with `offline_flightdata_` in the URL.", level=1)
    _bullet(doc, "**In-Memory JSON**: When the backend provides telemetry data directly embedded in the task payload as a JSON array, it is parsed in-memory without requiring file downloads.", level=1)
    _bullet(doc, "**Frame Decimation Engine**: Converts input video streams (e.g., 30 FPS native) to a decimated processing rate (default: `10 FPS`). The skip interval is calculated as `skip_interval = max(1, int(video_fps / target_fps))`. Only every Nth frame is processed, balancing real-time efficiency with detection precision.")
    _bullet(doc, "**Video Start Epoch Detection**: Parses the flight data log to find the exact Unix timestamp when video recording started, by searching for log messages containing 'video' combined with 'visible' or 'start'. This timestamp anchors all telemetry lookups.")

    # 3.2 Model Management
    _heading(doc, "Dynamic Model Management", 2)
    _body(doc,
        "Prior to running object detection, the system manages model checkpoint selection through "
        "a multi-step resolution and caching strategy. The approved detection model is no longer "
        "hardcoded; it is fetched dynamically from the backend API:"
    )
    _bullet(doc, "**Step 1 -- Task Payload Check**: The task payload is inspected for a `model_url` field (also checked under aliases `detection_model_url`, `weights_url`, `model_path`). If provided, this URL is used.")
    _bullet(doc, "**Step 2 -- Active Model API Fetch**: If no model URL exists in the payload, the worker queries the backend endpoint `GET /models?is_active=true` to retrieve the currently active, backend-approved detection model. The response includes `download_url` (or `blob_url` + `sas_token`) and `model_name`.")
    _bullet(doc, "**Step 3 -- Local Cache Verification**: The target filename is computed from `model_name` (sanitized to alphanumeric, hyphens, underscores) with a `.pt` extension. The system checks the `downloaded_model/` directory for this file.")
    _bullet(doc, "**Step 4 -- Conditional Download**: If the model file already exists locally, downloading is skipped entirely and the cached model is loaded immediately. If not found, the model is downloaded via HTTP, saved to `downloaded_model/`, and loaded.")
    _bullet(doc, "**Step 5 -- Hot Reload**: If the current task specifies a different model path than the previously loaded model, the `VideoProcessor` hot-reloads the new detector without restarting the service.")
    _body(doc,
        "This caching mechanism eliminates redundant network transfers and ensures rapid task "
        "startup for consecutive jobs using the same approved model."
    )

    # 3.3 Detection & Tracking
    _heading(doc, "Object Detection Engine", 2)
    _body(doc, "The detection subsystem is built around the Ultralytics YOLO framework:")
    _bullet(doc, "**Model Loading**: The approved `.pt` model weights are loaded via `YOLO(model_path)`. The model is loaded once and reused across multiple video jobs for efficiency.")
    _bullet(doc, "**Inference**: Each decimated frame is passed to `model.track()` with `persist=True` for continuous multi-object tracking across frames. ByteTrack (`bytetrack.yaml`) handles frame-to-frame spatial association.")
    _bullet(doc, "**Detection Confidence Threshold**: Configurable via `confidence` parameter (default: `0.50`). Only detections above this threshold are retained.")
    _bullet(doc, "**Dynamic Class Resolution**: For each detection, the model returns a `class_id` integer. This is resolved to a human-readable class name via `model.names[class_id]` -- the class names are inherent to the model's training configuration. A `normalize_class()` function standardizes synonyms:")
    _bullet(doc, "`person`, `human`, `pedestrian`, `people` --> `person`", level=1)
    _bullet(doc, "`van`, `car`, `truck`, `bus`, `motorbike`, `bicycle`, `automobile` --> `vehicle`", level=1)
    _bullet(doc, "All other class names are passed through unchanged.", level=1)
    _bullet(doc, "**Output per Detection**: Each detection yields: `track_id` (ByteTrack integer), `bbox` ([x1,y1,x2,y2] pixels), `confidence` (float 0-1), `class_id` (integer), `class` (normalized string).")

    # 3.4 Re-Identification
    _heading(doc, "OSNet Deep Re-Identification Subsystem", 2)
    _body(doc,
        "The Re-ID subsystem assigns stable, permanent identities to detected objects that "
        "survive ByteTrack ID switches, occlusions, and re-entries into the frame. It consists "
        "of four layered components:"
    )

    _heading(doc, "OSNet Feature Extractor (reid_model.py)", 3)
    _bullet(doc, "**Architecture**: OSNet_x1_0 -- a lightweight convolutional neural network designed for person re-identification, trained on Market-1501 dataset.")
    _bullet(doc, "**Input**: Cropped person image (BGR numpy array from OpenCV), resized to `256 x 128` pixels.")
    _bullet(doc, "**Output**: L2-normalized `512`-dimensional float32 embedding vector.")
    _bullet(doc, "**Preprocessing**: BGR-to-RGB conversion, PIL resize, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).")
    _bullet(doc, "**Optimization**: Supports FP16 half-precision inference on CUDA GPUs for reduced VRAM usage and faster inference. Batch processing via `extract_batch()` sends all crops through the network in a single forward pass.")
    _bullet(doc, "**Fallback**: If `torchreid` package is not installed, a bundled architecturally-identical fallback implementation is used automatically.")

    _heading(doc, "Embedding Database (embedding_database.py)", 3)
    _bullet(doc, "**Storage**: Thread-safe in-memory dictionary of `PersonRecord` objects, each containing: `global_id`, `embedding` (512-D), `tracker_id`, `first_seen`, `last_seen`, `bbox`, `confidence`, `camera_id`, `hits`.")
    _bullet(doc, "**Gallery Arrays**: Provides `gallery_arrays()` method returning all embeddings as a stacked numpy matrix for vectorized similarity search.")
    _bullet(doc, "**EMA Update**: Embedding smoothing via exponential moving average: `new_stored = (1 - alpha) * old + alpha * current`, then L2-renormalized. Default alpha: `0.2`.")
    _bullet(doc, "**ID Generation**: Sequential integer IDs with configurable prefix (default: no prefix) and zero-padding.")
    _bullet(doc, "**Stale Purge**: Optional TTL-based cleanup for long videos or live streams (currently disabled -- `identity_ttl_seconds = None`).")

    _heading(doc, "Cosine Similarity Engine (similarity.py)", 3)
    _bullet(doc, "**Cosine Similarity**: `cos_sim(a, b) = dot(normalize(a), normalize(b))`. Since embeddings are already L2-normalized by the extractor, this reduces to a dot product -- but defensive re-normalization is applied.")
    _bullet(doc, "**Matrix Search**: `cosine_similarity_matrix(query, gallery)` computes similarity of one query against all gallery entries via matrix multiplication `gallery @ query` for vectorized performance.")
    _bullet(doc, "**Best Match**: Returns `(best_id, best_score)` if the top score exceeds the `similarity_threshold` (default: `0.68`); otherwise returns `(None, score)`.")

    _heading(doc, "Identity Manager (reid/identity_manager.py)", 3)
    _body(doc, "The brain of the Re-ID subsystem. Resolution order for each incoming detection:")
    _bullet(doc, "**Fast Path**: If the detection's `tracker_id` is already mapped to a known identity in the database, the existing identity is updated (cheapest, most common case).")
    _bullet(doc, "**Gallery Search**: If the tracker_id is new (genuinely new object, or ByteTrack reassigned an ID), compute cosine similarity against all gallery embeddings.")
    _bullet(doc, "If the best match score >= `similarity_threshold` (0.68) and passes optional spatio-temporal gating, re-attach to that existing identity.", level=1)
    _bullet(doc, "Otherwise, mint a brand-new global identity.", level=1)
    _bullet(doc, "**Spatio-temporal Gating (Optional, disabled by default)**: Additional safety checks -- time gap (max 8s), spatial center jump (max 480px), bounding box area ratio (max 2.5x) -- to prevent false merges.")

    # 3.5 Person Manager
    _heading(doc, "Person Lifecycle Manager (person_manager.py)", 2)
    _body(doc, "Maintains the complete lifecycle of every detected entity:")
    _bullet(doc, "**Appearance Tracking**: Records each continuous appearance segment with `entry_time`, `exit_time`, `duration`, and GPS coordinates at both entry and exit.")
    _bullet(doc, "**Re-identification Counting**: Increments `reidentification_count` each time a previously-exited entity re-enters the frame.")
    _bullet(doc, "**GPS Tagging**: When telemetry is available, each appearance records: `entry_lat/lon/alt/distance/heading` and `exit_lat/lon/alt/distance/heading`.")
    _bullet(doc, "**Class Propagation**: The detected `class` label (from the YOLO model) is stored with each entity on first detection.")
    _bullet(doc, "**Finalization**: At video end, any still-active appearances are closed with the final frame timestamp.")

    # 3.6 Screenshot Manager
    _heading(doc, "Evidence Screenshot Manager (screenshot_manager.py)", 2)
    _bullet(doc, "**Best Detection Selection**: For each entity, tracks the detection with the largest bounding box area (highest visual quality).")
    _bullet(doc, "**Evidence Image**: Saves the full frame with a green bounding box and Person ID label overlaid as `best_detection.jpg` in `outputs/{task_id}/person_{id}/`.")
    _bullet(doc, "**File Renaming**: The worker renames evidence files to `{Person_ID}_{timestamp}` format before upload.")

    _separator(doc)

    # ── 4. TELEMETRY & GEOLOCATION ─────────────────────────────
    _heading(doc, "Telemetry Processing & Geolocation Engine", 1)

    _heading(doc, "Telemetry Data Sources", 2)
    _body(doc, "The telemetry engine (`telemetry.py`) supports two input formats:")
    _bullet(doc, "**CSV File**: `offline_teledata_{flight_id}.csv` -- a mixed-format CSV containing metadata key/value rows interleaved with actual per-second telemetry rows. The parser identifies valid telemetry rows by column count matching and numeric timestamp validation, discarding metadata rows (DroneId, FlightId, planBoundary, GpsInfo, etc.).")
    _bullet(doc, "**JSON Array**: In-memory list of telemetry dictionaries or list-of-lists. Supports both dict format (keys: `timestamp`, `lat`, `lon`, `alt`, `yaw`/`dir`, `ObsDist`/`obsDist`, `liveObstacleDistance`) and nested `data` sub-objects.")
    _body(doc, "Each telemetry sample is parsed into a tuple of: `(timestamp, lat, lon, alt, yaw, obsDist, liveObstacleDistance)` and sorted chronologically.")

    _heading(doc, "Temporal Interpolation", 2)
    _body(doc,
        "When the video timestamp for a detection falls between two telemetry sample points, "
        "linear interpolation is applied to compute the exact drone position:"
    )
    _bullet(doc, "**Linear interpolation** for latitude, longitude, altitude, obstacle distances: `value = val0 + (val1 - val0) * fraction` where `fraction = (target_ts - t0) / (t1 - t0)`.")
    _bullet(doc, "**Circular interpolation for yaw/heading** to correctly handle the 0/360 degree wrap-around:")
    _code(doc,
        "diff = (yaw1 - yaw0 + 180) mod 360 - 180\n"
        "yaw  = (yaw0 + diff * fraction) mod 360"
    )
    _bullet(doc, "If the target timestamp is before/after all samples, the first/last sample values are used (clamp behavior).")

    _heading(doc, "Distance Selection Priority", 2)
    _body(doc,
        "The distance `d` between the drone and the detected target is resolved through a cascading "
        "priority from available sensor data:"
    )
    _table(doc,
        ["Priority", "Source Field", "Description"],
        [
            ["1 (Highest)", "obsDist", "Obstacle Distance Sensor -- primary distance measurement from the drone's onboard obstacle avoidance sensor. Used if the value is > 0."],
            ["2", "liveObstacleDistance", "Live Obstacle Distance stream -- secondary real-time distance measurement. Used if obsDist is 0 but this value is > 0."],
            ["3 (Fallback)", "Default (10.0 m)", "Configurable baseline fallback distance used when no obstacle sensor data is available."],
        ]
    )

    _heading(doc, "Spherical Geodetic Forward Projection", 2)
    _body(doc,
        "To calculate the target's real-world GPS coordinates from the drone's position, the system "
        "uses the direct (forward) geodetic problem on a spherical Earth model. Given the drone's "
        "coordinates, its heading (yaw), and the distance to the target:"
    )

    _heading(doc, "Input Variables", 3)
    _table(doc,
        ["Variable", "Symbol", "Description"],
        [
            ["Drone Latitude", "Lat1", "Drone's real-time GPS latitude, converted to radians"],
            ["Drone Longitude", "Lon1", "Drone's real-time GPS longitude, converted to radians"],
            ["Drone Heading/Yaw", "psi", "Orientation angle relative to True North (0-360 degrees), converted to radians"],
            ["Target Distance", "d", "Horizontal distance from drone to target in meters (from sensor or default)"],
            ["Earth Radius", "R", "Mean radius of Earth = 6,378,137 meters (WGS-84 semi-major axis)"],
        ]
    )

    _heading(doc, "Formula: Target Latitude (Lat2)", 3)
    _code(doc,
        "Lat2 = arcsin(\n"
        "    sin(Lat1) * cos(d / R)  +  cos(Lat1) * sin(d / R) * cos(psi)\n"
        ")"
    )

    _heading(doc, "Formula: Target Longitude (Lon2)", 3)
    _code(doc,
        "Lon2 = Lon1 + atan2(\n"
        "    sin(psi) * sin(d / R) * cos(Lat1),\n"
        "    cos(d / R) - sin(Lat1) * sin(Lat2)\n"
        ")"
    )

    _heading(doc, "Output Precision", 3)
    _bullet(doc, "Latitude and Longitude: rounded to 7 decimal places (sub-meter precision).")
    _bullet(doc, "Altitude: rounded to 2 decimal places.")
    _bullet(doc, "Distance: rounded to 2 decimal places.")

    _heading(doc, "Cardinal Direction Mapping", 2)
    _body(doc,
        "The drone's heading yaw angle (0-360 degrees) is converted into standard 8-point cardinal "
        "compass directions using 45-degree sectors offset by 22.5 degrees:"
    )
    _code(doc,
        "index = int((yaw + 22.5) mod 360 / 45)\n"
        "directions = [N, NE, E, SE, S, SW, W, NW]"
    )
    _table(doc,
        ["Angle Range", "Direction", "Full Name"],
        [
            ["337.5 - 22.5",  "N",  "North"],
            ["22.5 - 67.5",   "NE", "North-East"],
            ["67.5 - 112.5",  "E",  "East"],
            ["112.5 - 157.5", "SE", "South-East"],
            ["157.5 - 202.5", "S",  "South"],
            ["202.5 - 247.5", "SW", "South-West"],
            ["247.5 - 292.5", "W",  "West"],
            ["292.5 - 337.5", "NW", "North-West"],
        ]
    )

    _separator(doc)

    # ── 5. RESULT BUILDER ──────────────────────────────────────
    _heading(doc, "Result Builder & Output Schema", 1)
    _body(doc,
        "The `result_builder.py` module transforms the internal person_manager output into the "
        "structured JSON format required by the Detection Update API:"
    )

    _heading(doc, "Identity Fields", 2)
    _table(doc,
        ["Field", "Type", "Description"],
        [
            ["id_number", "String", "The unique numerical identifier for the detected entity, sent as a string (e.g., '1', '2', '3'). This is the internal person ID converted to string."],
            ["class", "String", "The detected object's classification label, resolved from the model's class names at detection time (e.g., 'person', 'vehicle', 'car', 'truck', 'motorbike'). This field is dynamically determined by the active detection model."],
        ]
    )

    _heading(doc, "Time Formatting", 2)
    _body(doc, "Entry and exit times are formatted as `HH:MM:SS` strings relative to video start:")
    _code(doc,
        "hours   = int(seconds // 3600)\n"
        "minutes = int((seconds % 3600) // 60)\n"
        "secs    = int(seconds % 60)\n"
        "result  = f\"{hours:02}:{minutes:02}:{secs:02}\""
    )

    _separator(doc)

    # ── 6. AZURE UPLOAD ────────────────────────────────────────
    _heading(doc, "Azure Blob Storage -- Evidence Upload", 1)
    _body(doc, "Evidence snapshot images are uploaded to Azure Blob Storage using BlockBlob SAS authorization:")
    _bullet(doc, "**Upload Path Format**: `{container_url}/threatimages/{flight_id}/{filename}.jpg?{sas_token}`")
    _bullet(doc, "**SAS URL Source**: Provided in the task payload via `upload_url`, `sas_url`, `azure_sas_url`, or `upload_sas_url` fields.")
    _bullet(doc, "**Upload Method**: HTTP PUT request with header `x-ms-blob-type: BlockBlob` and 60-second timeout.")
    _bullet(doc, "**Return Value**: The public blob URL (without SAS token) is stored as `file_url` in the result payload.")
    _bullet(doc, "**Fallback**: If Azure upload fails, a fallback URL pointing to the local server path is generated.")

    _separator(doc)

    # ── 7. HYPERPARAMETERS ─────────────────────────────────────
    _heading(doc, "System Hyperparameters & Configuration", 1)
    _table(doc,
        ["Parameter", "Value", "Description"],
        [
            ["Detection Model", "Dynamic (API-approved .pt)", "Custom-trained YOLO weights fetched from the active model endpoint or task payload"],
            ["Detection Confidence", "0.50", "Minimum confidence threshold for detection validation"],
            ["Tracking Engine", "ByteTrack (bytetrack.yaml)", "Frame-to-frame spatial multi-object tracker with persistent tracking"],
            ["Re-ID Feature Model", "OSNet (osnet_x1_0)", "Deep convolutional embedding feature extractor (Market-1501 pretrained)"],
            ["Embedding Dimension", "512 Float32", "L2-normalized feature representation vector"],
            ["Cosine Similarity Cutoff", "0.68", "Minimum cosine similarity score for identity re-entry match"],
            ["Embedding EMA Alpha", "0.2", "Exponential moving average update rate for gallery embeddings"],
            ["Soft Margin", "0.05", "Additional margin below threshold for spatio-temporal gating acceptance"],
            ["Frame Processing Rate", "10 FPS", "Decimated frame rate for inference (configurable via target_fps)"],
            ["Input Size (Re-ID)", "256 x 128 px", "Height x Width resize for OSNet input"],
            ["FP16 Inference", "Enabled (CUDA)", "Half-precision inference for GPU memory and speed optimization"],
            ["Earth Radius Constant", "6,378,137 m", "WGS-84 semi-major axis used in geodetic calculations"],
            ["Default Target Distance", "10.0 m", "Baseline fallback when no obstacle sensor data available"],
            ["Identity TTL", "None (Disabled)", "Preserves all identities across full video length"],
            ["Max Download Size", "2 GB", "Safety cap for video file downloads"],
        ]
    )

    _separator(doc)

    # ── 8. FLASK APPLICATION ───────────────────────────────────
    _heading(doc, "Flask Web Application & Management Interface", 1)
    _body(doc,
        "The system includes a Flask web application (`app.py`) running on `http://localhost:5000` "
        "that provides a single-page operator interface for manual job submission:"
    )
    _bullet(doc, "**URL Mode**: Paste a video URL (and optional telemetry CSV URL), click Start. The server downloads and processes the video.")
    _bullet(doc, "**Upload Mode**: Directly upload a video file (and optional CSV) from the local machine via browser file picker.")
    _bullet(doc, "**Progress Tracking**: Real-time progress bar with percentage and status messages updated via polling at 1.5-second intervals.")
    _bullet(doc, "**Concurrency Control**: Single-job processing via a global `threading.Lock()`. If a job is already running, new submissions are rejected with HTTP 409.")
    _bullet(doc, "**Background Worker**: On startup, `start_worker()` launches an autonomous background polling thread that continuously queries the backend API for pending tasks, independent of the web UI.")

    _separator(doc)

    # ── 9. DIRECTORY STRUCTURE ─────────────────────────────────
    _heading(doc, "File System & Directory Structure", 1)
    _code(doc,
        "ml_pipeline/app/\n"
        "|-- app.py                      # Flask web application & management interface\n"
        "|-- downloader.py               # Video, telemetry CSV, and model downloader\n"
        "|-- main.py                     # Standalone CLI entry point\n"
        "|-- downloaded_model/           # Local cache for API-fetched detection models (.pt)\n"
        "|-- models/                     # Static model weights (OSNet Re-ID)\n"
        "|   |-- osnet_x1_0_imagenet.pth # OSNet pretrained weights\n"
        "|-- pipeline/\n"
        "|   |-- detector.py             # YOLO detector, class resolution & ByteTrack\n"
        "|   |-- person_manager.py       # Entity lifecycle & appearance tracker\n"
        "|   |-- telemetry.py            # Geodetic telemetry parser & forward projection\n"
        "|   |-- video_processor.py      # Main video decimation & pipeline coordinator\n"
        "|   |-- screenshot_manager.py   # Evidence crop selection & snapshot exporter\n"
        "|   |-- json_writer.py          # Local output JSON analytics writer\n"
        "|   |-- identity_manager.py     # Pipeline-level Re-ID adapter (wraps reid/)\n"
        "|   |-- utils.py                # Time formatting & cardinal direction conversion\n"
        "|   |-- reid/\n"
        "|       |-- config.py           # Re-ID subsystem configuration (all tunables)\n"
        "|       |-- identity_manager.py # Core Re-ID identity resolution brain\n"
        "|       |-- embedding_database.py # Thread-safe in-memory embedding store\n"
        "|       |-- reid_model.py       # OSNet feature extractor (batched PyTorch)\n"
        "|       |-- similarity.py       # Cosine similarity & gallery search\n"
        "|       |-- osnet_fallback.py   # Bundled OSNet architecture (no torchreid dependency)\n"
        "|       |-- utils.py            # Crop helper (person bounding box extraction)\n"
        "|-- worker/\n"
        "|   |-- api_client.py           # REST client: job polling, status updates, model fetch\n"
        "|   |-- result_builder.py       # Structured JSON schema builder for API output\n"
        "|   |-- worker.py               # Autonomous task worker, Azure uploader, cleanup\n"
        "|-- upload/                     # Temporary storage for downloaded videos/CSVs\n"
        "|-- outputs/                    # Per-task output directories with evidence images"
    )

    _separator(doc)

    # ── 10. FILE RESPONSIBILITY MATRIX ─────────────────────────
    _heading(doc, "File-by-File Responsibility Matrix", 1)
    _table(doc,
        ["File", "Category", "Responsibility"],
        [
            ["app.py", "Web Application", "Flask server, operator UI, manual job submission, progress polling, startup worker launch"],
            ["downloader.py", "Asset Manager", "HTTP file downloader for videos, CSVs, and detection model weights"],
            ["worker/worker.py", "Background Worker", "Autonomous polling loop, downloads assets, runs ML pipeline, renames/uploads evidence to Azure, notifies backend API"],
            ["worker/api_client.py", "REST Client", "GET /detection/pending, GET /models?is_active, PUT /detection/update/{task_id} for status/results/failure callbacks"],
            ["worker/result_builder.py", "Schema Builder", "Transforms person_manager output into API-compliant JSON with id_number, class, appearances, locations"],
            ["pipeline/video_processor.py", "Pipeline Orchestrator", "VideoProcessor class: model loading, frame decimation, detection/tracking/ReID/telemetry coordination"],
            ["pipeline/detector.py", "Detector", "YOLO model loading, track() inference, class resolution via model.names, normalize_class()"],
            ["pipeline/person_manager.py", "Entity Tracker", "PersonManager: appearance lifecycle, entry/exit GPS tagging, re-identification counting, finalization"],
            ["pipeline/telemetry.py", "Geolocation Engine", "TelemetryReader: CSV/JSON parsing, temporal interpolation, forward geodetic projection, cardinal directions"],
            ["pipeline/screenshot_manager.py", "Evidence Manager", "Best detection selection (max bounding box area), evidence image saving with annotations"],
            ["pipeline/identity_manager.py", "Re-ID Adapter", "Pipeline-level wrapper connecting detector to Re-ID subsystem (extract + resolve)"],
            ["pipeline/reid/config.py", "Configuration", "ReIDConfig dataclass: all Re-ID tunables (threshold, alpha, device, TTL, gating parameters)"],
            ["pipeline/reid/identity_manager.py", "Re-ID Brain", "Identity resolution: fast-path lookup, gallery cosine search, spatio-temporal gating, new ID minting"],
            ["pipeline/reid/embedding_database.py", "Embedding Store", "Thread-safe in-memory PersonRecord storage, gallery array export, EMA update, stale purge"],
            ["pipeline/reid/reid_model.py", "Feature Extractor", "OSNet_x1_0 model loading, weight matching, FP16 inference, batched embedding extraction"],
            ["pipeline/reid/similarity.py", "Math Engine", "Cosine similarity computation, matrix search, best-match threshold evaluation"],
        ]
    )

    doc.save(output_path)
    print(f"[OK] Generated: {output_path}")


# ══════════════════════════════════════════════════════════════
# DOCUMENT 2: API INTEGRATION & COMMUNICATION DOCUMENTATION
# ══════════════════════════════════════════════════════════════

def build_api_doc(output_path):
    doc = Document()

    _add_cover_page(
        doc,
        "API Integration &\nCommunication Protocol",
        "REST Interface Specification -- Endpoints, Payloads, Model Management & Cloud Storage",
        version="1.0"
    )
    
    toc_entries = [
        (1, "1. Executive Overview", "Page 3"),
        (1, "2. System Interaction Workflow", "Page 4"),
        (1, "3. Input -- Data Received via API", "Page 5"),
        (1, "4. Model Fetching & Dynamic Caching", "Page 6"),
        (1, "5. REST API Endpoint Specification", "Page 7"),
        (2, "1. Fetch Pending Job", "Page 7"),
        (2, "2. Fetch Active Detection Model", "Page 7"),
        (2, "3. Update Task Status", "Page 8"),
        (2, "4. Submit Completed Task Results", "Page 8"),
        (2, "5. Report Task Failure", "Page 9"),
        (1, "6. Output -- Completed Results JSON Schema", "Page 10"),
        (1, "7. Output Field Definitions", "Page 11"),
        (2, "Top-Level Result Fields", "Page 11"),
        (2, "Evidence Fields", "Page 11"),
        (2, "Appearance Fields", "Page 11"),
        (2, "Location Fields (entry_location / exit_location)", "Page 12"),
        (2, "Reserved Fields (Future Use)", "Page 12"),
        (1, "8. Azure Blob Storage -- Evidence Upload Protocol", "Page 13"),
        (1, "9. Autonomous Worker Behavior", "Page 14")
    ]
    _add_toc(doc, toc_entries)
    _add_header_footer(doc, "API Integration & Communication Documentation  |  v1.0")

    # ── 1. EXECUTIVE OVERVIEW ──────────────────────────────────
    _heading(doc, "Executive Overview", 1)
    _body(doc,
        "This document provides the complete specification of REST API interfaces, data flow "
        "mechanisms, request/response payload schemas, and cloud storage integration for the "
        "**AeroGCS IDMS Threat Detection Service**."
    )
    _body(doc,
        "The autonomous worker service operates a continuous polling cycle: it queries the backend "
        "API for pending surveillance tasks, downloads raw video and flight telemetry assets, "
        "dynamically fetches or verifies approved object detection models, executes the full ML "
        "inference pipeline, uploads evidence snapshots to Azure Blob Storage, and dispatches "
        "structured analytical results back to the central server via REST callbacks."
    )

    _separator(doc)

    # ── 2. SYSTEM INTERACTION WORKFLOW ─────────────────────────
    _heading(doc, "System Interaction Workflow", 1)
    _body(doc, "The following sequence illustrates the complete task lifecycle from polling to completion:")
    _code(doc,
        "+-------------------+            +--------------------+            +------------------+\n"
        "| Central Backend   |            | Autonomous Worker  |            | Cloud Storage    |\n"
        "|   API Server      |            |     Service        |            |  (Azure Blob)    |\n"
        "+--------+----------+            +---------+----------+            +--------+---------+\n"
        "         |                                 |                                |\n"
        "         |  1. GET /detection/pending      |                                |\n"
        "         |<--------------------------------+                                |\n"
        "         |  2. Job payload response        |                                |\n"
        "         +-------------------------------->|                                |\n"
        "         |                                 |                                |\n"
        "         |  3. PUT /detection/update       |                                |\n"
        "         |    status=IN_PROGRESS           |                                |\n"
        "         |<--------------------------------+                                |\n"
        "         |                                 |                                |\n"
        "         |  4. GET /models?is_active=true  |                                |\n"
        "         |    (if no model in payload)     |                                |\n"
        "         |<--------------------------------+                                |\n"
        "         |  5. Active model URL response   |                                |\n"
        "         +-------------------------------->|                                |\n"
        "         |                                 |                                |\n"
        "         |  6. Download video, CSVs, model |                                |\n"
        "         +-------------------------------->|                                |\n"
        "         |                                 |                                |\n"
        "         |                                 | 7. Execute Detection Pipeline  |\n"
        "         |                                 |    (YOLO + ByteTrack + ReID    |\n"
        "         |                                 |     + Geodetic Telemetry)      |\n"
        "         |                                 |                                |\n"
        "         |                                 | 8. Upload evidence images      |\n"
        "         |                                 +------------------------------->|\n"
        "         |                                 | 9. Return blob storage URL     |\n"
        "         |                                 |<-------------------------------+\n"
        "         |                                 |                                |\n"
        "         | 10. PUT /detection/update       |                                |\n"
        "         |     status=COMPLETED + results  |                                |\n"
        "         |<--------------------------------+                                |\n"
        "         | 11. Acknowledged                |                                |\n"
        "         +-------------------------------->|                                |"
    )

    _separator(doc)

    # ── 3. INPUT DATA ──────────────────────────────────────────
    _heading(doc, "Input -- Data Received via API", 1)
    _body(doc,
        "The worker receives all task inputs through the pending job API response. "
        "The following data elements are extracted from the task payload and processed:"
    )
    _table(doc,
        ["Input Field", "Required", "Source", "Description"],
        [
            ["task_id", "Yes", "API Response", "Unique identifier for the detection task"],
            ["flight_id", "Yes", "API Response", "Associated drone flight session identifier"],
            ["video_url", "Yes", "API Response", "URL to download the surveillance video (.mp4)"],
            ["status", "Yes", "API Response", "Current task status (expected: PENDING)"],
            ["csv_url", "No", "API Response", "URL to download offline_teledata_ CSV telemetry file. The flightdata CSV URL is derived by replacing offline_teledata_ with offline_flightdata_ in this URL."],
            ["telemetry / teledata", "No", "API Response", "In-memory telemetry data as JSON array. Used when embedded directly in the task payload instead of CSV files."],
            ["flightdata / flight_data / flight_logs", "No", "API Response", "Flight event log data (JSON array). Used to detect video start epoch by scanning for video-related log messages."],
            ["model_url", "No", "API Response", "URL to download the approved detection model (.pt). Also checked under aliases: detection_model_url, weights_url, model_path."],
            ["model_name", "No", "API Response", "Human-readable model name. Also checked under alias: detection_model_name."],
            ["upload_url / sas_url / azure_sas_url / upload_sas_url", "No", "API Response", "Azure Blob Storage SAS URL for evidence image upload."],
        ]
    )

    _separator(doc)

    # ── 4. MODEL MANAGEMENT ────────────────────────────────────
    _heading(doc, "Model Fetching & Dynamic Caching", 1)
    _body(doc,
        "The detection model lifecycle is managed through a cascading resolution strategy "
        "that prioritizes local cache to minimize network overhead and startup latency:"
    )

    _heading(doc, "Resolution Workflow", 2)
    _bullet(doc, "**Step 1 -- Task Payload Inspection**: Check if the job payload contains `model_url` (or aliases `detection_model_url`, `weights_url`, `model_path`). If present, use it as the download source.")
    _bullet(doc, "**Step 2 -- Active Model API Query**: If no model URL is in the payload, query `GET /models?is_active=true` to retrieve the currently approved model. The response contains a `models` array; the first model is selected.")
    _bullet(doc, "**Step 3 -- Download URL Assembly**: If the model object has a `download_url`, use it directly. Otherwise, construct the URL from `blob_url` + `sas_token` fields.")
    _bullet(doc, "**Step 4 -- Filename Computation**: Derive the target local filename from `model_name` (sanitized: only alphanumeric, hyphens, underscores allowed, appended with `.pt`) or from the URL basename if no name is provided.")
    _bullet(doc, "**Step 5 -- Local Cache Check**: Check `downloaded_model/` directory for the computed filename. If the file exists, skip download entirely and load the cached model.")
    _bullet(doc, "**Step 6 -- Remote Download**: If not cached, download the model from the remote URL via HTTP streaming and save to `downloaded_model/`.")

    _heading(doc, "Active Model API Endpoint", 2)
    _table(doc,
        ["Property", "Value"],
        [
            ["Method", "GET"],
            ["Endpoint", "/AeroMegh/idms/models?is_active=true"],
            ["Timeout", "30 seconds"],
            ["Description", "Returns the currently active, backend-approved detection model"],
        ]
    )
    _heading(doc, "Response Payload", 3)
    _code(doc,
        '{\n'
        '    "models": [\n'
        '        {\n'
        '            "model_name": "yolov8_drone_v2",\n'
        '            "download_url": "http://server/uploads/models/yolov8_drone_v2.pt",\n'
        '            "blob_url": "https://storage.blob.core.windows.net/models/model.pt",\n'
        '            "sas_token": "?sv=2026-06-06&sp=r&sig=token_here",\n'
        '            "file_name": "yolov8_drone_v2.pt"\n'
        '        }\n'
        '    ]\n'
        '}'
    )

    _separator(doc)

    # ── 5. REST API SPECIFICATION ──────────────────────────────
    _heading(doc, "REST API Endpoint Specification", 1)
    _body(doc, "Base URL: `http://114.143.215.196:5005/AeroMegh/idms`")

    # 5.1 Fetch Pending Job
    _heading(doc, "1. Fetch Pending Job", 2)
    _table(doc,
        ["Property", "Value"],
        [
            ["Method", "GET"],
            ["Endpoint", "/detection/pending"],
            ["Full URL", "http://114.143.215.196:5005/AeroMegh/idms/detection/pending"],
            ["Timeout", "30 seconds"],
            ["Content-Type", "N/A (no request body)"],
            ["Description", "Queries for pending detection jobs. Returns HTTP 204 or empty body if no jobs available."],
        ]
    )
    _heading(doc, "Response Payload", 3)
    _code(doc,
        '{\n'
        '    "tasks": [\n'
        '        {\n'
        '            "task_id": "TASK_98412",\n'
        '            "flight_id": "FLIGHT_20260803_01",\n'
        '            "video_url": "http://server/uploads/video_98412.mp4",\n'
        '            "csv_url": "http://server/uploads/offline_teledata_FLIGHT_01.csv",\n'
        '            "model_url": "http://server/uploads/models/approved_model.pt",\n'
        '            "model_name": "yolov8_aerial_approved",\n'
        '            "upload_url": "https://blob.core.windows.net/container?sas_token",\n'
        '            "status": "PENDING",\n'
        '            "telemetry": [ { "timestamp": 1723..., "data": { "lat": ..., "lon": ... } } ],\n'
        '            "flight_logs": [ { "key": "AS_TELE_DATA", ... } ]\n'
        '        }\n'
        '    ]\n'
        '}'
    )
    _body(doc, "**Telemetry extraction from flight_logs**: If the task contains `flight_logs` but no `telemetry` field, the worker filters logs where `key == \"AS_TELE_DATA\"` to construct the telemetry array. Similarly, `flightdata` falls back to the full `flight_logs` array for video start epoch scanning.")

    # 5.2 Update Status
    _heading(doc, "2. Update Task Status", 2)
    _table(doc,
        ["Property", "Value"],
        [
            ["Method", "PUT"],
            ["Endpoint", "/detection/update/{task_id}"],
            ["Content-Type", "application/json"],
            ["Timeout", "60 seconds"],
            ["Description", "Updates the task processing status"],
        ]
    )
    _heading(doc, "Request Payload", 3)
    _code(doc, '{\n    "status": "IN_PROGRESS"\n}')
    _body(doc, "**Status Values**: `IN_PROGRESS` (detection is running), `COMPLETED` (results ready), `FAILED` (error occurred).")

    # 5.3 Submit Completed Results
    _heading(doc, "3. Submit Completed Task Results", 2)
    _table(doc,
        ["Property", "Value"],
        [
            ["Method", "PUT"],
            ["Endpoint", "/detection/update/{task_id}"],
            ["Content-Type", "multipart/form-data (with evidence files) or application/json (without files)"],
            ["Timeout", "120 seconds"],
            ["Description", "Submits full detection analytics, evidence files, and GPS telemetry results"],
        ]
    )
    _body(doc,
        "**Submission Strategy**: The worker first attempts a multipart upload that includes both the "
        "JSON results and evidence image files. If the multipart upload fails, it falls back to a "
        "JSON-only submission. File size metadata is included as form fields:"
    )
    _bullet(doc, "`{filename}_size` -- file size in bytes")
    _bullet(doc, "`{filename}_folder_size` -- total folder size in bytes")
    _bullet(doc, "`size_{filename}` -- duplicate size field for compatibility")

    # 5.4 Report Failure
    _heading(doc, "4. Report Task Failure", 2)
    _table(doc,
        ["Property", "Value"],
        [
            ["Method", "PUT"],
            ["Endpoint", "/detection/update/{task_id}"],
            ["Content-Type", "application/json"],
            ["Timeout", "60 seconds"],
            ["Description", "Reports task failure with error message"],
        ]
    )
    _heading(doc, "Request Payload", 3)
    _code(doc,
        '{\n'
        '    "status": "FAILED",\n'
        '    "error": "Detailed error message describing the failure"\n'
        '}'
    )

    _separator(doc)

    # ── 6. OUTPUT PAYLOAD ──────────────────────────────────────
    _heading(doc, "Output -- Completed Results JSON Schema", 1)
    _body(doc,
        "Upon successful detection processing, the following structured JSON payload is submitted "
        "to the backend API. This represents the complete analytical output of one surveillance task:"
    )
    _code(doc,
        '{\n'
        '    "status": "COMPLETED",\n'
        '    "results": [\n'
        '        {\n'
        '            "id_number": "1",\n'
        '            "class": "person",\n'
        '            "total_duration": 14.50,\n'
        '            "reidentification_count": 1,\n'
        '            "evidence": {\n'
        '                "file_name": "Person_1_sec_10",\n'
        '                "file_size": 245120,\n'
        '                "file_url": "https://pdrlimage.blob.core.windows.net/.../Person_1_sec_10.jpg"\n'
        '            },\n'
        '            "appearances": [\n'
        '                {\n'
        '                    "appearance_id": 1,\n'
        '                    "entry_time": "00:00:02",\n'
        '                    "exit_time": "00:00:16",\n'
        '                    "duration": 14.50,\n'
        '                    "confidence": 0.89,\n'
        '                    "entry_location": {\n'
        '                        "lat": 18.5204312,\n'
        '                        "lon": 73.8567421,\n'
        '                        "alt": 45.20,\n'
        '                        "distance": 10.00,\n'
        '                        "heading": 45.00,\n'
        '                        "direction": "NE"\n'
        '                    },\n'
        '                    "exit_location": {\n'
        '                        "lat": 18.5204620,\n'
        '                        "lon": 73.8567789,\n'
        '                        "alt": 45.20,\n'
        '                        "distance": 10.00,\n'
        '                        "heading": 48.50,\n'
        '                        "direction": "NE"\n'
        '                    },\n'
        '                    "bounding_box": {\n'
        '                        "x": null,\n'
        '                        "y": null,\n'
        '                        "width": null,\n'
        '                        "height": null\n'
        '                    },\n'
        '                    "frame_range": {\n'
        '                        "start": null,\n'
        '                        "end": null\n'
        '                    }\n'
        '                }\n'
        '            ]\n'
        '        }\n'
        '    ]\n'
        '}'
    )

    _separator(doc)

    # ── 7. FIELD DEFINITIONS ───────────────────────────────────
    _heading(doc, "Output Field Definitions", 1)

    _heading(doc, "Top-Level Result Fields", 2)
    _table(doc,
        ["Field", "Type", "Description"],
        [
            ["id_number", "String", "Unique numerical identifier string for the detected entity session. This is the internal person ID (integer) converted to string format (e.g., '1', '2', '3'). NOT the 'Person_001' format -- just the number as a string."],
            ["class", "String", "Detected target class label dynamically resolved from the active detection model's trained class set via model.names[class_id]. Normalized values include: person, vehicle, car, truck, motorbike. Unknown classes pass through unchanged."],
            ["total_duration", "Float", "Cumulative time in seconds the object was observed across all appearances (rounded to 2 decimal places)"],
            ["reidentification_count", "Integer", "Number of times the entity was re-identified after disappearing from frame and re-entering"],
        ]
    )

    _heading(doc, "Evidence Fields", 2)
    _table(doc,
        ["Field", "Type", "Description"],
        [
            ["evidence.file_name", "String", "Formatted evidence filename: {Person_ID}_{timestamp_or_sec_N} (e.g., 'Person_1_sec_10')"],
            ["evidence.file_size", "Integer", "File size of the evidence snapshot in bytes"],
            ["evidence.file_url", "String", "Azure Blob Storage URL to the uploaded snapshot image (public blob path without SAS token)"],
        ]
    )

    _heading(doc, "Appearance Fields", 2)
    _table(doc,
        ["Field", "Type", "Description"],
        [
            ["appearance_id", "Integer", "Sequential appearance counter within the entity's lifecycle (starts at 1)"],
            ["entry_time", "String", "Time when the entity entered the frame, formatted as HH:MM:SS relative to video start"],
            ["exit_time", "String", "Time when the entity exited the frame, formatted as HH:MM:SS relative to video start"],
            ["duration", "Float", "Duration in seconds for this specific appearance segment (rounded to 2 decimals)"],
            ["confidence", "Float", "Detection confidence score from the YOLO model (0.0 - 1.0)"],
        ]
    )

    _heading(doc, "Location Fields (entry_location / exit_location)", 2)
    _table(doc,
        ["Field", "Type", "Description"],
        [
            ["lat", "Float", "Calculated target latitude via spherical geodetic forward projection from drone position (7 decimal places)"],
            ["lon", "Float", "Calculated target longitude via spherical geodetic forward projection from drone position (7 decimal places)"],
            ["alt", "Float", "Drone altitude at the time of detection in meters (2 decimal places)"],
            ["distance", "Float", "Distance in meters between drone and target. Source: obsDist sensor > liveObstacleDistance > default 10m baseline"],
            ["heading", "Float", "Drone yaw/heading angle in degrees (0.0 - 360.0)"],
            ["direction", "String", "8-point cardinal compass direction derived from heading (N, NE, E, SE, S, SW, W, NW)"],
        ]
    )

    _heading(doc, "Reserved Fields (Future Use)", 2)
    _table(doc,
        ["Field", "Type", "Current Value", "Description"],
        [
            ["bounding_box.x/y/width/height", "Integer", "null", "Bounding box coordinates in the video frame (not yet populated)"],
            ["frame_range.start/end", "Integer", "null", "Frame number range for this appearance (not yet populated)"],
        ]
    )

    _separator(doc)

    # ── 8. AZURE UPLOAD ────────────────────────────────────────
    _heading(doc, "Azure Blob Storage -- Evidence Upload Protocol", 1)
    _body(doc,
        "Evidence snapshot images (best detection crops with bounding box overlay) are uploaded "
        "to Azure Blob Storage using Container-level Shared Access Signature (SAS) authorization."
    )

    _heading(doc, "Upload Path Construction", 2)
    _code(doc, "{container_url}/threatimages/{flight_id}/{filename}.jpg?{sas_token}")
    _body(doc, "The SAS URL is parsed into components: container URL (before `?`) and SAS token (after `?`). The blob path `threatimages/{flight_id}/{filename}.jpg` is inserted between them.")

    _heading(doc, "Upload Request", 2)
    _table(doc,
        ["Property", "Value"],
        [
            ["Method", "HTTP PUT"],
            ["Header", "x-ms-blob-type: BlockBlob"],
            ["Body", "Raw image file bytes"],
            ["Timeout", "60 seconds"],
            ["Return Value", "Public blob URL (container_url/threatimages/{flight_id}/{filename}.jpg)"],
        ]
    )

    _heading(doc, "Fallback Behavior", 2)
    _body(doc,
        "If the Azure upload fails for any reason (network error, expired SAS token, etc.), "
        "a fallback URL is generated pointing to the backend server's local file path: "
        "`http://114.143.215.196:5005/AeroMegh/idms/uploads/threatimages/{flight_id}/{filename}.jpg`"
    )

    _separator(doc)

    # ── 9. WORKER BEHAVIOR ─────────────────────────────────────
    _heading(doc, "Autonomous Worker Behavior", 1)

    _heading(doc, "Polling Loop", 2)
    _bullet(doc, "The worker runs an infinite loop, polling `GET /detection/pending` every **5 seconds** when idle.")
    _bullet(doc, "Only one task is processed at a time, enforced by a `threading.Lock()`. If the lock is held, the worker sleeps and retries.")
    _bullet(doc, "When a job is found, the task status is immediately updated to `IN_PROGRESS` before any processing begins.")

    _heading(doc, "Processing Sequence", 2)
    _bullet(doc, "1. Fetch pending job from API")
    _bullet(doc, "2. Update status to `IN_PROGRESS`")
    _bullet(doc, "3. Download video file from `video_url`")
    _bullet(doc, "4. Download telemetry CSVs (if `csv_url` provided) or use in-memory telemetry data")
    _bullet(doc, "5. Resolve and download/cache detection model")
    _bullet(doc, "6. Execute full detection pipeline (`process_video()`)")
    _bullet(doc, "7. Rename evidence files to `{Person_ID}_{timestamp}` format")
    _bullet(doc, "8. Upload evidence images to Azure Blob Storage")
    _bullet(doc, "9. Build structured JSON result payload")
    _bullet(doc, "10. Submit completed results via `PUT /detection/update/{task_id}`")

    _heading(doc, "Error Recovery", 2)
    _bullet(doc, "If any step fails, the task status is updated to `FAILED` with the error message.")
    _bullet(doc, "The worker catches all exceptions, reports failure, and continues polling after a 5-second delay.")
    _bullet(doc, "Evidence file handles are always closed in a `finally` block to prevent resource leaks.")

    _heading(doc, "Request Headers", 2)
    _table(doc,
        ["Header", "Value", "Used For"],
        [
            ["Content-Type", "application/json", "All JSON API requests (status updates, JSON-only results)"],
            ["Content-Type", "(auto-set by requests)", "Multipart form-data uploads (evidence files + results)"],
            ["x-ms-blob-type", "BlockBlob", "Azure Blob Storage PUT uploads"],
        ]
    )

    doc.save(output_path)
    print(f"[OK] Generated: {output_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_detection_pipeline_doc(
        r"d:\deployment_model\Detection_Pipeline_System_Documentation.docx"
    )
    build_api_doc(
        r"d:\deployment_model\API_Integration_Communication_Documentation.docx"
    )
