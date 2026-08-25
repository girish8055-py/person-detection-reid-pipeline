# 🎯 Person Detection, Re-Identification & VLM Threat Assessment Pipeline
## Master Technical Specification & Architecture Document

---

## 📑 Executive Summary

The **Person Detection, Re-Identification & VLM Threat Assessment Pipeline** is an end-to-end computer vision and artificial intelligence engine designed for autonomous drone surveillance and aerial security monitoring.

The system integrates multi-object detection (custom YOLO), frame-to-frame tracking (ByteTrack), deep appearance re-identification (OSNet), drone telemetry fusion (GPS/Compass logging), local Vision-Language Model threat assessment (Ollama LLaVA), and cloud evidence upload (Azure Blob Storage).

---

## 🏗️ 1. High-Level Architecture & Pipeline Workflow

```
                             +-----------------------------------+
                             |     Input Drone Video / Flight    |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | Frame Decimation Engine (10 FPS)  |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |   Custom YOLO Person Detector     |
                             |      (class_id = 0, conf >= 0.5)  |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | ByteTrack Multi-Object Tracker    |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |  OSNet Re-ID Feature Extractor    |
                             | (512-d L2 Normalized Vector)      |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | Identity Gallery Search & Match   |
                             |   (Cosine Sim Cutoff = 0.68)      |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |  Keyframe Sampling & Exporter     |
                             |  (Entry, Peak, Exit Keyframes)    |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |   Ollama LLaVA VLM Threat Engine  |
                             | (Description, Action, Severity)   |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | Dual-Pass Severity Resolution     |
                             |    (100% Non-Null Guarantee)      |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |  Telemetry GPS & Compass Fusion   |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | Azure Blob Evidence Image Upload  |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |  Final Structured JSON Payload    |
                             +-----------------------------------+
```

---

## 🤖 2. Vision-Language Model (VLM) & Threat Intelligence Engine

### 2.1 Local VLM Architecture
The pipeline calls a local **LLaVA Vision-Language Model** served via Ollama (`http://localhost:11434/api/generate`) using low temperature (`0.2`) for deterministic tactical threat evaluation.

### 2.2 Dual-Pass Non-Null Severity Resolution (`resolve_severity`)
To guarantee that no threat assessment results in `null` severity:
1. **Primary Pass (VLM Visual Analysis)**: The VLM evaluates extracted keyframe crops for weapons (firearms, knives, explosives), hostile posture, tactical gear, and suspicious behavior.
2. **Fallback Pass (Re-ID & Duration Metrics)**: If the VLM model is unavailable, times out, or returns unparseable output, `resolve_severity()` automatically resolves severity based on target duration and re-identification frequency:

| Severity Level | Severity Label | Re-ID Count Range | Duration / Activity Criteria |
| :---: | :---: | :---: | :---: |
| **Level 1** | **LOW** | 0 – 3 Re-IDs | Brief appearance (< 60s), benign civilian movement. |
| **Level 2** | **MEDIUM** | 4 – 6 Re-IDs | Moderate loitering (60s – 120s), repeated sightings. |
| **Level 3** | **HIGH** | 7 – 8 Re-IDs | Prolonged presence (> 120s), tactical/suspicious posture. |
| **Level 4** | **CRITICAL** | 9+ Re-IDs | Visible firearm/weapon, active firing, hostile engagement. |

### 2.3 Tactical Action Directive Expansion (`ensure_detailed_suggested_action`)
Single-word VLM recommendations (e.g. `"monitor"`, `"track"`, `"alert"`, `"detain"`) are automatically post-processed and expanded into complete Army-Chief tactical directives:
- **`MONITOR`** → `"MONITOR — Maintain passive surveillance on Target Target_1 and log movement patterns across area."`
- **`TRACK`** → `"TRACK — Actively track Target Target_1 with drone sensors; observe for loitering or hostile activity."`
- **`ALERT`** → `"ALERT — Raise force readiness level and notify perimeter security units regarding Target Target_1."`
- **`DETAIN`** → `"DETAIN — Dispatch Quick Reaction Force (QRF) to intercept and challenge Target Target_1 immediately."`

---

## 🔍 3. Core Modules & Component Responsibilities

| File Path | Component Name | Functionality & Key Responsibilities |
| :--- | :--- | :--- |
| `app/pipeline/detector.py` | **Person Detector** | Executes YOLO object detection fine-tuned for small aerial targets (`class_id=0`) and runs ByteTrack tracking. |
| `app/pipeline/identity_manager.py` | **Identity Brain** | Computes 512-dimensional OSNet deep feature vectors, manages identity gallery matching (threshold `0.68`), and maintains EMA embedding updates (`alpha=0.2`). |
| `app/pipeline/video_processor.py` | **Pipeline Orchestrator** | Controls 10 FPS frame decimation, coordinate accumulation, keyframe snapshot extraction, and VLM evaluation workflow. |
| `app/pipeline/vlm_verifier.py` | **VLM Verifier** | Performs multi-crop visual classification verification to confirm object classes (`person`, `vehicle`, etc.). |
| `app/pipeline/utils.py` | **Pipeline Utilities** | Implements `resolve_severity()` and `ensure_detailed_suggested_action()`, timestamp formatters, and cardinal heading algorithms. |
| `app/pipeline/telemetry.py` | **GPS Fusion Engine** | Parses drone CSV telemetry logs (`offline_teledata_`), interpolates latitude/longitude/altitude per frame, and computes cardinal headings. |
| `app/pipeline/json_writer.py` | **JSON Output Exporter** | Generates localized structured output files (`metadata.json`) containing complete appearance histories. |

---

## ⚙️ 4. System Hyperparameters & Technical Cutoffs

| Configuration Parameter | Value | Technical Rationale |
| :--- | :--- | :--- |
| **Target Processing FPS** | `10 FPS` | Optimal trade-off between tracking accuracy and real-time processing throughput. |
| **Detection Confidence Cutoff** | `0.50` | Filters out low-confidence background clutter while retaining small aerial targets. |
| **Re-ID Similarity Threshold** | `0.68` | Cosine similarity threshold for identifying re-entering targets across long occlusions. |
| **Re-ID Embedding Dimension** | `512` | L2-normalized feature representation from OSNet deep neural network. |
| **VLM Model Name** | `llava:latest` | Multimodal Vision-Language Model hosted locally on Ollama. |
| **VLM Temperature** | `0.20` | Ensures deterministic and repeatable threat evaluation. |

---

## 📄 5. Sample Output Metadata Schema (`metadata.json`)

```json
{
    "id_number": "Person_001",
    "class": "person",
    "total_duration": 14.5,
    "reidentification_count": 2,
    "severity_level": 3,
    "severity": "HIGH",
    "description": "Individual in dark tactical clothing loitering near perimeter boundary.",
    "suggest_action": "ALERT — Raise force readiness level and notify perimeter security units regarding Target Person_001.",
    "appearances": [
        {
            "appearance_id": 1,
            "entry_time": "00:00:02",
            "exit_time": "00:00:10",
            "duration": 8.0,
            "severity_level": 3,
            "severity": "HIGH",
            "entry_location": {
                "lat": 18.52043,
                "lon": 73.85674,
                "alt": 45.2,
                "distance": 12.4,
                "heading": 45.0,
                "direction": "NE",
                "detection_time": "2026-08-25T11:00:00"
            },
            "exit_location": {
                "lat": 18.52046,
                "lon": 73.85677,
                "alt": 45.2,
                "distance": 12.6,
                "heading": 46.5,
                "direction": "NE",
                "detection_time": "2026-08-25T11:00:08"
            }
        }
    ]
}
```
