# 🎯 Person Detection, OSNet AIN Re-Identification & VLM Threat Assessment Pipeline
## Master Technical Specification & Architecture Document

---

## 📑 Executive Summary

The **Person Detection, OSNet AIN Re-Identification & VLM Threat Assessment Pipeline** is an end-to-end computer vision and artificial intelligence engine designed for autonomous drone surveillance, aerial security monitoring, and target re-identification.

The system integrates multi-object detection (custom YOLO), frame-to-frame tracking (ByteTrack), deep appearance re-identification with multi-pose feature banks (OSNet AIN), drone telemetry fusion (GPS/Compass logging), local Vision-Language Model threat assessment (Ollama Qwen2.5-VL / LLaVA), and cloud evidence upload (Azure Blob Storage), with automated post-processing disk cleanup.

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
                             |   Custom YOLO Person/Vehicle/     |
                             |         Drone Detector            |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | ByteTrack Multi-Object Tracker    |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |  OSNet AIN Re-ID Feature Engine   |
                             | (Adaptive Instance Normalization) |
                             +-----------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             | Multi-Embedding Feature Bank      |
                             | (Up to 10 Pose/Angle Keyframes)   |
                             |   (Cosine Sim Cutoff = 0.28)      |
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
                             | Ollama Qwen2.5-VL / LLaVA Engine  |
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
                             | Post-Processing Disk Space        |
                             | Cleanup (Video, CSV, Outputs)     |
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
The pipeline calls a local **Qwen2.5-VL / LLaVA Vision-Language Model** served via Ollama (`http://localhost:11434/api/generate`) using low temperature (`0.2`) for deterministic tactical threat evaluation.
All model weights are stored locally inside the project folder at `ml_pipeline/app/vlm_models/`.

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

---

## 🧬 3. OSNet AIN & Multi-Embedding Feature Bank Engine

### 3.1 Adaptive Instance Normalization (AIN)
The ReID subsystem uses `osnet_ain_x1_0` to normalize out cross-domain variations (lighting shifts, shadows, motion blur, and camera resolution differences). Local weights are saved in `ml_pipeline/app/models/osnet_ain_x1_0_imagenet.pth`.

### 3.2 Multi-Embedding Feature Bank
Instead of storing a single moving-average feature vector per target:
- Each target record stores a **Feature Bank of up to 10 diverse pose/angle keyframe embeddings**.
- When searching the gallery, cosine similarity is computed against **all feature bank embeddings** per candidate target.
- If ANY stored pose clears the threshold (`0.28`), the candidate target is re-identified, eliminating duplicate ID creation when targets turn around or change viewing angle.

---

## 🧹 4. Automated Post-Processing Cleanup

To prevent local disk storage from filling up during continuous live processing on servers:
- Downloaded video files (`.mp4`) and telemetry CSVs are automatically deleted in a `finally` block upon job completion.
- Local output directories (`outputs/{task_id}/`) containing keyframes, annotated videos, and cropped snapshots are cleaned up after evidence is successfully uploaded to Azure Blob Storage and callbacks are pushed to the backend server.
