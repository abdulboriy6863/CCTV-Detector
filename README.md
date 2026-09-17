# CCTV EV Detector — Korean License Plate & EV Classification System
**한국 전기차 충전소 CCTV 번호판 인식 및 전기차/일반차량 자동 판별 시스템**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg?style=flat)](https://ultralytics.com)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR%20(Korean)-blue.svg)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An intelligent, on-premise CCTV surveillance system designed for EV charging stations in South Korea. The system automatically captures video/snapshots from CCTV streams, recognizes Korean vehicle license plates (LPR), distinguishes electric vehicles (EVs) from regular internal combustion engine (ICE) vehicles via spectral color analysis, and generates instant alerts when non-EV vehicles occupy dedicated charging spots.

---

## 🌟 Key Features

- **⚡ 100% Local / On-Premise AI Execution:**
  - Zero external API dependency (No OpenAI / Google Cloud API keys or recurring costs required).
  - High privacy and security compliant with Korean data privacy regulations.
- **🔌 Multi-Layer Parking Session & Cable Occlusion Engine:**
  - **Hardware Ground-Truth:** CSMS charger integration (`TINF_CP_CONNECTOR_STATUS` & `TINF_CURRENT_TX`) locks parking sessions while vehicle is physically plugged in.
  - **Anchor Plate Lock-in:** Preserves pristine initial license plate recognition against vertical charging cable shadows (`47호6633` vs `47오3633`).
  - **Visual Vehicle Presence:** Dual verification with YOLO vehicle bounding box ensures zero premature exit events.
  - **24h+ Accurate Duration:** Calculates continuous multi-day parking time accurately (`1일 3시간`) without duplicate log spam.
- **🔍 Angle-Tolerant & Distant LPR Pipeline:**
  - Automated **minAreaRect deskewing** for diagonal/angled vehicle positions (up to 45° tilt).
  - Multi-scale **Lanczos super-resolution zoom** for distant or small CCTV plates.
  - Collinear pairwise text box merging for fragmented OCR outputs.
- **🎨 Multi-Spectral EV Plate Classifier:**
  - Dual HSV + RGB + CIE-LAB differential analysis targeting Korean sky-blue (`하늘색`) EV plates.
  - Inner core region inspection avoiding vehicle bumper and asphalt color dilution.
- **🛡️ Strict Korean Legal Plate Validator:**
  - Whitelisted against official 40 Korean Hangul license plate characters (`가~하`).
  - Rejects non-car objects (phone numbers, elevator stickers, receipts) to prevent false DB entries.
- **🖥️ Single-Page Interactive Dashboard:**
  - Bilingual interface: **🇺🇿 Uzbek** & **🇰🇷 Korean** with instant language switching.
  - Dark / Light Mode with seamless theme toggling.
  - **Real-Time Stay Duration Ticker:** Dynamic live parking duration counter per table row for ongoing sessions with complete historical duration records.
  - Snapshot history viewer, filter by EV status, live charging badge, interactive license plate photo preview, and graceful image fallback.


---

## 🚀 Architecture Overview

```
[ CCTV Stream / Image Upload ]
              │
              ▼
    1. YOLOv8 Detection  ──► Detects vehicles & prioritizes foreground foreground target
              │
              ▼
    2. Auto-Deskew Engine ──► Extracts plate contour & rotates to horizontal rectangle (<10ms)
              │
              ▼
    3. Deep Learning OCR  ──► Reads Korean Hangul + Numeric digits (EasyOCR / CRAFT)
              │
              ▼
    4. Legal Regex Engine ──► Validates 7/8-digit legal format & filters non-vehicle noise
              │
              ▼
    5. Color Classifier   ──► Multi-spectral HSV/RGB inspection (Blue EV vs White ICE)
              │
              ▼
    6. Decision & Storage ──► MySQL database persistence & instant NON_EV_WARNING alerts
```

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic v2, SQLAlchemy, PyMySQL
- **AI & Computer Vision:** PyTorch, Ultralytics YOLOv8, EasyOCR, OpenCV, NumPy
- **Database:** MySQL / MariaDB
- **Frontend:** Jinja2 Templates, Vanilla JavaScript, Tailwind-inspired CSS

---

## 📦 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/abdulboriy6863/CCTV-Detector.git
cd CCTV-Detector
```

### 2. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your MySQL credentials and server settings
```

### 4. Run the application
```bash
python run.py
```
Open your browser at `http://localhost:8000` to access the Dashboard.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/detect/upload` | Upload an image for real-time LPR & EV classification |
| `GET` | `/api/v1/vehicles` | List vehicle detection history with duration and ongoing flags |
| `GET` | `/api/v1/vehicles/{id}/image` | Retrieve snapshot with automatic Base64 DB & session fallback |
| `GET` | `/api/v1/vehicles/{id}/plate-image` | Retrieve cropped plate thumbnail directly from database |
| `GET` | `/api/v1/vehicles/{id}/download` | Download vehicle snapshot with safe attachment headers |
| `GET` | `/api/v1/vehicles/export` | Export vehicle logs as CSV formatted in Korean |
| `GET` | `/api/v1/cameras` | List all registered CCTV camera connections |
| `POST` | `/api/v1/cameras` | Register a new CCTV camera (RTSP / HTTP snapshot) |
| `GET` | `/api/v1/cameras/slots-status` | Real-time charger slot status, telemetry & violation HUD |
| `POST` | `/api/v1/cameras/{id}/snapshot` | Capture live test frame from camera |
| `GET` | `/api/v1/stats/summary` | Retrieve summary analytics for today and total stats |
| `GET` | `/health` | System health check endpoint |

---

## 💾 Storage & Base64 Fallback Architecture

To ensure zero image preview failures even across distributed server environments:
1. **Vehicle Entry (`START`):** Captures high-resolution frame, extracts tightly cropped license plate, and encodes JPEG Base64 (<25KB) directly into the `plate_region_image` database column.
2. **Vehicle Departure (`END`):** Applies Zero-Image Exit strategy (no new disk files created), linking directly to the entry session's snapshot.
3. **Multi-Tier Image Fallback:** The `/image` and `/plate-image` endpoints seamlessly serve:
   - Primary: Physical file on local storage disk.
   - Secondary: Embedded Base64 JPEG data decoded on the fly.
   - Tertiary: Paired `START` session record for departed vehicles.

---

## 📄 License
This project is licensed under the MIT License.
