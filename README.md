# intelligent-Parking-Space-Occupancy-Detection-System

## Overview
This project implements a **robust parking space occupancy detection system** using classical computer vision with **temporal state modeling**.  
It is designed for **top-down / fixed parking cameras** and correctly distinguishes **parked vehicles from road markings (arrows, lines, symbols)**, avoiding false positives commonly seen in naive approaches.

The system displays **occupied slots, available slots, and total slot count directly on a live OpenCV window** and is suitable for **edge deployment** (Jetson / CPU).

---

## Key Features
- Slot-wise parking occupancy detection
- Robust to painted arrows and road markings
- Temporal belief-based state estimation (no flickering)
- Real-time OpenCV visualization
- No deep learning dependency
- Edge-friendly and lightweight

---

## Methodology
1. **Slot Calibration**  
   Parking slots are manually marked once and stored as polygons.

2. **Background Modeling (Per Slot)**  
   Each slot learns its empty appearance over initial frames.

3. **Vehicle Presence Gating**  
   Occupancy is detected only when vehicle-like visual characteristics are present, preventing false positives from paint markings.

4. **Temporal Belief Update**  
   Occupancy is estimated as a probability over time for stable decisions.

5. **Visualization**  
   Slot states and occupancy counts are rendered directly on the OpenCV window.

---

## Slot State Visualization
| Color  | Meaning |
|-------|---------|
| Green | Slot is free |
| Red   | Slot is occupied |
| Yellow| Transitional / uncertain (entry or exit) |

---

## Project Files
- `parking_foundation_cv.py` – Main parking occupancy detection code  
- `slot_calibration.py` – One-time slot marking tool  
- `slots.json` – Saved parking slot geometry  
- `README.md`

---

## Requirements
- Python 3.8+
- OpenCV
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
