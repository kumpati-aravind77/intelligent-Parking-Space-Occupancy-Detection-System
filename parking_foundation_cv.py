import cv2
import json
import time
import numpy as np

# ================= CONFIG =================
VIDEO_PATH = "/home/s186/Videos/space/12125602_3840_2160_30fps.mp4"
SLOTS_FILE = "slots.json"

BG_FRAMES = 40
ALPHA = 0.92

OCC_TH = 0.55
FREE_TH = 0.25

# vehicle appearance constraint
DARK_RATIO_TH = 0.18   # <-- KEY FIX
# ==========================================

# -------- Load slots --------
with open(SLOTS_FILE, "r") as f:
    slots = [np.array(s, dtype=np.int32) for s in json.load(f)]

N = len(slots)

belief = np.zeros(N)
stability = np.zeros(N)
last_state = ["FREE"] * N

cap = cv2.VideoCapture(VIDEO_PATH)

bg_models = [None] * N
bg_count = 0
prev_time = time.time()

def extract_gray_roi(frame, poly):
    x, y, w, h = cv2.boundingRect(poly)
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def vehicle_likelihood(roi):
    """
    Cars have darker, dense regions.
    Painted arrows are bright and sparse.
    """
    _, bin_dark = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.count_nonzero(bin_dark) / bin_dark.size
    return dark_ratio

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    occupied_count = 0

    for i, poly in enumerate(slots):
        roi = extract_gray_roi(frame, poly)
        if roi is None:
            continue

        # -------- Background learning --------
        if bg_count < BG_FRAMES:
            if bg_models[i] is None:
                bg_models[i] = roi.astype(np.float32)
            else:
                bg_models[i] = 0.95 * bg_models[i] + 0.05 * roi
            continue

        bg = bg_models[i].astype(np.uint8)

        # -------- Evidence (difference) --------
        diff = cv2.absdiff(roi, bg)
        diff_score = np.mean(diff) / 255.0

        # -------- Vehicle likelihood (CRITICAL) --------
        dark_ratio = vehicle_likelihood(roi)

        # Combine evidence correctly
        if dark_ratio < DARK_RATIO_TH:
            evidence = 0.0   # NO vehicle, force free
        else:
            evidence = min(diff_score * 3.0, 1.0)

        # -------- Temporal belief --------
        belief[i] = ALPHA * belief[i] + (1 - ALPHA) * evidence

        # -------- State decision --------
        if belief[i] > OCC_TH:
            state = "OCCUPIED"
            color = (0, 0, 255)
            occupied_count += 1
        elif belief[i] < FREE_TH:
            state = "FREE"
            color = (0, 255, 0)
        else:
            state = "UNCERTAIN"
            color = (0, 255, 255)

        if state == last_state[i]:
            stability[i] += dt
        else:
            stability[i] = 0.0
            last_state[i] = state

        thickness = 2 if stability[i] < 2 else 4
        cv2.polylines(frame, [poly], True, color, thickness)

    # -------- Visualization --------
    if bg_count < BG_FRAMES:
        cv2.putText(frame, "Learning background...",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        bg_count += 1
    else:
        total_slots = N
        free_slots = total_slots - occupied_count

        cv2.rectangle(frame, (10,10), (380,120), (30,30,30), -1)
        cv2.putText(frame, f"Total Slots     : {total_slots}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
        cv2.putText(frame, f"Occupied Slots  : {occupied_count}",
                    (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Available Slots : {free_slots}",
                    (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Parking Space Occupancy Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
