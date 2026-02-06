import cv2
import json
import time
import numpy as np

# ---------------- CONFIG ----------------
CAMERA_ID = "/home/s186/Videos/space/12125602_3840_2160_30fps.mp4"
SLOTS_FILE = "slots.json"

ALPHA = 0.85                 # temporal smoothing
OCCUPIED_TH = 0.65
FREE_TH = 0.35

# ----------------------------------------

# Load parking slots
with open(SLOTS_FILE, "r") as f:
    slots = [np.array(s, dtype=np.int32) for s in json.load(f)]

NUM_SLOTS = len(slots)

# Slot belief state (foundation-style)
belief = np.zeros(NUM_SLOTS)        # P(occupied)
stability = np.zeros(NUM_SLOTS)     # seconds
last_state = ["FREE"] * NUM_SLOTS

cap = cv2.VideoCapture(CAMERA_ID)
prev_time = time.time()

def slot_evidence(frame, poly):
    """Appearance-change evidence (lighting robust)"""
    x, y, w, h = cv2.boundingRect(poly)
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 120)

    score = np.count_nonzero(edges) / edges.size
    return min(score * 3.0, 1.0)   # normalized evidence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    occupied_count = 0

    for i, poly in enumerate(slots):
        evidence = slot_evidence(frame, poly)

        # Temporal Bayesian update
        belief[i] = ALPHA * belief[i] + (1 - ALPHA) * evidence

        # State decision with hysteresis
        if belief[i] > OCCUPIED_TH:
            state = "OCCUPIED"
            color = (0, 0, 255)
            occupied_count += 1
        elif belief[i] < FREE_TH:
            state = "FREE"
            color = (0, 255, 0)
        else:
            state = "UNCERTAIN"
            color = (0, 255, 255)

        # Stability tracking
        if state == last_state[i]:
            stability[i] += dt
        else:
            stability[i] = 0.0
            last_state[i] = state

        thickness = 2 if stability[i] < 2.0 else 4
        cv2.polylines(frame, [poly], True, color, thickness)

    total_slots = NUM_SLOTS
    free_slots = total_slots - occupied_count

    # ---------------- UI OVERLAY ----------------
    cv2.rectangle(frame, (10,10), (380,120), (25,25,25), -1)

    cv2.putText(frame, f"Total Slots     : {total_slots}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.putText(frame, f"Occupied Slots  : {occupied_count}",
                (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(frame, f"Available Slots : {free_slots}",
                (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Foundation Parking Occupancy (Jetson)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
