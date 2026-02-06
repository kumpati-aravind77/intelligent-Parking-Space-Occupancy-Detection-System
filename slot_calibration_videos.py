import cv2
import json
import numpy as np

VIDEO_SOURCE = "/home/s186/Videos/space/12125602_3840_2160_30fps.mp4"
OUTPUT_FILE = "slots.json"

slots = []
current = []

def mouse_cb(event, x, y, flags, param):
    global current, slots
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x, y))
        if len(current) == 4:
            slots.append(current.copy())
            print(f"[INFO] Slot {len(slots)} added")
            current.clear()

cap = cv2.VideoCapture(VIDEO_SOURCE)
cv2.namedWindow("Slot Calibration")
cv2.setMouseCallback("Slot Calibration", mouse_cb)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for s in slots:
        cv2.polylines(frame, [np.array(s)], True, (0,255,0), 2)

    for p in current:
        cv2.circle(frame, p, 4, (0,0,255), -1)

    cv2.putText(
        frame,
        "Click 4 points per slot | Press S to save | Q to quit",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow("Slot Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        with open(OUTPUT_FILE, "w") as f:
            json.dump(slots, f)
        print(f"[INFO] Saved {len(slots)} slots to slots.json")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
