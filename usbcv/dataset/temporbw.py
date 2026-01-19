import cv2
import pickle
import numpy as np
from collections import deque

# ===================== LOAD DATASET =====================

with open("orb_dataset.pkl", "rb") as f:
    orb_dataset = pickle.load(f)

print(f"[INFO] Loaded ORB dataset with {len(orb_dataset)} entries")

# ===================== ORB + MATCHER ====================

orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# ===================== PARAMETERS =======================

MIN_SCORE = 25              # minimum ORB matches
CONFIRM_FRAMES = 5          # frames needed to confirm detection
HOLD_FRAMES = 10            # frames to keep detection alive
BBOX_SMOOTH_FRAMES = 5      # frames to smooth bbox

# ===================== STATE ============================

label_history = deque(maxlen=CONFIRM_FRAMES)
bbox_history = deque(maxlen=BBOX_SMOOTH_FRAMES)

stable_label = None
hold_counter = 0

# ===================== PREPROCESS =======================

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# ===================== ORB MATCH + POINTS ================

def identify_and_points(edges):
    kp_live, des_live = orb.detectAndCompute(edges, None)
    if des_live is None or len(kp_live) == 0:
        return None, 0, None

    best_label = None
    best_score = 0
    best_points = None

    for label, des_ref in orb_dataset.items():
        matches = bf.knnMatch(des_ref, des_live, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > best_score:
            best_score = len(good)
            best_label = label
            best_points = np.array(
                [kp_live[m.trainIdx].pt for m in good],
                dtype=np.int32
            )

    return best_label, best_score, best_points

# ===================== CAMERA ============================

cap = cv2.VideoCapture(4)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera")

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess(frame)
    label, score, pts = identify_and_points(edges)

    # ---------- TEMPORAL CONFIRMATION ----------
    if score >= MIN_SCORE and pts is not None:
        label_history.append(label)
    else:
        label_history.append(None)

    if len(label_history) == CONFIRM_FRAMES:
        candidate = max(set(label_history), key=label_history.count)
        if candidate is not None and label_history.count(candidate) >= CONFIRM_FRAMES - 1:
            stable_label = candidate
            hold_counter = HOLD_FRAMES

    # ---------- HOLD LOGIC ----------
    if hold_counter > 0:
        hold_counter -= 1
    else:
        stable_label = None
        bbox_history.clear()

    # ---------- DRAW BOUNDING BOX ----------
    if stable_label is not None and pts is not None:
        x, y, w, h = cv2.boundingRect(pts)
        bbox_history.append((x, y, w, h))

        avg_x = int(sum(b[0] for b in bbox_history) / len(bbox_history))
        avg_y = int(sum(b[1] for b in bbox_history) / len(bbox_history))
        avg_w = int(sum(b[2] for b in bbox_history) / len(bbox_history))
        avg_h = int(sum(b[3] for b in bbox_history) / len(bbox_history))

        pad = 10
        avg_x = max(avg_x - pad, 0)
        avg_y = max(avg_y - pad, 0)
        avg_w += 2 * pad
        avg_h += 2 * pad

        cv2.rectangle(
            frame,
            (avg_x, avg_y),
            (avg_x + avg_w, avg_y + avg_h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{stable_label}",
            (avg_x, avg_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "Unknown",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ---------- DISPLAY ----------
    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live", 640, 480)
    cv2.imshow("Live", frame)

    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edges", 640, 480)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

