import cv2
import os
import re
import pickle
import numpy as np
from collections import deque

# ===================== CONFIG =====================

DATASET_DIR = "imgr"   # contains test-20.png etc
ORB_DATASET_PKL = "orb_dataset.pkl"

MIN_ORB_SCORE = 25
MAX_HU_DIST = 2.5

CONFIRM_FRAMES = 5
HOLD_FRAMES = 10
BBOX_SMOOTH_FRAMES = 5

# ===================== HELPERS =====================

def extract_label(filename):
    """
    Extract numeric label from filenames like:
    test-20.png → '20'
    foo_003.jpg → '003'
    """
    m = re.search(r'(\d+)', filename)
    return m.group(1) if m else None

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# ===================== LOAD ORB DATASET =====================

with open(ORB_DATASET_PKL, "rb") as f:
    orb_dataset = pickle.load(f)

print(f"[INFO] Loaded ORB dataset with {len(orb_dataset)} labels")

# ===================== BUILD HU DATASET =====================

hu_dataset = {}

for fname in os.listdir(DATASET_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    label = extract_label(fname)
    if label is None or label not in orb_dataset:
        continue

    path = os.path.join(DATASET_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"[WARN] Could not read {path}")
        continue

    edges = preprocess(img)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print(f"[WARN] No contours in {fname}")
        continue

    c = max(cnts, key=cv2.contourArea)

    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    hu_dataset[label] = hu
    print(f"[OK] Hu moments for label {label}")

print(f"[INFO] Hu dataset ready with {len(hu_dataset)} labels")

# ===================== ORB + MATCHER =====================

orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# ===================== TEMPORAL STATE =====================

label_history = deque(maxlen=CONFIRM_FRAMES)
bbox_history = deque(maxlen=BBOX_SMOOTH_FRAMES)

stable_label = None
hold_counter = 0

# ===================== IDENTIFICATION =====================

def identify_fused(edges):
    kp_live, des_live = orb.detectAndCompute(edges, None)
    if des_live is None or len(kp_live) == 0:
        return None, None

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    c_live = max(cnts, key=cv2.contourArea)
    hu_live = cv2.HuMoments(cv2.moments(c_live)).flatten()
    hu_live = -np.sign(hu_live) * np.log10(np.abs(hu_live) + 1e-10)

    best_label = None
    best_pts = None
    best_score = -1

    for label, des_ref in orb_dataset.items():
        if label not in hu_dataset:
            continue

        matches = bf.knnMatch(des_ref, des_live, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < MIN_ORB_SCORE:
            continue

        hu_dist = np.linalg.norm(hu_live - hu_dataset[label])
        if hu_dist > MAX_HU_DIST:
            continue

        fused_score = len(good) - (hu_dist * 10)

        if fused_score > best_score:
            best_score = fused_score
            best_label = label
            best_pts = np.array(
                [kp_live[m.trainIdx].pt for m in good],
                dtype=np.int32
            )

    return best_label, best_pts

# ===================== CAMERA LOOP =====================

cap = cv2.VideoCapture(4)
if not cap.isOpened():
    raise RuntimeError("Camera not available")

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess(frame)
    label, pts = identify_fused(edges)

    # -------- Temporal confirmation --------
    label_history.append(label)

    if len(label_history) == CONFIRM_FRAMES:
        candidate = max(set(label_history), key=label_history.count)
        if candidate is not None and label_history.count(candidate) >= CONFIRM_FRAMES - 1:
            stable_label = candidate
            hold_counter = HOLD_FRAMES

    if hold_counter > 0:
        hold_counter -= 1
    else:
        stable_label = None
        bbox_history.clear()

    # -------- Draw bbox --------
    if stable_label and pts is not None:
        x, y, w, h = cv2.boundingRect(pts)
        bbox_history.append((x, y, w, h))

        bx = int(sum(b[0] for b in bbox_history) / len(bbox_history))
        by = int(sum(b[1] for b in bbox_history) / len(bbox_history))
        bw = int(sum(b[2] for b in bbox_history) / len(bbox_history))
        bh = int(sum(b[3] for b in bbox_history) / len(bbox_history))

        pad = 10
        cv2.rectangle(
            frame,
            (bx - pad, by - pad),
            (bx + bw + pad, by + bh + pad),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Detected: {stable_label}",
            (bx, by - 12),
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

    cv2.imshow("Live", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

