import cv2
import pickle
import numpy as np


with open("orb_dataset.pkl", "rb") as f:
    orb_dataset = pickle.load(f)

print(f"Loaded ORB dataset with {len(orb_dataset)} entries")


orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def identify_and_bbox(edges):
    kp_live, des_live = orb.detectAndCompute(edges, None)
    if des_live is None or len(kp_live) == 0:
        return None, 0, None

    best_label = None
    best_score = 0
    best_points = None

    for label, des_ref in orb_dataset.items():
        matches = bf.knnMatch(des_ref, des_live, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > best_score:
            best_score = len(good_matches)
            best_label = label

            # matching orbs
            pts = np.array([
                kp_live[m.trainIdx].pt
                for m in good_matches
            ], dtype=np.int32)

            best_points = pts

    return best_label, best_score, best_points


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Press 'q' to quit")
wl,wh = 16*70,9*70
while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess(frame)
    label, score, pts = identify_and_bbox(edges)

    #bounding box
    if pts is not None and score > 25:
        x, y, w, h = cv2.boundingRect(pts)

        pad = 10
        x = max(x - pad, 0)
        y = max(y - pad, 0)
        w += 2 * pad
        h += 2 * pad

        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{label} ({score})",
            (x, y - 10),
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

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live", wh, wl)
    cv2.imshow("Live", frame)

    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edges", wh, wl)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

