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


def identify(edges):
    kp_live, des_live = orb.detectAndCompute(edges, None)
    if des_live is None:
        return None, 0, kp_live

    best_label = None
    best_score = 0

    for label, des_ref in orb_dataset.items():
        matches = bf.knnMatch(des_ref, des_live, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        score = len(good)

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score, kp_live


cap = cv2.VideoCapture(4)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess(frame)
    label, score, kp = identify(edges)

    if score > 25:
        text = f"Detected: {label}  (score={score})"
        color = (0, 255, 0)
    else:
        text = "Unknown"
        color = (0, 0, 255)

    cv2.putText(
        frame, text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    kp_img = cv2.drawKeypoints(
        edges, kp, None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )

    def show(name, img, w=16*80, h=9*80):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w, h)
        cv2.imshow(name, img)

    show("Live", frame)
    show("Edges", edges)
    show("ORB Keypoints", kp_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

