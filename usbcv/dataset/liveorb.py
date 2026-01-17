import cv2
import pickle
import numpy as np

# ds load

with open("orb_dataset.pkl", "rb") as f:
    orb_dataset = pickle.load(f)

print("Loaded dataset with", len(orb_dataset), "symbols")


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
        return None, 0

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

    return best_label, best_score

# live stuff

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not found")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess(frame)
    label, score = identify(edges)

    if score > 5:   # threshold (tune once)
        text = f"Detected: {label} ({score})"
        color = (0, 255, 0)
    else:
        text = "Unknown"
        color = (0, 0, 255)

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

