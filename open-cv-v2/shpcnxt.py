import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

# ======================= CONFIG =======================

CAMERA_INDEX = 0
TARGET_FPS = 10                 # ONE FPS for EVERYTHING
FRAME_TIME = 1.0 / TARGET_FPS

MIN_CONTOUR_AREA = 800

TEMPORAL_BUFFER = 10
TEMPORAL_MIN_VOTES = 6

# ======================= SHAPE CONTEXT =======================

class ShapeContextDescriptor:
    def __init__(self, n_points=100, n_bins_r=5, n_bins_theta=12):
        self.n_points = n_points
        self.n_bins_r = n_bins_r
        self.n_bins_theta = n_bins_theta
        self.dim = n_bins_r * n_bins_theta

    def extract_points(self, img):
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        pts = c.reshape(-1, 2).astype(np.float32)

        if len(pts) < self.n_points:
            return pts

        idx = np.linspace(0, len(pts) - 1, self.n_points, dtype=int)
        return pts[idx]

    def compute(self, pts):
        if pts is None or len(pts) < 2:
            return None

        desc = np.zeros((len(pts), self.dim), dtype=np.float32)

        r_bins = np.logspace(np.log10(0.125), np.log10(2.0), self.n_bins_r + 1)
        t_bins = np.linspace(0, 2 * np.pi, self.n_bins_theta + 1)

        for i, p in enumerate(pts):
            rel = pts - p
            d = np.linalg.norm(rel, axis=1)
            a = (np.arctan2(rel[:, 1], rel[:, 0]) + np.pi)

            mask = d > 0
            d, a = d[mask], a[mask]
            if len(d) == 0:
                continue

            d /= (np.mean(d) + 1e-6)

            hist = np.zeros((self.n_bins_r, self.n_bins_theta))
            for di, ai in zip(d, a):
                rb = np.clip(np.digitize(di, r_bins) - 1, 0, self.n_bins_r - 1)
                tb = np.clip(np.digitize(ai, t_bins) - 1, 0, self.n_bins_theta - 1)
                hist[rb, tb] += 1

            hist /= (np.sum(hist) + 1e-6)
            desc[i] = hist.flatten()

        return desc

    def describe(self, img):
        pts = self.extract_points(img)
        if pts is None:
            return None
        return self.compute(pts)


# ======================= MATCHER =======================

class ShapeContextMatcher:
    @staticmethod
    def chi_square(h1, h2):
        return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))

    def score(self, q_desc, r_desc):
        if q_desc is None or r_desc is None:
            return 0

        good = 0
        for q in q_desc:
            best = min(self.chi_square(q, r) for r in r_desc)
            if best < 2.0:
                good += 1

        return good / max(len(q_desc), len(r_desc))


# ======================= TEMPORAL =======================

class TemporalStabilizer:
    def __init__(self, size, min_votes):
        self.buf = deque(maxlen=size)
        self.min_votes = min_votes

    def update(self, label):
        self.buf.append(label)
        if len(self.buf) < self.min_votes:
            return None

        most, count = Counter(self.buf).most_common(1)[0]
        return most if count >= self.min_votes else None


# ======================= LOAD REFERENCES =======================

def load_references(dir="sym"):
    refs = []
    sc = ShapeContextDescriptor()

    for i, f in enumerate(sorted(Path(dir).glob("*.png"))):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        bin_img = cv2.resize(bin_img, (256, 256))

        desc = sc.describe(bin_img)
        if desc is not None:
            refs.append({
                "id": i,
                "name": f.stem,
                "desc": desc
            })

    print(f"[INFO] Loaded {len(refs)} reference symbols")
    return refs


# ======================= MAIN =======================

def main():
    refs = load_references("sym")
    if not refs:
        print("❌ No reference symbols")
        return

    sc = ShapeContextDescriptor()
    matcher = ShapeContextMatcher()
    stabilizer = TemporalStabilizer(TEMPORAL_BUFFER, TEMPORAL_MIN_VOTES)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Camera error")
        return

    last_time = 0

    print(f"🎥 Running at {TARGET_FPS} FPS (single clock)")

    while True:
        now = time.time()
        if now - last_time < FRAME_TIME:
            continue
        last_time = now

        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        r2 = cv2.inRange(hsv, (170, 70, 50), (179, 255, 255))
        mask = cv2.bitwise_not(cv2.bitwise_or(r1, r2))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = frame.copy()
        detected = None
        best_score = 0

        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.resize(mask[y:y+h, x:x+w], (256, 256))

            q_desc = sc.describe(roi)
            if q_desc is None:
                continue

            for ref in refs:
                score = matcher.score(q_desc, ref["desc"])
                if score > best_score:
                    best_score = score
                    detected = ref["name"]

            if detected:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    output, detected, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                break  # one symbol per frame

        stable = stabilizer.update(detected)
        if stable:
            cv2.rectangle(output, (10, 10), (420, 50), (0, 100, 0), -1)
            cv2.putText(
                output, f"STABLE: {stable}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        cv2.imshow("Shape Context (Single FPS)", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

