import cv2
import os
import pickle
import re


IMAGE_DIR = "imgr"
OUT_FILE = "orb_dataset.pkl"

ORB_FEATURES = 1000



def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def extract_index(filename):
    """
    Extracts numeric index from filenames like 02.png, img_12.jpg, etc.
    """
    m = re.search(r'(\d+)', filename)
    return m.group(1) if m else None


def main():
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    dataset = {}

    for fname in sorted(os.listdir(IMAGE_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        idx = extract_index(fname)
        if idx is None:
            continue

        img_path = os.path.join(IMAGE_DIR, fname)
        img = cv2.imread(img_path)

        if img is None:
            print("⚠️ could not read:", fname)
            continue

        edges = preprocess(img)

        kp, des = orb.detectAndCompute(edges, None)

        if des is None or len(des) == 0:
            print(f"⚠️ no features found in {fname}")
            continue

        dataset[idx] = des
        print(f"✔ {fname}: {len(des)} descriptors")

    with open(OUT_FILE, "wb") as f:
        pickle.dump(dataset, f)

    print("\nORB dataset saved to:", OUT_FILE)
    print("Total symbols:", len(dataset))


if __name__ == "__main__":
    main()

