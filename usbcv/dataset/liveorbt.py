import cv2
import sys


IMAGE_PATH = "imgr/test-02.png"  

# preprocessing 
BLUR_KSIZE = (3, 3)
CANNY_LOW = 50
CANNY_HIGH = 150



def preprocess_debug(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return gray, blur, edges


def main():
    img = cv2.imread(IMAGE_PATH)
    wh = 720
    wl = 1280

    if img is None:
        print("❌ Could not read image:", IMAGE_PATH)
        sys.exit(1)

    gray, blur, edges = preprocess_debug(img)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", wh, wl)
    cv2.imshow("Original", img)

    cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale", wh, wl)
    cv2.imshow("Grayscale", gray)

    cv2.namedWindow("Blurred", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Blurred", wh, wl)
    cv2.imshow("Blurred", blur)

    cv2.namedWindow("ORB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ORB", wh, wl)
    cv2.imshow("ORB", edges)

    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

