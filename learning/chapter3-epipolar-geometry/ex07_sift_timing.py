# ex07_sift_timing.py
# Same pipeline as ex06_orb.py, but with SIFT, to compare running time.

import os
import time
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))

img1 = cv2.imread(os.path.join(script_dir, "0005.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(script_dir, "0006.jpg"), cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise FileNotFoundError("Could not read 0005.jpg or 0006.jpg")

# Start timing: detection + description + matching
start = time.perf_counter()

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# SIFT descriptors are real-valued, so we use L2 distance
bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

elapsed = time.perf_counter() - start

print("keypoints in image 1:", len(kp1))
print("good matches after ratio test:", len(good_matches))
print("time (detect + describe + match) [s]:", round(elapsed, 3))