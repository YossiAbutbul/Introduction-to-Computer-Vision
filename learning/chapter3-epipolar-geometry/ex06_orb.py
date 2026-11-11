# ex06_orb.py
# Detect and match ORB features on images 0005-0006 and measure the time.

import os
import time
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

img1 = cv2.imread(os.path.join(script_dir, "0005.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(script_dir, "0006.jpg"), cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise FileNotFoundError("Could not read 0005.jpg or 0006.jpg")

# Start timing: detection + description + matching (as required in Task 1)
start = time.perf_counter()

# ORB detector; default is 500 features, we use 5000 for a fair comparison with SIFT
orb = cv2.ORB_create(nfeatures=5000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# ORB descriptors are binary, so we must use Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
knn_matches = bf.knnMatch(des1, des2, k=2)

# Ratio test, same threshold as with SIFT (0.75)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

elapsed = time.perf_counter() - start

print("keypoints in image 1:", len(kp1))
print("keypoints in image 2:", len(kp2))
print("descriptor shape:", des1.shape, des1.dtype)   # (N, 32) uint8 = 256 bits
print("good matches after ratio test:", len(good_matches))
print("time (detect + describe + match) [s]:", round(elapsed, 3))