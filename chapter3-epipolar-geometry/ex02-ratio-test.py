import os
import cv2

# Work relative to this script, so the terminal's folder does not matter
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load both images in grayscale
img1 = cv2.imread(os.path.join(script_dir, "0005.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(script_dir, "0006.jpg"), cv2.IMREAD_GRAYSCALE)

# Stop early with a clear message if a file was not loaded
if img1 is None or img2 is None:
    raise FileNotFoundError("Could not read 0005.jpg or 0006.jpg")

# Detect keypoints and compute descriptors in both images
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Brute-force matcher with L2 distance (suitable for SIFT descriptors)
bf = cv2.BFMatcher(cv2.NORM_L2)

# For each descriptor in image 1, find its 2 nearest neighbors in image 2
knn_matches = bf.knnMatch(des1, des2, k=2)

# Ratio test: keep a match only if the best candidate is clearly better than the second
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

print("keypoints in image 1:", len(kp1))
print("keypoints in image 2:", len(kp2))
print("good matches after ratio test:", len(good_matches))

import numpy as np

# For a sideways camera move, the vertical (v) difference of a correct match should be near 0
dv = [abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) for m in good_matches]

print("median |dv|:", np.median(dv))
print("matches with |dv| < 2 px:", sum(d < 2 for d in dv))

# Draw the first 50 good matches side by side, to inspect them visually
img1_color = cv2.imread(os.path.join(script_dir, "0005.jpg"))
img2_color = cv2.imread(os.path.join(script_dir, "0006.jpg"))
vis = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches[:50], None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(script_dir, "matches_sift.jpg"), vis)