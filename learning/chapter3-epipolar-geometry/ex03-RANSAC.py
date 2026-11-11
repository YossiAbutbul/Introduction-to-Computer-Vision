import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load both images in grayscale
img1 = cv2.imread(os.path.join(script_dir, "0005.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(script_dir, "0006.jpg"), cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not read 0005.jpg or 0006.jpg")

# Detect keypoints and compute SIFT descriptors in both images
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match descriptors with L2 distance and keep matches that pass the ratio test
bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

# Build arrays of matched point coordinates (u, v) for both images
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Approximate intrinsics (assumption, not calibrated yet):
# focal length ~ image width, principal point at the image center
h, w = img1.shape
# Real intrinsics of image 0005 (from the dataset), width 3072, height 2048
K = np.array([[2759.48,    0.0, 1520.69],
              [   0.0, 2764.16, 1006.81],
              [   0.0,    0.0,    1.0]], dtype=np.float64)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                               prob=0.999, threshold=1.0)
n_inliers, R_hat, t_hat, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

print("inliers:", n_inliers)
print("R_hat =\n", R_hat)
print("t_hat =", t_hat.ravel())


# Recover the relative rotation R and translation direction t from E
# Only the inliers from RANSAC are used (mask)
n_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

print("recoverPose inliers:", n_inliers)
print("R =\n", R)
print("t =", t.ravel())