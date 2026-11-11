# ex04_pose_vs_ground_truth.py
# Estimate the relative pose between two images of the Fountain-P11 dataset
# and compare it with the ground truth from the dataset.

import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Load the two images in grayscale
# ---------------------------------------------------------------------------
img1 = cv2.imread(os.path.join(script_dir, "0005.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(script_dir, "0006.jpg"), cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not read 0005.jpg or 0006.jpg")

# ---------------------------------------------------------------------------
# 2. Detect SIFT keypoints and descriptors in both images
# ---------------------------------------------------------------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# ---------------------------------------------------------------------------
# 3. Match descriptors (L2 distance) and apply the Ratio Test (threshold 0.75)
# ---------------------------------------------------------------------------
bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

print("good matches after ratio test:", len(good_matches))

# ---------------------------------------------------------------------------
# 4. Real intrinsic matrix K from the dataset (same camera for both images)
# ---------------------------------------------------------------------------
K = np.array([[2759.48,    0.0, 1520.69],
              [   0.0, 2764.16, 1006.81],
              [   0.0,    0.0,    1.0]], dtype=np.float64)

# ---------------------------------------------------------------------------
# 5. Ground-truth extrinsics from the dataset
#    The stored rotations are camera -> world, so we transpose them
#    to get world -> camera (the convention used in the assignment formulas).
# ---------------------------------------------------------------------------
R1_stored = np.array([[ 0.962742, -0.0160548, -0.269944],
                      [-0.270399, -0.0444283, -0.961723],
                      [ 0.00344709, 0.998884, -0.0471142]])

R2_stored = np.array([[ 0.994915, -0.00462005, -0.100616],
                      [-0.100715, -0.0339759, -0.994335],
                      [ 0.00117536, 0.999412, -0.0342684]])

# Camera centers in world coordinates
C1 = np.array([-14.1604, -3.32084, 0.0862032])
C2 = np.array([-15.8818, -3.15083, 0.0592619])

R1 = R1_stored.T   # world -> camera 1
R2 = R2_stored.T   # world -> camera 2

# Relative pose from camera 1 to camera 2:
#   X_c2 = R2 (X_w - C2),  X_w = R1^T X_c1 + C1
#   => R_gt = R2 R1^T,  t_gt = R2 (C1 - C2)
R_gt = R2 @ R1.T
t_gt = R2 @ (C1 - C2)
t_gt_dir = t_gt / np.linalg.norm(t_gt)

# Baseline: distance between the two camera centers (real scale)
baseline = np.linalg.norm(C2 - C1)

# ---------------------------------------------------------------------------
# 6. Estimate E with RANSAC, then recover R and t from E
# ---------------------------------------------------------------------------
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                               prob=0.999, threshold=1.0)

n_inliers, R_hat, t_hat, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
t_hat = t_hat.ravel()   # unit-length direction, scale is unknown

# ---------------------------------------------------------------------------
# 7. Compare with ground truth
# ---------------------------------------------------------------------------
# Rotation error: e_R = arccos((trace(R_hat R_gt^T) - 1) / 2)
cos_eR = np.clip((np.trace(R_hat @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
e_R = np.degrees(np.arccos(cos_eR))

# Translation direction error: e_t = arccos(t_hat . t_gt / (|t_hat| |t_gt|))
cos_et = np.clip(np.dot(t_hat, t_gt_dir) / np.linalg.norm(t_hat), -1.0, 1.0)
e_t = np.degrees(np.arccos(cos_et))

# ---------------------------------------------------------------------------
# 8. Print the results
# ---------------------------------------------------------------------------
print("RANSAC + recoverPose inliers:", n_inliers, "of", len(good_matches))
print("R_hat =\n", R_hat)
print("t_hat =", t_hat)
print("R_gt =\n", R_gt)
print("t_gt direction =", t_gt_dir)
print("e_R (deg):", round(e_R, 3))
print("e_t (deg):", round(e_t, 3))
print("baseline (real scale):", round(baseline, 4))