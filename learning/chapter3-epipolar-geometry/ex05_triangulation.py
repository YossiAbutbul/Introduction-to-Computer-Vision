# ex05_triangulation.py
# Triangulate 3D points from two views of Fountain-P11 (images 0005, 0006),
# using the relative pose estimated from SIFT matches.

import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Load images and find SIFT matches (same as ex04)
# ---------------------------------------------------------------------------
img1 = cv2.imread(os.path.join(script_dir, "0005.jpg"), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(script_dir, "0006.jpg"), cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise FileNotFoundError("Could not read 0005.jpg or 0006.jpg")

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# ---------------------------------------------------------------------------
# 2. Intrinsics and real baseline from the dataset
# ---------------------------------------------------------------------------
K = np.array([[2759.48,    0.0, 1520.69],
              [   0.0, 2764.16, 1006.81],
              [   0.0,    0.0,    1.0]], dtype=np.float64)

C1 = np.array([-14.1604, -3.32084, 0.0862032])
C2 = np.array([-15.8818, -3.15083, 0.0592619])
baseline = np.linalg.norm(C2 - C1)   # real distance between camera centers

# ---------------------------------------------------------------------------
# 3. Estimate relative pose (R_hat, t_hat) with RANSAC
# ---------------------------------------------------------------------------
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                               prob=0.999, threshold=1.0)
_, R_hat, t_hat, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
t_hat = t_hat.ravel()

# Keep only RANSAC inliers
inl = mask.ravel() > 0
p1 = pts1[inl]
p2 = pts2[inl]

# ---------------------------------------------------------------------------
# 4. Projection matrices: camera 1 is the world origin
# ---------------------------------------------------------------------------
s = baseline / np.linalg.norm(t_hat)   # scale factor, t_hat has length 1
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = K @ np.hstack([R_hat, (s * t_hat).reshape(3, 1)])

# ---------------------------------------------------------------------------
# 5. Triangulate: result is 4 x N homogeneous points
# ---------------------------------------------------------------------------
X_h = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))

# Convert to Cartesian: divide by the 4th component -> N x 3
X = (X_h[:3] / X_h[3]).T

# ---------------------------------------------------------------------------
# 6. Filter invalid points: finite values and positive depth in both cameras
# ---------------------------------------------------------------------------
z1 = X[:, 2]                                   # depth in camera 1
z2 = (X @ R_hat.T + s * t_hat)[:, 2]           # depth in camera 2
valid = np.isfinite(X).all(axis=1) & (z1 > 0) & (z2 > 0)
X_valid = X[valid]

print("scale s:", round(s, 4))
print("inlier matches:", len(p1))
print("valid 3D points:", len(X_valid))
print("median depth (camera 1):", round(np.median(X_valid[:, 2]), 3))
print("first 3 points:\n", X_valid[:3])


# ---------------------------------------------------------------------------
# 7. Reprojection error: project 3D points back into both images
# ---------------------------------------------------------------------------
# 3D points in homogeneous coordinates: 4 x N
Xh_valid = np.hstack([X_valid, np.ones((len(X_valid), 1))]).T

# Project with P1 and P2, then divide by the last element -> pixel coordinates
x1_proj = P1 @ Xh_valid
x1_proj = (x1_proj[:2] / x1_proj[2]).T
x2_proj = P2 @ Xh_valid
x2_proj = (x2_proj[:2] / x2_proj[2]).T

# Original pixel coordinates of the valid points
x1_orig = p1[valid]
x2_orig = p2[valid]

# e = sqrt((|x1 - x1_hat|^2 + |x2 - x2_hat|^2) / 2), per point
err = np.sqrt((np.sum((x1_orig - x1_proj) ** 2, axis=1) +
               np.sum((x2_orig - x2_proj) ** 2, axis=1)) / 2)

print("reprojection error mean (px):", round(err.mean(), 3))
print("reprojection error median (px):", round(np.median(err), 3))
print("max reprojection error (px):", round(err.max(), 3))


# ---------------------------------------------------------------------------
# 8. Visualize the 3D point cloud and the two camera centers
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Camera centers in camera-1 coordinates: C1 = origin, C2 = -R^T (s t)
cam1 = np.zeros(3)
cam2 = -R_hat.T @ (s * t_hat)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(projection="3d")

# Plot X (right), Z (depth), -Y (up), so the scene is not upside down
ax.scatter(X_valid[:, 0], X_valid[:, 2], -X_valid[:, 1], s=1, c="steelblue")
ax.scatter([cam1[0], cam2[0]], [cam1[2], cam2[2]], [-cam1[1], -cam2[1]],
           s=60, c="red", marker="^", label="cameras")

ax.set_xlabel("X (right)")
ax.set_ylabel("Z (depth)")
ax.set_zlabel("-Y (up)")
ax.set_title("Triangulated point cloud (0005-0006, SIFT)")
ax.legend()

plt.savefig(os.path.join(script_dir, "point_cloud_sift.png"), dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 9. Top-down view (X vs Z): the wall should appear as a line
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(X_valid[:, 0], X_valid[:, 2], s=1, c="steelblue")
plt.scatter([cam1[0], cam2[0]], [cam1[2], cam2[2]],
            s=60, c="red", marker="^", label="cameras")

plt.axis("equal")          # same scale on both axes, so shapes are not distorted
plt.xlabel("X (right)")
plt.ylabel("Z (depth)")
plt.title("Top-down view of the point cloud")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(script_dir, "point_cloud_top_view.png"), dpi=150)
plt.show()