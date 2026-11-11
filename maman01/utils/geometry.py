"""
geometry.py
Two-view geometry: ground-truth relative pose and F, Sampson distance,
pose estimation from matches and pose errors.
Used in tasks 1, 2 and 3.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TAU, RANSAC_PROB, RANSAC_THRESH
from .dataset import load_camera


# ----------------------------------------------------------------------------
# Ground-truth relative geometry
# ----------------------------------------------------------------------------

def skew(t):
    """[t]_x, so that skew(t) @ v == cross(t, v)."""
    tx, ty, tz = np.asarray(t, dtype=np.float64).ravel()
    return np.array([[0.0, -tz, ty],
                     [tz, 0.0, -tx],
                     [-ty, tx, 0.0]])


def rotation_angle_deg(R):
    """Angle of a rotation matrix: arccos((trace(R) - 1) / 2), in degrees."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def relative_pose(cam1, cam2):
    """
    Relative pose camera 1 -> camera 2:
        R = R2 R1^T
        t = t2 - R t1 = R2 (C1 - C2)
    Returns R_gt, t_gt, baseline = |C2 - C1|.

    The assert compares the two forms of t. The tolerance is 1e-3 and not
    smaller because R in the files has only 6 digits (R R^T differs from I by
    ~1e-6), so the two forms differ by ~4e-6.
    """
    R_gt = cam2.R @ cam1.R.T
    t_gt = cam2.R @ (cam1.C - cam2.C)
    assert np.allclose(t_gt, cam2.t - R_gt @ cam1.t, atol=1e-3), \
        "the two forms of t_gt do not agree"
    baseline = float(np.linalg.norm(cam2.C - cam1.C))
    return R_gt, t_gt, baseline


def essential_from_cameras(cam1, cam2):
    """E = [t]_x R"""
    R_gt, t_gt, _ = relative_pose(cam1, cam2)
    return skew(t_gt) @ R_gt


def fundamental_from_cameras(cam1, cam2, normalize=True):
    """
    F_GT = K2^-T E K1^-1
    F is defined only up to scale. Dividing by the norm does not change the
    Sampson distance, it only makes the printed numbers readable.
    """
    E_gt = essential_from_cameras(cam1, cam2)
    F_gt = np.linalg.inv(cam2.K).T @ E_gt @ np.linalg.inv(cam1.K)
    if normalize:
        F_gt = F_gt / np.linalg.norm(F_gt)
    return F_gt


def gt_pair_geometry(idx1, idx2):
    """All ground-truth values for one pair, in a dict."""
    cam1, cam2 = load_camera(idx1), load_camera(idx2)
    R_gt, t_gt, baseline = relative_pose(cam1, cam2)
    return {
        "cam1": cam1,
        "cam2": cam2,
        "R_gt": R_gt,
        "t_gt": t_gt,
        "baseline": baseline,
        "rot_angle_deg": rotation_angle_deg(R_gt),
        "F_gt": fundamental_from_cameras(cam1, cam2),
    }


# ----------------------------------------------------------------------------
# Checking matches against F_GT
# ----------------------------------------------------------------------------

def to_homogeneous(pts):
    """(N,2) -> (N,3), adds a column of ones."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.hstack([pts, np.ones((len(pts), 1))])


def sampson_distance(pts1, pts2, F):
    """
    Sampson distance for each match:
        d = (x2^T F x1)^2 / ((Fx1)_1^2 + (Fx1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2)
    Units are px^2.
    """
    x1 = to_homogeneous(pts1)
    x2 = to_homogeneous(pts2)
    Fx1 = x1 @ F.T          # row i = F x1_i
    Ftx2 = x2 @ F           # row i = F^T x2_i
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den > 0, num / den, np.inf)
    return d


def geometric_consistency(pts1, pts2, F_gt, tau=TAU):
    """
    A match is consistent if d_S < tau^2.
    Returns mask, N_inliers, r_inliers = N_inliers / N, and all distances.
    A consistent match is not necessarily a correct match - a wrong match can
    still lie close to the epipolar line.
    """
    d = sampson_distance(pts1, pts2, F_gt)
    mask = d < tau ** 2
    n = int(mask.sum())
    total = len(d)
    return mask, n, (n / total if total else 0.0), d


# ----------------------------------------------------------------------------
# Relative pose from matches
# ----------------------------------------------------------------------------

@dataclass
class PoseEstimate:
    R: np.ndarray          # 3x3
    t: np.ndarray          # (3,) unit vector - only the direction is known
    E: np.ndarray          # 3x3
    mask: np.ndarray       # (N,) bool, RANSAC inliers of findEssentialMat
    n_inliers: int         # RANSAC inliers
    n_inliers_pose: int    # inliers that also pass the positive depth check of recoverPose


def estimate_relative_pose(pts1, pts2, K1, K2=None,
                           thresh=RANSAC_THRESH, prob=RANSAC_PROB):
    """
    findEssentialMat (RANSAC) + recoverPose.

    In Fountain-P11 all cameras have the same K, so K goes directly into
    findEssentialMat and the threshold is in pixels.
    If K1 != K2 (my own photos, bonus), the points are first converted to
    normalized coordinates with their own K, K = I is used, and the threshold
    is divided by the average focal length so it stays ~1 px.
    """
    K2 = K1 if K2 is None else K2
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

    if np.allclose(K1, K2):
        p1, p2, K_eff, thr = pts1, pts2, K1, thresh
    else:
        p1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
        p2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)
        K_eff = np.eye(3)
        focal = (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) / 4.0
        thr = thresh / focal

    E, mask = cv2.findEssentialMat(p1, p2, K_eff, method=cv2.RANSAC,
                                   prob=prob, threshold=thr)
    if E is None:
        raise RuntimeError("findEssentialMat failed on %d matches" % len(pts1))
    mask = mask.ravel().astype(bool)

    n_pose, R, t, _ = cv2.recoverPose(E, p1, p2, K_eff,
                                      mask=mask.astype(np.uint8).copy())
    t = t.ravel() / np.linalg.norm(t)
    return PoseEstimate(R=R, t=t, E=E, mask=mask,
                        n_inliers=int(mask.sum()), n_inliers_pose=int(n_pose))


def pose_errors(R_hat, t_hat, R_gt, t_gt):
    """
    e_R = arccos((trace(R_hat R_gt^T) - 1) / 2)
    e_t = arccos(t_hat . t_gt / (|t_hat| |t_gt|))
    Both in degrees. The arccos argument is clipped to [-1, 1] because
    rounding can push it slightly outside.
    """
    e_R = rotation_angle_deg(R_hat @ R_gt.T)
    t_hat = np.asarray(t_hat, dtype=np.float64).ravel()
    t_gt = np.asarray(t_gt, dtype=np.float64).ravel()
    cos_t = float(t_hat @ t_gt / (np.linalg.norm(t_hat) * np.linalg.norm(t_gt)))
    e_t = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    return e_R, e_t
