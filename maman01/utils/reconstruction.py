"""
reconstruction.py
Triangulation, depth / angle checks, reprojection error and filtering of
3D points. Used in task 3.
"""

import cv2
import numpy as np


def projection_matrices(K1, K2, R, t):
    """P1 = K1 [I | 0],  P2 = K2 [R | t]  (camera 1 is the origin)."""
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, np.asarray(t, dtype=np.float64).reshape(3, 1)])
    return P1, P2


def triangulate(P1, P2, pts1, pts2):
    """
    cv2.triangulatePoints, then divide by the 4th coordinate.
    Returns X (N,3) in camera 1 coordinates, and w (the 4th coordinate before
    dividing) so that points with w close to 0 can be removed later.
    """
    p1 = np.ascontiguousarray(np.asarray(pts1, dtype=np.float64).reshape(-1, 2).T)
    p2 = np.ascontiguousarray(np.asarray(pts2, dtype=np.float64).reshape(-1, 2).T)
    Xh = cv2.triangulatePoints(P1, P2, p1, p2)       # 4 x N
    w = Xh[3].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (Xh[:3] / Xh[3]).T
    return X, w


def depths(X, R, t):
    """Z of each point in camera 1 (= X[:,2]) and in camera 2 (= (R X + t)_z)."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    z1 = X[:, 2]
    z2 = (X @ R.T + np.asarray(t, dtype=np.float64).ravel())[:, 2]
    return z1, z2


def triangulation_angle_deg(X, R, t):
    """
    Angle between the two rays (camera 1 -> X and camera 2 -> X), in degrees.
    When the angle is small the rays are almost parallel and the depth is
    not reliable.
    Camera 2 center in camera 1 coordinates: C2 = -R^T t.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    C2 = (-R.T @ np.asarray(t, dtype=np.float64).ravel()).reshape(1, 3)
    v1 = X                        # camera 1 is at the origin
    v2 = X - C2
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_a = np.sum(v1 * v2, axis=1) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def reprojection_error(X, P1, P2, pts1, pts2):
    """
    e = sqrt((|x1 - x1_hat|^2 + |x2 - x2_hat|^2) / 2), in pixels, per point.
    x_hat = projection of X back into each image.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    Xh = np.hstack([X, np.ones((len(X), 1))])

    def project(P):
        xh = Xh @ P.T
        with np.errstate(divide="ignore", invalid="ignore"):
            return xh[:, :2] / xh[:, 2:3]

    r1 = np.sum((np.asarray(pts1).reshape(-1, 2) - project(P1)) ** 2, axis=1)
    r2 = np.sum((np.asarray(pts2).reshape(-1, 2) - project(P2)) ** 2, axis=1)
    return np.sqrt((r1 + r2) / 2.0)


def filter_3d(clouds, min_angle_deg=2.0, max_depth=None):
    """
    One mask for all clouds, so point i in the GT cloud and point i in the
    estimated cloud come from the same match.

    clouds: list of dicts with keys X, w, R, t (one dict per reconstruction).
    A point is kept only if in every cloud:
      - all coordinates are finite
      - |w| > 1e-10
      - depth > 0 in both cameras
      - triangulation angle >= min_angle_deg
      - (optional) distance from camera 1 <= max_depth

    Returns the mask and a dict with how many points each rule removed.
    """
    n = len(clouds[0]["X"])
    mask = np.ones(n, dtype=bool)
    criteria = {}

    for k, c in enumerate(clouds):
        X, w, R, t = c["X"], c["w"], c["R"], c["t"]
        finite = np.all(np.isfinite(X), axis=1)
        w_ok = np.abs(w) > 1e-10
        z1, z2 = depths(X, R, t)
        depth_ok = (z1 > 0) & (z2 > 0)
        angle = triangulation_angle_deg(X, R, t)
        angle_ok = angle >= min_angle_deg
        local = finite & w_ok & depth_ok & angle_ok
        if max_depth is not None:
            near = np.linalg.norm(X, axis=1) <= max_depth
            local &= near
            criteria["cloud%d_too_far" % k] = int(np.sum(~near))
        criteria["cloud%d_non_finite" % k] = int(np.sum(~finite))
        criteria["cloud%d_degenerate_w" % k] = int(np.sum(~w_ok))
        criteria["cloud%d_negative_depth" % k] = int(np.sum(~depth_ok))
        criteria["cloud%d_small_angle" % k] = int(np.sum(~angle_ok))
        mask &= local

    criteria["n_input"] = n
    criteria["n_kept"] = int(mask.sum())
    criteria["min_angle_deg"] = min_angle_deg
    return mask, criteria
