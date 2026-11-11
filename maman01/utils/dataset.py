"""
dataset.py
Paths, images and ground-truth cameras of Fountain-P11.

Dataset: Fountain-P11 (11 images + ground-truth cameras).
Dataset Source: https://tinyurl.com/3nrd4xxa
Paths are built relative to the maman01 folder.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .config import N_IMAGES


# ----------------------------------------------------------------------------
# Paths and loading
# ----------------------------------------------------------------------------

def get_paths():
    """Folders of the project. This file is in maman01/utils, so go one level up."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "root": here,
        "images": os.path.join(here, "data", "images"),
        "cameras": os.path.join(here, "data", "cameras"),
        "outputs": os.path.join(here, "outputs"),
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def image_path(idx):
    """5 -> .../data/images/0005.jpg"""
    return os.path.join(get_paths()["images"], "%04d.jpg" % idx)


def camera_path(idx):
    """5 -> .../data/cameras/0005.jpg.camera"""
    return os.path.join(get_paths()["cameras"], "%04d.jpg.camera" % idx)


def load_image(idx, color=False):
    """Read an image, grayscale by default."""
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(image_path(idx), flag)
    if img is None:
        raise FileNotFoundError("cannot read %s" % image_path(idx))
    return img


# ----------------------------------------------------------------------------
# Cameras
#
# Camera file (*.jpg.camera), 9 lines:
#   1-3  K
#   4    0 0 0   (not used)
#   5-7  R_stored
#   8    C = camera center in world coordinates
#   9    width height
#
# R_stored rotates camera -> world, so world -> camera is R = R_stored.T,
# and t = -R C.
# I checked this on pair 0005-0006: with the transpose the median Sampson
# distance of the SIFT matches w.r.t. F_GT is 0.027 px^2, without it ~6810.
#
# R in the files has only 6 digits, so R R^T differs from I by ~1e-6.
# That is enough to break e_R for small angles (trace(R) ~ 3 - theta^2, and
# for theta = 0.1 deg theta^2 is ~3e-6). So R is replaced by the closest
# rotation matrix (SVD: R = U V^T) right after reading it.
# ----------------------------------------------------------------------------

@dataclass
class Camera:
    idx: int
    K: np.ndarray     # 3x3
    R: np.ndarray     # 3x3, world -> camera
    C: np.ndarray     # (3,) camera center, world coordinates
    t: np.ndarray     # (3,) t = -R C
    width: int
    height: int

    @property
    def P(self):
        """P = K [R | t]"""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])


def closest_rotation(R):
    """Closest rotation matrix to R (in Frobenius norm): U V^T from the SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_rot = U @ Vt
    if np.linalg.det(R_rot) < 0:        # avoid a reflection
        U[:, -1] *= -1
        R_rot = U @ Vt
    return R_rot


def load_camera(idx):
    with open(camera_path(idx), "r") as f:
        rows = [line.split() for line in f if line.strip()]
    if len(rows) != 9:
        raise ValueError("%s: expected 9 non-empty lines, got %d"
                         % (camera_path(idx), len(rows)))

    K = np.array([[float(v) for v in rows[i]] for i in range(3)], dtype=np.float64)
    R_stored = np.array([[float(v) for v in rows[4 + i]] for i in range(3)],
                        dtype=np.float64)
    C = np.array([float(v) for v in rows[7]], dtype=np.float64)
    width, height = (int(v) for v in rows[8])

    R = closest_rotation(R_stored.T)    # camera->world  ->  world->camera
    t = -R @ C
    return Camera(idx=idx, K=K, R=R, C=C, t=t, width=width, height=height)


def load_all_cameras():
    return [load_camera(i) for i in range(N_IMAGES)]
