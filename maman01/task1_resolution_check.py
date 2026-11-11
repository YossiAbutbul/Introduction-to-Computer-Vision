"""
task1_resolution_check.py  -  extra check for task 1.

In task 1 SuperPoint had a lower r_inliers and a much higher Sampson distance
than SIFT. My guess was that this comes from SuperPoint running on an image
whose long side is resized to 1024, so its keypoints are less precise.
This script tests that guess on the 3 pairs of task 1:

  1. split the matches by sqrt(d_S): < 2 px (consistent), 2-6 px (near miss),
     >= 6 px (gross error)
  2. check that SuperPoint keypoints lie on the pixel grid of the resized image
  3. run SuperPoint with the long side resized to 1024 (default), 2048 and 3072
     (full resolution)
  4. control: run SIFT on images resized to 1024 and scale its keypoints back

Output: outputs/task1/resolution_check.csv

Run:  python task1_resolution_check.py
"""

import os
import time

import cv2
import numpy as np

from utils.dataset import ensure_dir, get_paths, load_image
from utils.features import _get_superpoint, load_detector_input, match_descriptors
from utils.geometry import estimate_pose_repeated, gt_pair_geometry, pose_errors, sampson_distance
from utils.plots import save_table

PAIRS = [(5, 6), (4, 7), (3, 8)]
OUT = ensure_dir(os.path.join(get_paths()["outputs"], "task1"))


def superpoint_features(idx, long_side):
    """SuperPoint keypoints / descriptors with the long side resized to long_side."""
    model, _ = _get_superpoint()
    img = load_detector_input(idx, "superpoint")
    feats = model.extract(img, resize=long_side)
    kp = feats["keypoints"][0].cpu().numpy().astype(np.float32)
    desc = feats["descriptors"][0].cpu().numpy().astype(np.float32)
    return kp, desc, cv2.NORM_L2


def sift_features(idx, long_side):
    """SIFT on the image resized so the long side is long_side, keypoints in original pixels."""
    img = load_image(idx)
    h, w = img.shape
    s = long_side / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    sx, sy = small.shape[1] / w, small.shape[0] / h
    kp, desc = cv2.SIFT_create().detectAndCompute(small, None)
    pts = np.float32([k.pt for k in kp])
    # pixel centres: (x + 0.5) / s - 0.5
    pts = np.stack([(pts[:, 0] + 0.5) / sx - 0.5, (pts[:, 1] + 0.5) / sy - 0.5], axis=1)
    return pts.astype(np.float32), desc, cv2.NORM_L2


def on_grid_share(idx):
    """Share of SuperPoint x and y coordinates that fall on the integer grid of the 1024 image."""
    model, _ = _get_superpoint()
    img = load_detector_input(idx, "superpoint")
    h, w = img.shape[-2:]
    kp = model.extract(img)["keypoints"][0].cpu().numpy()
    # lightglue resizes to (1024, round(1024 * h / w)) for a landscape image
    w_small, h_small = 1024, int(round(1024 * h / w))
    res = []
    for k, (n_small, n) in enumerate([(w_small, w), (h_small, h)]):
        c = (kp[:, k] + 0.5) * n_small / n - 0.5
        res.append(float(np.mean(np.abs(c - np.round(c)) < 0.01)))
    return res, w / w_small


def evaluate(method, long_side, i, j, gt):
    feat = superpoint_features if method == "superpoint" else sift_features
    start = time.perf_counter()
    kp1, d1, norm = feat(i, long_side)
    kp2, d2, _ = feat(j, long_side)
    m1, m2 = match_descriptors(d1, d2, norm)
    elapsed = time.perf_counter() - start
    p1, p2 = kp1[m1].astype(np.float64), kp2[m2].astype(np.float64)

    e = np.sqrt(sampson_distance(p1, p2, gt["F_gt"]))           # ~ px
    poses = estimate_pose_repeated(p1, p2, gt["cam1"].K)
    errs = np.array([pose_errors(p.R, p.t, gt["R_gt"], gt["t_gt"]) for p in poses])
    return {
        "pair": "%04d-%04d" % (i, j),
        "method": method,
        "long_side": long_side,
        "n_matches": len(p1),
        "r_inliers_FGT": round(float(np.mean(e < 2)), 4),
        "share_near_miss_2_6px": round(float(np.mean((e >= 2) & (e < 6))), 4),
        "share_gross_6px": round(float(np.mean(e >= 6)), 4),
        "median_sqrt_sampson_px": round(float(np.median(e)), 4),
        "e_R_deg": round(float(np.median(errs[:, 0])), 4),
        "e_t_deg": round(float(np.median(errs[:, 1])), 4),
        "time_s": round(elapsed, 2),
    }


def main():
    shares, step = on_grid_share(5)
    print("SuperPoint keypoints on the grid of the 1024 image: x %.3f, y %.3f (grid step %.1f px)"
          % (shares[0], shares[1], step))

    runs = [("superpoint", 1024), ("superpoint", 2048), ("superpoint", 3072),
            ("sift", 3072), ("sift", 1024)]
    rows = []
    for i, j in PAIRS:
        gt = gt_pair_geometry(i, j)
        for method, side in runs:
            row = evaluate(method, side, i, j, gt)
            rows.append(row)
            print("%s %-10s %4d | matches %5d  r_in %.3f  near-miss %.3f  gross %.3f  med sqrt(dS) %.3f px  "
                  "e_R %.3f  e_t %.3f  %.1fs" % (row["pair"], method, side, row["n_matches"], row["r_inliers_FGT"],
                                               row["share_near_miss_2_6px"], row["share_gross_6px"],
                                               row["median_sqrt_sampson_px"], row["e_R_deg"], row["e_t_deg"], row["time_s"]))
    rows.insert(0, {"pair": "grid check (image 0005)", "method": "superpoint", "long_side": 1024,
                    "n_matches": "", "r_inliers_FGT": "", "share_near_miss_2_6px": "x on grid %.3f" % shares[0],
                    "share_gross_6px": "y on grid %.3f" % shares[1], "median_sqrt_sampson_px": "step %.1f px" % step,
                    "e_R_deg": "", "e_t_deg": "", "time_s": ""})
    save_table(rows, os.path.join(OUT, "resolution_check.csv"), fieldnames=list(rows[1].keys()))


if __name__ == "__main__":
    main()
