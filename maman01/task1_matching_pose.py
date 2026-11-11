"""
task1_matching_pose.py  -  Task 1: point matching and camera pose estimation.

Compares SIFT, ORB and SuperPoint on 3 image pairs with growing viewpoint
change. For every pair and method:
  - match image (green = consistent with F_GT, red = not)
  - number of matches, N_inliers and r_inliers w.r.t. F_GT
  - time of detection + description + matching
  - RANSAC inliers, e_R, e_t

Outputs (in outputs/task1/):
  pair_selection.csv   GT baseline and rotation angle of all 55 pairs
  results_ratio.csv    main table (ratio test 0.75)
  results_mnn.csv      same with mutual nearest neighbors
  hardware.txt         CPU / device / versions used for the timing
  matches_<pair>_<method>.jpg

Run:  python task1_matching_pose.py
"""

import os
import platform

import cv2
import numpy as np

from utils.config import (METHODS, N_IMAGES, RANSAC_PROB, RANSAC_THRESH, RATIO_TEST, TAU,
                          TIMING_RUNS, TIMING_WARMUP)
from utils.dataset import ensure_dir, get_paths, load_all_cameras
from utils.features import superpoint_device_name, time_match_pair
from utils.geometry import (estimate_relative_pose, geometric_consistency, gt_pair_geometry,
                            pose_errors, relative_pose, rotation_angle_deg)
from utils.plots import draw_matches, save_table

# Chosen pairs: gap 1, 3, 5 around the middle of the sequence.
# baseline ~1.7 / 5.3 / 8.9, rotation ~10 / 33 / 59 deg.
PAIRS = [(5, 6), (4, 7), (3, 8)]

OUT = ensure_dir(os.path.join(get_paths()["outputs"], "task1"))


def pair_name(i, j):
    return "%04d-%04d" % (i, j)


def save_pair_selection():
    """GT baseline and rotation for all pairs - used to choose PAIRS."""
    cams = load_all_cameras()
    rows = []
    for i in range(N_IMAGES):
        for j in range(i + 1, N_IMAGES):
            R, _, b = relative_pose(cams[i], cams[j])
            rows.append({"pair": pair_name(i, j), "gap": j - i,
                         "baseline": round(b, 4),
                         "rot_angle_deg": round(rotation_angle_deg(R), 3),
                         "chosen": int((i, j) in PAIRS)})
    save_table(rows, os.path.join(OUT, "pair_selection.csv"))


def save_hardware():
    lines = [
        "platform: %s" % platform.platform(),
        "processor: %s" % platform.processor(),
        "cpu count: %s" % os.cpu_count(),
        "python: %s" % platform.python_version(),
        "opencv: %s" % cv2.__version__,
        "superpoint device: %s" % superpoint_device_name(),
        "timing: %d warm-up run(s), median of %d runs" % (TIMING_WARMUP, TIMING_RUNS),
    ]
    with open(os.path.join(OUT, "hardware.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def run_one(i, j, method, strategy, gt):
    """All task 1 numbers for one pair, one method and one filter strategy."""
    res, times = time_match_pair(i, j, method, strategy=strategy)

    # (b) quality of the matches w.r.t. F_GT
    mask_gt, n_gt, r_gt, d = geometric_consistency(res.pts1, res.pts2, gt["F_gt"])

    # (c) relative pose from the matches
    pose = estimate_relative_pose(res.pts1, res.pts2, gt["cam1"].K, gt["cam2"].K)
    e_R, e_t = pose_errors(pose.R, pose.t, gt["R_gt"], gt["t_gt"])

    row = {
        "pair": pair_name(i, j),
        "baseline": round(gt["baseline"], 3),
        "rot_angle_deg": round(gt["rot_angle_deg"], 2),
        "method": method,
        "filter": strategy,
        "n_kp1": res.n_kp1,
        "n_kp2": res.n_kp2,
        "n_matches": res.n_matches,
        "n_inliers_FGT": n_gt,
        "r_inliers_FGT": round(r_gt, 4),
        "median_sampson_px2": round(float(np.median(d)), 4),
        "time_median_s": round(float(np.median(times)), 3),
        "time_min_s": round(float(times.min()), 3),
        "time_max_s": round(float(times.max()), 3),
        "n_inliers_ransac": pose.n_inliers,
        "e_R_deg": round(e_R, 4),
        "e_t_deg": round(e_t, 4),
    }
    return row, res, mask_gt


def print_table(rows):
    print("%-10s %-10s %8s %6s %6s %7s %7s %7s %7s"
          % ("pair", "method", "matches", "N_in", "r_in", "time_s", "RANSAC", "e_R", "e_t"))
    for r in rows:
        print("%-10s %-10s %8d %6d %6.3f %7.3f %7d %7.3f %7.3f"
              % (r["pair"], r["method"], r["n_matches"], r["n_inliers_FGT"],
                 r["r_inliers_FGT"], r["time_median_s"], r["n_inliers_ransac"],
                 r["e_R_deg"], r["e_t_deg"]))


def main():
    save_pair_selection()
    for line in save_hardware():
        print(line)
    print("params: ratio=%.2f  tau=%.1f px  RANSAC prob=%.3f thresh=%.1f px\n"
          % (RATIO_TEST, TAU, RANSAC_PROB, RANSAC_THRESH))

    for strategy in ("ratio", "mnn"):
        rows = []
        for i, j in PAIRS:
            gt = gt_pair_geometry(i, j)
            for method in METHODS:
                row, res, mask_gt = run_one(i, j, method, strategy, gt)
                rows.append(row)
                # (a) match images, only for the main filter
                if strategy == "ratio":
                    draw_matches(i, j, res.pts1, res.pts2,
                                 os.path.join(OUT, "matches_%s_%s.jpg" % (pair_name(i, j), method)),
                                 mask=mask_gt, max_draw=150, scale=0.5)
        save_table(rows, os.path.join(OUT, "results_%s.csv" % strategy))
        print("=== filter: %s ===" % strategy)
        print_table(rows)
        print()


if __name__ == "__main__":
    main()
