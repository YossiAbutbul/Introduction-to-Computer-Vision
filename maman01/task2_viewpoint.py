"""
task2_viewpoint.py  -  Task 2: effect of the viewpoint change.

Uses SIFT, the best method in task 1. Image 0000 is the reference and is
matched with each of 0001 ... 0010, in order. For every pair:
  - matches, N_inliers and r_inliers w.r.t. F_GT
  - RANSAC inliers, e_R, e_t (median, min, max over the RANSAC repeats)
  - GT baseline and relative rotation angle

Outputs (in outputs/task2/):
  results.csv
  plot_inliers.png    N_inliers and r_inliers vs image index
  plot_errors.png     e_R and e_t vs image index
  plot_geometry.png   baseline and rotation angle vs image index
  matches_<pair>.jpg  match images for a few of the pairs

Run:  python task2_viewpoint.py
"""

import os

import numpy as np

from utils.dataset import ensure_dir, get_paths
from utils.features import match_pair
from utils.geometry import estimate_pose_repeated, geometric_consistency, gt_pair_geometry, pose_errors
from utils.plots import draw_matches, plot_vs_index, save_table

METHOD = "sift"
REF = 0
OTHERS = list(range(1, 11))
MATCH_IMAGES = (1, 5, 10)          # which pairs get a match image

OUT = ensure_dir(os.path.join(get_paths()["outputs"], "task2"))


def run_pair(j):
    gt = gt_pair_geometry(REF, j)
    res = match_pair(REF, j, METHOD)
    mask_gt, n_gt, r_gt, d = geometric_consistency(res.pts1, res.pts2, gt["F_gt"])

    poses = estimate_pose_repeated(res.pts1, res.pts2, gt["cam1"].K, gt["cam2"].K)
    errs = np.array([pose_errors(p.R, p.t, gt["R_gt"], gt["t_gt"]) for p in poses])
    n_ransac = np.array([p.n_inliers for p in poses])

    if j in MATCH_IMAGES:
        draw_matches(REF, j, res.pts1, res.pts2,
                     os.path.join(OUT, "matches_%04d-%04d.jpg" % (REF, j)),
                     mask=mask_gt, max_draw=150, scale=0.5)

    return {
        "image": j,
        "pair": "%04d-%04d" % (REF, j),
        "baseline": round(gt["baseline"], 3),
        "rot_angle_deg": round(gt["rot_angle_deg"], 2),
        "n_kp1": res.n_kp1,
        "n_kp2": res.n_kp2,
        "n_matches": res.n_matches,
        "n_inliers_FGT": n_gt,
        "r_inliers_FGT": round(r_gt, 4),
        "median_sampson_px2": round(float(np.median(d)), 4),
        "n_inliers_ransac": int(np.median(n_ransac)),
        "e_R_deg": round(float(np.median(errs[:, 0])), 4),
        "e_t_deg": round(float(np.median(errs[:, 1])), 4),
        "e_R_min": round(float(errs[:, 0].min()), 4),
        "e_R_max": round(float(errs[:, 0].max()), 4),
        "e_t_min": round(float(errs[:, 1].min()), 4),
        "e_t_max": round(float(errs[:, 1].max()), 4),
    }


def make_plots(rows):
    x = [r["image"] for r in rows]
    col = lambda k: [r[k] for r in rows]

    plot_vs_index(x, [
        {"ylabel": r"$N_{inliers}$ (consistent with $F_{GT}$)", "series": [{"y": col("n_inliers_FGT"), "label": "N_inliers"}]},
        {"ylabel": r"$r_{inliers} = N_{inliers} / N$", "ylim": (0, 1.02), "series": [{"y": col("r_inliers_FGT"), "label": "r_inliers"}]},
    ], os.path.join(OUT, "plot_inliers.png"))

    # log scale: the errors go from ~0.1 deg to tens of degrees
    plot_vs_index(x, [
        {"ylabel": "error [deg]  (log scale)", "log": True, "series": [
            {"y": col("e_R_deg"), "lo": col("e_R_min"), "hi": col("e_R_max"), "label": r"$e_R$ (rotation)"},
            {"y": col("e_t_deg"), "lo": col("e_t_min"), "hi": col("e_t_max"), "label": r"$e_t$ (translation direction)"},
        ]},
    ], os.path.join(OUT, "plot_errors.png"))

    plot_vs_index(x, [
        {"ylabel": "baseline [dataset units]", "series": [{"y": col("baseline"), "label": "baseline"}]},
        {"ylabel": "relative rotation angle [deg]", "series": [{"y": col("rot_angle_deg"), "label": "rotation"}]},
    ], os.path.join(OUT, "plot_geometry.png"))


def main():
    rows = [run_pair(j) for j in OTHERS]
    save_table(rows, os.path.join(OUT, "results.csv"))
    make_plots(rows)

    print("%-10s %6s %6s %7s %6s %6s %7s %8s %8s"
          % ("pair", "base", "angle", "matches", "N_in", "r_in", "RANSAC", "e_R", "e_t"))
    for r in rows:
        print("%-10s %6.2f %6.1f %7d %6d %6.3f %7d %8.3f %8.3f"
              % (r["pair"], r["baseline"], r["rot_angle_deg"], r["n_matches"], r["n_inliers_FGT"],
                 r["r_inliers_FGT"], r["n_inliers_ransac"], r["e_R_deg"], r["e_t_deg"]))


if __name__ == "__main__":
    main()
