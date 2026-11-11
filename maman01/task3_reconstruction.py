"""
task3_reconstruction.py  -  Task 3: reconstruction with the true and the estimated camera pose.
(The numbers here are also used to answer task 4.)

For one pair and SIFT:
  1. match, keep only the matches consistent with F_GT (d_S < tau^2)
  2. estimate R_hat, t_hat from all the matches (same as task 1)
  3. triangulate the consistent matches twice, with camera 1 as the origin:
       GT:        P1 = K1 [I | 0],  P2 = K2 [R_gt  | t_gt]
       estimated: P1 = K1 [I | 0],  P2 = K2 [R_hat | s t_hat],  s = |t_gt| / |t_hat|
  4. one filter for both clouds (finite, |w| > 0, depth > 0 in both cameras,
     triangulation angle >= MIN_TRI_ANGLE_DEG), so point i is the same match in both
  5. e_3D = |X_est - X_gt| per point, mean and median, reprojection errors

The main pair is MAIN_PAIR. The same numbers are also computed for the other
two pairs of task 1, for the comparison in task 4.

Outputs (in outputs/task3/):
  summary.csv          one row per pair
  cloud_<pair>.png     both clouds on the same axes (3D, top view, front view)
  e3d_vs_depth_<pair>.png

Run:  python task3_reconstruction.py
"""

import os

import numpy as np

from utils.config import MIN_TRI_ANGLE_DEG
from utils.dataset import ensure_dir, get_paths
from utils.features import match_pair
from utils.geometry import estimate_pose_repeated, geometric_consistency, gt_pair_geometry, pose_errors
from utils.plots import plot_point_clouds, save_table
from utils.reconstruction import (filter_3d, projection_matrices, reprojection_error, triangulate,
                                  triangulation_angle_deg)

METHOD = "sift"
MAIN_PAIR = (5, 6)
OTHER_PAIRS = [(4, 7), (3, 8)]

OUT = ensure_dir(os.path.join(get_paths()["outputs"], "task3"))


def pick_pose(poses):
    """
    From the RANSAC repeats take the run with the most inliers. This is what I
    would do without ground truth (choosing by e_R / e_t would use the GT).
    """
    return max(poses, key=lambda p: p.n_inliers)


def run_pair(i, j, make_plots):
    gt = gt_pair_geometry(i, j)
    K1, K2 = gt["cam1"].K, gt["cam2"].K
    res = match_pair(i, j, METHOD)

    # matches consistent with F_GT - the same matches go into both reconstructions
    mask_gt, n_gt, _, _ = geometric_consistency(res.pts1, res.pts2, gt["F_gt"])
    p1, p2 = res.pts1[mask_gt], res.pts2[mask_gt]

    # estimated pose, from all the matches like in task 1
    pose = pick_pose(estimate_pose_repeated(res.pts1, res.pts2, K1, K2))
    e_R, e_t = pose_errors(pose.R, pose.t, gt["R_gt"], gt["t_gt"])
    s = np.linalg.norm(gt["t_gt"]) / np.linalg.norm(pose.t)
    t_est = s * pose.t

    P1_gt, P2_gt = projection_matrices(K1, K2, gt["R_gt"], gt["t_gt"])
    P1_es, P2_es = projection_matrices(K1, K2, pose.R, t_est)
    X_gt, w_gt = triangulate(P1_gt, P2_gt, p1, p2)
    X_es, w_es = triangulate(P1_es, P2_es, p1, p2)

    keep, crit = filter_3d([
        {"X": X_gt, "w": w_gt, "R": gt["R_gt"], "t": gt["t_gt"]},
        {"X": X_es, "w": w_es, "R": pose.R, "t": t_est},
    ], min_angle_deg=MIN_TRI_ANGLE_DEG)

    Xg, Xe = X_gt[keep], X_es[keep]
    diff = Xe - Xg
    e3d = np.linalg.norm(diff, axis=1)
    depth = Xg[:, 2]
    # how much of the error is along the viewing ray of camera 1 (= depth error)
    rays = Xg / np.linalg.norm(Xg, axis=1, keepdims=True)
    along_ray = np.abs(np.sum(diff * rays, axis=1)) / e3d
    # close to 1 = all points move the same way (systematic), close to 0 = random
    systematic = np.linalg.norm(diff.mean(axis=0)) / e3d.mean()
    tri_angle = triangulation_angle_deg(Xg, gt["R_gt"], gt["t_gt"])
    rep_gt = reprojection_error(Xg, P1_gt, P2_gt, p1[keep], p2[keep])
    rep_es = reprojection_error(Xe, P1_es, P2_es, p1[keep], p2[keep])

    name = "%04d-%04d" % (i, j)
    if make_plots:
        plot_point_clouds(Xg, Xe, os.path.join(OUT, "cloud_%s.png" % name),
                          title="%s, SIFT, %d points" % (name, len(Xg)))
        plot_e3d_vs_depth(depth, e3d, os.path.join(OUT, "e3d_vs_depth_%s.png" % name))

    return {
        "pair": name,
        "baseline": round(gt["baseline"], 3),
        "rot_angle_deg": round(gt["rot_angle_deg"], 2),
        "n_matches": res.n_matches,
        "n_consistent": n_gt,
        "n_points_kept": int(keep.sum()),
        "removed_non_finite": crit["cloud0_non_finite"] + crit["cloud1_non_finite"],
        "removed_w": crit["cloud0_degenerate_w"] + crit["cloud1_degenerate_w"],
        "removed_neg_depth": crit["cloud0_negative_depth"] + crit["cloud1_negative_depth"],
        "removed_small_angle": crit["cloud0_small_angle"] + crit["cloud1_small_angle"],
        "pose_ransac_inliers": pose.n_inliers,
        "e_R_deg": round(e_R, 4),
        "e_t_deg": round(e_t, 4),
        "scale_s": round(float(s), 4),
        "e3d_mean": round(float(e3d.mean()), 5),
        "e3d_median": round(float(np.median(e3d)), 5),
        "e3d_max": round(float(e3d.max()), 5),
        "depth_median": round(float(np.median(depth)), 3),
        "e3d_rel_median_pct": round(float(np.median(e3d / depth)) * 100, 3),
        "e3d_along_ray_median": round(float(np.median(along_ray)), 3),
        "e3d_systematic": round(float(systematic), 3),
        "depth_ratio_est_gt_median": round(float(np.median(Xe[:, 2] / Xg[:, 2])), 4),
        "tri_angle_median_deg": round(float(np.median(tri_angle)), 1),
        "reproj_gt_mean_px": round(float(rep_gt.mean()), 4),
        "reproj_gt_median_px": round(float(np.median(rep_gt)), 4),
        "reproj_est_mean_px": round(float(rep_es.mean()), 4),
        "reproj_est_median_px": round(float(np.median(rep_es)), 4),
    }


def plot_e3d_vs_depth(depth, e3d, out_path):
    """e_3D of every point against its depth (GT cloud)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.plots import GRID, INK, MUTED, SERIES_COLORS

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(depth, e3d, s=6, color=SERIES_COLORS[0], alpha=0.6, linewidths=0)
    ax.set_xlabel("depth Z in camera 1 [dataset units]", color=INK)
    ax.set_ylabel(r"$e_{3D}$ [dataset units]", color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    rows = [run_pair(*MAIN_PAIR, make_plots=True)]
    rows += [run_pair(i, j, make_plots=True) for i, j in OTHER_PAIRS]
    save_table(rows, os.path.join(OUT, "summary.csv"))
    for r in rows:
        print("--- %s ---" % r["pair"])
        for k, v in r.items():
            if k != "pair":
                print("  %-22s %s" % (k, v))


if __name__ == "__main__":
    main()
