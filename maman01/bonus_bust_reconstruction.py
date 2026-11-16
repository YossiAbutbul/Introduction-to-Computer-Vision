"""
bonus_bust_reconstruction.py  -  Bonus question: 3D reconstruction of an object I photographed.

10 photos of one object (a 3D-printed bust, about 20 cm high) in bonus/images.
Phone: Galaxy S24 Ultra, main camera, normal photo mode. I walked on an arc
around the bust at about 35 cm from it, with the camera always facing it.
There is no ground truth here.

Steps:
  1. K from the EXIF data:  f_px = f_35mm / 36 * image width
     (assumes: image not cropped, principal point in the centre, fx = fy, no distortion)
  2. SIFT + ratio test between consecutive photos, at full resolution.
     The object has one colour, so SIFT runs with a lower contrast threshold
     (0.01 instead of the default 0.04) to find more keypoints on it.
  3. essential matrix with RANSAC + recoverPose (same function as in task 1,
     the RANSAC repeat with the most inliers is used)
  4. triangulation of the RANSAC inliers with P1 = K[I|0], P2 = K[R|t].
     |t| = 1, because the real scale is unknown.
  5. same filter as in task 3 (finite, |w| > 0, positive depth, angle >= 2 deg)
     and the reprojection error

Outputs (in outputs/bonus_bust/):
  photos.jpg             the photos
  photos_small/          small copies of the photos without EXIF (no GPS), for the submission
  K.txt                  the calibration matrix and where it came from
  results.csv            one row per pair (also how many 3D points are on the object)
  focal_check.csv        how the result changes when f is changed
  threshold_check.csv    RANSAC threshold 1 / 2 / 3 / 5 px on the main pair
  scale_check.csv        rough size check using the ~35 cm shooting distance
  matches_<pair>.jpg     matches of the main pair (green = RANSAC inlier)
  cloud_<pair>.png       point cloud of the main pair

Run:  python bonus_bust_reconstruction.py
"""

import os

import cv2
import numpy as np
from PIL import Image

from utils.config import MIN_TRI_ANGLE_DEG
from utils.dataset import ensure_dir, get_paths
from utils.features import match_descriptors
from utils.geometry import estimate_pose_repeated, estimate_relative_pose, rotation_angle_deg, sampson_distance
from utils.plots import save_table
from utils.reconstruction import filter_3d, projection_matrices, reprojection_error, triangulate

ROOT = get_paths()["root"]
IMG_DIR = os.path.join(ROOT, "bonus", "images")
OUT = ensure_dir(os.path.join(get_paths()["outputs"], "bonus_bust"))
SCALE = 1.0                      # 1.0 = full resolution
SIFT_CONTRAST = 0.01             # OpenCV default is 0.04
DIST_CM = 35.0                   # my rough distance from the bust while shooting
SMALL_WIDTH = 1600               # width of the small copies for the submission


def list_images():
    return sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg")))


def short_name(fname):
    """20261116_154333.jpg -> 4333 (last 4 digits of the file name)"""
    return fname.split(".")[0][-4:]


def intrinsics_from_exif(fname, scale=SCALE, f_factor=1.0):
    """K from the 35 mm equivalent focal length in the EXIF data."""
    img = Image.open(os.path.join(IMG_DIR, fname))
    width, height = img.size
    exif = img.getexif().get_ifd(0x8769)                 # Exif sub-IFD
    f35 = float(exif[0xA405])                            # FocalLengthIn35mmFilm
    f_px = f35 / 36.0 * width * scale * f_factor
    K = np.array([[f_px, 0.0, width * scale / 2.0],
                  [0.0, f_px, height * scale / 2.0],
                  [0.0, 0.0, 1.0]])
    return K, f35, (width, height)


def load_small(fname, color=False):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(os.path.join(IMG_DIR, fname), flag)
    if SCALE != 1.0:
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return img


def object_mask(bgr):
    """
    Rough mask of my object: it is the only purple thing in the photos, so the
    mask is the convex hull of the largest purple region. Used only to count
    how many points are on the object and how many on the background.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(hsv, (125, 50, 20), (165, 255, 255))
    purple = cv2.morphologyEx(purple, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(purple)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(purple)
    cv2.fillConvexPoly(mask, cv2.convexHull(cv2.findNonZero((labels == biggest).astype(np.uint8))), 255)
    return mask


def on_mask(mask, pts):
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return mask[y, x] > 0


def reconstruct_pair(p1, p2, K):
    """Pose from the matches, then triangulation of the RANSAC inliers."""
    poses = estimate_pose_repeated(p1, p2, K)
    pose = max(poses, key=lambda p: p.n_inliers)
    q1, q2 = p1[pose.mask], p2[pose.mask]
    P1, P2 = projection_matrices(K, K, pose.R, pose.t)
    X, w = triangulate(P1, P2, q1, q2)
    keep, crit = filter_3d([{"X": X, "w": w, "R": pose.R, "t": pose.t}], min_angle_deg=MIN_TRI_ANGLE_DEG)
    rep = reprojection_error(X[keep], P1, P2, q1[keep], q2[keep])
    return {"pose": pose, "q1": q1[keep], "X": X[keep], "rep": rep, "crit": crit,
            "n_ransac_median": int(np.median([p.n_inliers for p in poses]))}


def save_photos_sheet(files):
    tiles = []
    for f in files:
        t = cv2.resize(load_small(f, color=True), (476, 357), interpolation=cv2.INTER_AREA)
        cv2.putText(t, short_name(f), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % 4:                                  # fill the last row
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(os.path.join(OUT, "photos.jpg"), np.vstack(rows))


def save_small_copies(files):
    """Small copies for the submission. cv2.imwrite does not copy the EXIF, so no GPS."""
    out = ensure_dir(os.path.join(OUT, "photos_small"))
    for f in files:
        img = cv2.imread(os.path.join(IMG_DIR, f))
        s = SMALL_WIDTH / img.shape[1]
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, short_name(f) + ".jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])


def step_cm(rot_deg, dist_cm=DIST_CM):
    """
    I moved on a circle of radius dist_cm around the bust, facing it, so the
    rotation between two photos is the angle I walked. The camera moved by the
    chord of that angle: 2 r sin(angle / 2).
    """
    return 2.0 * dist_cm * np.sin(np.radians(rot_deg) / 2.0)


def save_matches_image(f1, f2, p1, p2, mask, out_path, max_draw=200):
    img1, img2 = load_small(f1, color=True), load_small(f2, color=True)
    canvas = np.hstack([img1, img2])
    w1 = img1.shape[1]
    idx = np.arange(len(p1))
    if len(idx) > max_draw:
        idx = np.random.default_rng(0).choice(idx, size=max_draw, replace=False)
    for i in idx:
        a = (int(round(p1[i, 0])), int(round(p1[i, 1])))
        b = (int(round(p2[i, 0])) + w1, int(round(p2[i, 1])))
        color = (0, 255, 0) if mask[i] else (0, 0, 255)          # BGR
        cv2.line(canvas, a, b, color, 4, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.resize(canvas, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))


def plot_cloud(X, colors_bgr, out_path, title):
    """Top view and front view of the cloud, each point in its colour from photo 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    rgb = colors_bgr[:, ::-1] / 255.0
    # a few far points would squeeze the plot, so the axes show the central 96% of the points
    lo, hi = np.percentile(X, 2, axis=0), np.percentile(X, 98, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4))
    views = [("top view (X-Z)", 0, 2, 1.0, "X", "Z = depth"), ("front view (X vs -Y)", 0, 1, -1.0, "X", "-Y = up")]
    for ax, (name, a, b, sign, xl, yl) in zip(axes, views):
        ax.scatter(X[:, a], sign * X[:, b], s=6, c=rgb, linewidths=0)
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(*sorted((sign * lo[b], sign * hi[b])))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xl + "  [baseline = 1]")
        ax.set_ylabel(yl + "  [baseline = 1]")
        ax.grid(True, color="#e1e0d9", linewidth=0.6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    files = list_images()
    save_photos_sheet(files)
    save_small_copies(files)

    K, f35, size = intrinsics_from_exif(files[0])
    with open(os.path.join(OUT, "K.txt"), "w") as f:
        f.write("image size: %d x %d, used at scale %.1f\n" % (size[0], size[1], SCALE))
        f.write("EXIF focal length in 35 mm film: %.0f mm\n" % f35)
        f.write("f_px = f35 / 36 * width * scale = %.1f\n" % K[0, 0])
        f.write("K =\n%s\n" % np.array2string(K, precision=1, suppress_small=True))
    print("K =\n", np.round(K, 1))

    sift = cv2.SIFT_create(contrastThreshold=SIFT_CONTRAST)
    feats, masks = [], []
    for fname in files:
        kp, desc = sift.detectAndCompute(load_small(fname), None)
        feats.append((np.float32([k.pt for k in kp]), desc))
        masks.append(object_mask(load_small(fname, color=True)))

    rows, results = [], []
    for i in range(len(files) - 1):
        (k1, d1), (k2, d2) = feats[i], feats[i + 1]
        m1, m2 = match_descriptors(d1, d2, cv2.NORM_L2)
        p1, p2 = k1[m1].astype(np.float64), k2[m2].astype(np.float64)
        res = reconstruct_pair(p1, p2, K)
        res.update({"i": i, "p1": p1, "p2": p2})
        results.append(res)
        rows.append({
            "pair": "%s-%s" % (short_name(files[i]), short_name(files[i + 1])),
            "rot_deg": round(rotation_angle_deg(res["pose"].R), 1),
            "step_cm": round(float(step_cm(rotation_angle_deg(res["pose"].R))), 1),
            "n_kp1": len(k1), "n_kp2": len(k2),
            "n_matches": len(p1),
            "n_inliers_ransac": res["pose"].n_inliers,
            "n_points_3d_valid": len(res["X"]),
            "n_points_on_object": int(on_mask(masks[i], res["q1"]).sum()),
            "removed_neg_depth": res["crit"]["cloud0_negative_depth"],
            "removed_small_angle": res["crit"]["cloud0_small_angle"],
            "reproj_mean_px": round(float(res["rep"].mean()), 3),
            "reproj_median_px": round(float(np.median(res["rep"])), 3),
        })
        print(rows[-1])
    save_table(rows, os.path.join(OUT, "results.csv"))

    # figures for the pair with the most valid 3D points
    best = max(results, key=lambda r: len(r["X"]))
    f1, f2 = files[best["i"]], files[best["i"] + 1]
    name = "%s-%s" % (short_name(f1), short_name(f2))
    save_matches_image(f1, f2, best["p1"], best["p2"], best["pose"].mask, os.path.join(OUT, "matches_%s.jpg" % name))
    img1 = load_small(f1, color=True)
    px = np.clip(np.round(best["q1"]).astype(int), 0, [img1.shape[1] - 1, img1.shape[0] - 1])
    plot_cloud(best["X"], img1[px[:, 1], px[:, 0]].astype(float), os.path.join(OUT, "cloud_%s.png" % name),
               "%s, %d points" % (name, len(best["X"])))

    # how sensitive is the result to f? (1.04 = f35 defined by the diagonal instead of the width)
    check = []
    for factor in (0.8, 0.9, 1.0, 1.04, 1.1, 1.2):
        Kf, _, _ = intrinsics_from_exif(f1, f_factor=factor)
        r = reconstruct_pair(best["p1"], best["p2"], Kf)
        check.append({"pair": name, "f_factor": factor, "f_px": round(float(Kf[0, 0]), 1),
                      "n_inliers_ransac_median": r["n_ransac_median"],
                      "n_points_3d_valid": len(r["X"]),
                      "reproj_median_px": round(float(np.median(r["rep"])), 3)})
        print(check[-1])
    save_table(check, os.path.join(OUT, "focal_check.csv"))

    # In the match image most matches on the head are rejected by RANSAC although
    # they look right. Check: how far are they from the epipolar lines, and what
    # happens with a larger RANSAC threshold. "head" = upper third of the object mask.
    mask = masks[best["i"]]
    p1, p2 = best["p1"], best["p2"]
    on_obj = on_mask(mask, p1)
    rows_y = np.where(mask.any(axis=1))[0]
    head = on_obj & (p1[:, 1] < rows_y.min() + (rows_y.max() - rows_y.min()) / 3.0)
    K_inv = np.linalg.inv(K)
    thr_rows = []
    for thr in (1.0, 2.0, 3.0, 5.0):
        pose = estimate_relative_pose(p1, p2, K, thresh=thr)
        dist = np.sqrt(sampson_distance(p1, p2, K_inv.T @ pose.E @ K_inv))      # ~ px from the epipolar line
        P1, P2 = projection_matrices(K, K, pose.R, pose.t)
        X, _ = triangulate(P1, P2, p1[pose.mask], p2[pose.mask])
        rep = reprojection_error(X, P1, P2, p1[pose.mask], p2[pose.mask])
        thr_rows.append({"pair": name, "ransac_thresh_px": thr, "n_matches": len(p1),
                         "n_matches_on_object": int(on_obj.sum()), "n_matches_on_head": int(head.sum()),
                         "n_inliers": pose.n_inliers, "n_inliers_on_object": int((pose.mask & on_obj).sum()),
                         "n_inliers_on_head": int((pose.mask & head).sum()),
                         "head_dist_median_px": round(float(np.median(dist[head])), 2),
                         "reproj_median_px": round(float(np.median(rep)), 3)})
        print(thr_rows[-1])
    save_table(thr_rows, os.path.join(OUT, "threshold_check.csv"))

    # Rough size check. The reconstruction has |t| = 1, and the camera moved
    # step_cm, so 1 unit ~ step_cm. Then the depth of the bust should be about
    # the 35 cm I stood from it, and its height about 20 cm. Both distances are
    # my own rough numbers, so this is only a sanity check.
    rot = rotation_angle_deg(best["pose"].R)
    unit_cm = step_cm(rot)
    obj = on_mask(masks[best["i"]], best["q1"])
    Xo = best["X"][obj]
    depth_cm = float(np.median(Xo[:, 2])) * unit_cm
    height_cm = float(np.percentile(Xo[:, 1], 98) - np.percentile(Xo[:, 1], 2)) * unit_cm
    scale_row = {"pair": name, "rot_deg": round(rot, 1), "one_unit_cm": round(float(unit_cm), 2),
                 "n_points_on_object": int(obj.sum()),
                 "bust_depth_median_cm": round(depth_cm, 1), "bust_height_cm": round(height_cm, 1),
                 "expected_depth_cm": DIST_CM, "expected_height_cm": 20.0}
    print(scale_row)
    save_table([scale_row], os.path.join(OUT, "scale_check.csv"))


if __name__ == "__main__":
    main()
