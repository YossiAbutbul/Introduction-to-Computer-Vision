"""
plots.py
Figures and CSV tables: match images, result tables, point clouds.
"""

import csv
import os

import cv2
import numpy as np

from .dataset import ensure_dir, load_image


def draw_matches(idx1, idx2, pts1, pts2, out_path, mask=None, max_draw=150, seed=0):
    """
    Two images side by side with lines between matched points.
    With a mask: green = consistent, red = not consistent.
    Only `max_draw` random matches are drawn (fixed seed), otherwise the
    image is too crowded.
    """
    img1 = load_image(idx1, color=True)
    img2 = load_image(idx2, color=True)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    pts1 = np.asarray(pts1).reshape(-1, 2)
    pts2 = np.asarray(pts2).reshape(-1, 2)
    order = np.arange(len(pts1))
    if len(order) > max_draw:
        order = np.random.default_rng(seed).choice(order, size=max_draw, replace=False)

    for i in order:
        p = (int(round(pts1[i, 0])), int(round(pts1[i, 1])))
        q = (int(round(pts2[i, 0])) + w1, int(round(pts2[i, 1])))
        color = (0, 255, 0) if mask is None or mask[i] else (0, 0, 255)   # BGR
        cv2.line(canvas, p, q, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, q, 5, color, -1, cv2.LINE_AA)

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    cv2.imwrite(out_path, canvas)
    return out_path


def save_table(rows, out_path, fieldnames=None):
    """List of dicts -> CSV. Columns are taken from the first row by default."""
    if not rows:
        raise ValueError("save_table: no rows")
    fieldnames = fieldnames or list(rows[0].keys())
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return out_path


def plot_point_clouds(X_gt, X_est, out_path, labels=("ground truth", "estimated"),
                      title=None, point_size=2):
    """
    Both clouds on the same axes: 3D view, top view (X-Z) and front view
    (X vs -Y, because the camera Y axis points down).
    The points are semi-transparent because the two clouds almost overlap and
    otherwise one hides the other. The legend uses fixed-size markers since the
    plotted points are very small.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    X_gt = np.asarray(X_gt).reshape(-1, 3)
    X_est = np.asarray(X_est).reshape(-1, 3)
    kw = dict(s=point_size, alpha=0.55, linewidths=0)

    def legend_handles(include_camera):
        handles = [Line2D([], [], marker="o", ls="", ms=7, color="tab:blue",
                          label=labels[0]),
                   Line2D([], [], marker="o", ls="", ms=7, color="tab:red",
                          label=labels[1])]
        if include_camera:
            handles.append(Line2D([], [], marker="^", ls="", ms=9, color="black",
                                  label="camera 1"))
        return handles

    fig = plt.figure(figsize=(16, 5.5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(X_est[:, 0], X_est[:, 1], X_est[:, 2], c="tab:red", **kw)
    ax.scatter(X_gt[:, 0], X_gt[:, 1], X_gt[:, 2], c="tab:blue", **kw)
    ax.set_xlabel("X [dataset units]")
    ax.set_ylabel("Y [dataset units]")
    ax.set_zlabel("Z [dataset units]")
    ax.set_title("3D view (camera 1 frame)")
    ax.legend(handles=legend_handles(False), loc="upper right")

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(X_est[:, 0], X_est[:, 2], c="tab:red", **kw)
    ax.scatter(X_gt[:, 0], X_gt[:, 2], c="tab:blue", **kw)
    ax.plot([0], [0], marker="^", ms=9, color="black")
    ax.set_xlabel("X [dataset units]")
    ax.set_ylabel("Z = depth [dataset units]")
    ax.set_title("top view (X-Z)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles(True))

    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(X_est[:, 0], -X_est[:, 1], c="tab:red", **kw)
    ax.scatter(X_gt[:, 0], -X_gt[:, 1], c="tab:blue", **kw)
    ax.set_xlabel("X [dataset units]")
    ax.set_ylabel("-Y = up [dataset units]")
    ax.set_title("front view (X vs -Y)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles(False))

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
