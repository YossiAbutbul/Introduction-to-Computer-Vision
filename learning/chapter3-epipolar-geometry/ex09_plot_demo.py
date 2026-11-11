# ex09_plot_demo.py
# Demo of the Task 2 plot style, using made-up numbers (NOT real results).

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# Image 0000 is matched to images 0001..0010
image_ids = np.arange(1, 11)

# Illustrative values only
e_R = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.9, 1.3, 2.0, 3.5]
e_t = [0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.6, 2.4, 3.8, 6.0]
n_inliers = [2100, 1900, 1650, 1400, 1100, 850, 600, 400, 250, 120]
r_inliers = [0.95, 0.93, 0.90, 0.85, 0.78, 0.70, 0.60, 0.48, 0.35, 0.20]

# Graph 1: count and ratio in two subplots (different units)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(image_ids, n_inliers, marker="o")
ax1.set_xlabel("image index (matched to 0000)")
ax1.set_ylabel("number of consistent matches")
ax1.set_title("N_inliers")
ax1.set_xticks(image_ids)
ax1.grid(True)

ax2.plot(image_ids, r_inliers, marker="o", color="darkorange")
ax2.set_xlabel("image index (matched to 0000)")
ax2.set_ylabel("ratio of consistent matches")
ax2.set_title("r_inliers")
ax2.set_xticks(image_ids)
ax2.set_ylim(0, 1)
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "demo_inliers.png"), dpi=150)

# Graph 2: both errors are in degrees, so one plot is fine
plt.figure(figsize=(7, 4))
plt.plot(image_ids, e_R, marker="o", label="rotation error e_R")
plt.plot(image_ids, e_t, marker="s", label="translation direction error e_t")
plt.xlabel("image index (matched to 0000)")
plt.ylabel("error [deg]")
plt.title("Pose estimation error vs. image index")
plt.xticks(image_ids)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "demo_errors.png"), dpi=150)

plt.show()