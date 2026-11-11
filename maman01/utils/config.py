"""
config.py
All parameters in one place. The same values are used for all three methods.
"""

TAU = 2.0                 # Sampson threshold in pixels (consistent if d_S < TAU^2)
RATIO_TEST = 0.75         # Lowe ratio test
RANSAC_PROB = 0.999       # findEssentialMat confidence
RANSAC_THRESH = 1.0       # findEssentialMat threshold [px]
SIFT_N_FEATURES = 0       # 0 = no limit
ORB_N_FEATURES = 5000     # default is 500, too few compared to SIFT
SP_MAX_KEYPOINTS = 4096   # SuperPoint keypoint limit
N_IMAGES = 11

TIMING_WARMUP = 1         # runs that are not counted
TIMING_RUNS = 5           # timed runs, the median is reported

# OpenCV's RANSAC uses a fixed seed and picks samples by their position in
# the list, so the result depends on the order of the matches. I run it on
# RANSAC_REPEATS shuffled orders and report the median.
RANSAC_REPEATS = 30
RANSAC_SEED = 0           # seed of my shuffles, same on every computer

METHODS = ("sift", "orb", "superpoint")

# task 3: a 3D point is dropped if the angle between its two viewing rays is
# smaller than this (almost parallel rays = unreliable depth)
MIN_TRI_ANGLE_DEG = 2.0
