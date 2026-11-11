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

METHODS = ("sift", "orb", "superpoint")
