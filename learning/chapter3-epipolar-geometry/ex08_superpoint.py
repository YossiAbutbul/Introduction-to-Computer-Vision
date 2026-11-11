# ex08_superpoint.py
# Detect and describe keypoints with a pretrained SuperPoint model,
# then match them with the same Ratio Test (L2) as SIFT, and measure the time.

import os
import time
import cv2
import numpy as np
import torch
from lightglue import SuperPoint
from lightglue.utils import load_image

script_dir = os.path.dirname(os.path.abspath(__file__))

# Use the GPU if available, otherwise the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Load the pretrained model (not included in the timing)
extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)

# Load images as tensors (3, H, W), values in [0, 1]
image1 = load_image(os.path.join(script_dir, "0005.jpg")).to(device)
image2 = load_image(os.path.join(script_dir, "0006.jpg")).to(device)

# Start timing: detection + description + matching
start = time.perf_counter()

# Run the network; no_grad because we only do inference, not training
with torch.no_grad():
    feats1 = extractor.extract(image1)
    feats2 = extractor.extract(image2)

# Convert to NumPy: keypoints (N, 2) in original image pixels, descriptors (N, 256)
kp1 = feats1["keypoints"][0].cpu().numpy()
kp2 = feats2["keypoints"][0].cpu().numpy()
des1 = feats1["descriptors"][0].cpu().numpy().astype(np.float32)
des2 = feats2["descriptors"][0].cpu().numpy().astype(np.float32)

# Real-valued descriptors -> L2 distance, same Ratio Test as SIFT (0.75)
bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]

elapsed = time.perf_counter() - start

# Matched point coordinates, ready for findEssentialMat later
pts1 = np.float32([kp1[m.queryIdx] for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx] for m in good_matches])

print("keypoints in image 1:", len(kp1))
print("keypoints in image 2:", len(kp2))
print("descriptor shape:", des1.shape)
print("good matches after ratio test:", len(good_matches))
print("time (detect + describe + match) [s]:", round(elapsed, 3))