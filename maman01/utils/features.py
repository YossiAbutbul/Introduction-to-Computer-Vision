"""
features.py
Keypoint detection, description and matching for SIFT, ORB and SuperPoint,
plus timing. Used in tasks 1 and 2.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import (METHODS, ORB_N_FEATURES, RATIO_TEST, SIFT_N_FEATURES,
                     SP_MAX_KEYPOINTS, TIMING_RUNS, TIMING_WARMUP)
from .dataset import image_path, load_image


_SUPERPOINT = None          # loaded once, so loading time is not in the timing


def _get_superpoint():
    """
    Load the pretrained SuperPoint from the lightglue package (only once).
    Only the extractor is used, matching is done with BFMatcher like the
    other methods.
    """
    global _SUPERPOINT
    if _SUPERPOINT is None:
        import torch
        from lightglue import SuperPoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SuperPoint(max_num_keypoints=SP_MAX_KEYPOINTS).eval().to(device)
        _SUPERPOINT = (model, device)
    return _SUPERPOINT


def superpoint_device_name():
    """'cpu' or 'cuda'"""
    return str(_get_superpoint()[1])


def load_detector_input(idx, method):
    """
    Load the image in the format each method needs:
      sift / orb  -> grayscale uint8 array
      superpoint  -> torch tensor (3, H, W), values in [0, 1]
    Done before the timer starts, so reading the file is not timed.
    """
    method = method.lower()
    if method in ("sift", "orb"):
        return load_image(idx, color=False)
    if method == "superpoint":
        from lightglue.utils import load_image as lg_load_image
        _, device = _get_superpoint()
        return lg_load_image(image_path(idx)).to(device)
    raise ValueError("unknown method %r (expected one of %s)" % (method, METHODS))


def detect_and_describe(image_input, method):
    """
    Returns:
      kp         (N,2) keypoint coordinates in the original image (pixels)
      desc       (N,D) descriptors
      norm_type  NORM_L2 for SIFT/SuperPoint, NORM_HAMMING for ORB (binary)

    SuperPoint: lightglue resizes the image so the long side is 1024 before
    running the network, and then scales the keypoints back to the original
    size. So the coordinates can be compared directly to SIFT/ORB.
    """
    method = method.lower()

    if method == "sift":
        det = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)
        kp, desc = det.detectAndCompute(image_input, None)
        return np.float32([k.pt for k in kp]), desc, cv2.NORM_L2

    if method == "orb":
        det = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
        kp, desc = det.detectAndCompute(image_input, None)
        return np.float32([k.pt for k in kp]), desc, cv2.NORM_HAMMING

    if method == "superpoint":
        extractor, _ = _get_superpoint()
        feats = extractor.extract(image_input)      # extract() already runs with no_grad
        kp = feats["keypoints"][0].cpu().numpy().astype(np.float32)
        desc = feats["descriptors"][0].cpu().numpy().astype(np.float32)
        return kp, desc, cv2.NORM_L2

    raise ValueError("unknown method %r (expected one of %s)" % (method, METHODS))


def match_descriptors(desc1, desc2, norm_type, strategy="ratio", ratio=RATIO_TEST):
    """
    strategy = "ratio": keep a match if best distance < ratio * second best.
    strategy = "mnn":   mutual nearest neighbors (crossCheck=True).
    Returns two index arrays (into desc1 and desc2).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    if strategy == "ratio":
        bf = cv2.BFMatcher(norm_type)
        knn = bf.knnMatch(desc1, desc2, k=2)
        good = [pair[0] for pair in knn
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    elif strategy == "mnn":
        bf = cv2.BFMatcher(norm_type, crossCheck=True)
        good = bf.match(desc1, desc2)
    else:
        raise ValueError("unknown strategy %r (expected 'ratio' or 'mnn')" % strategy)

    return (np.array([m.queryIdx for m in good], dtype=int),
            np.array([m.trainIdx for m in good], dtype=int))


@dataclass
class MatchResult:
    method: str
    idx1: int
    idx2: int
    pts1: np.ndarray      # (N,2) matched points in image 1
    pts2: np.ndarray      # (N,2) matched points in image 2
    n_kp1: int
    n_kp2: int
    elapsed: float        # seconds (detection + description + matching)

    @property
    def n_matches(self):
        return len(self.pts1)


def match_pair(idx1, idx2, method, strategy="ratio", ratio=RATIO_TEST, inputs=None):
    """
    Detect, describe and match one image pair.
    The timer includes only detection + description + matching (not file
    reading and not model loading). `inputs` lets me reuse already loaded
    images when timing several runs.
    """
    if inputs is None:
        inputs = (load_detector_input(idx1, method), load_detector_input(idx2, method))
    if method.lower() == "superpoint":
        _get_superpoint()          # load the model before starting the timer

    start = time.perf_counter()
    kp1, desc1, norm_type = detect_and_describe(inputs[0], method)
    kp2, desc2, _ = detect_and_describe(inputs[1], method)
    m1, m2 = match_descriptors(desc1, desc2, norm_type, strategy=strategy, ratio=ratio)
    elapsed = time.perf_counter() - start

    return MatchResult(method=method, idx1=idx1, idx2=idx2,
                       pts1=kp1[m1].astype(np.float64).reshape(-1, 2),
                       pts2=kp2[m2].astype(np.float64).reshape(-1, 2),
                       n_kp1=len(kp1), n_kp2=len(kp2), elapsed=elapsed)


def time_match_pair(idx1, idx2, method, strategy="ratio", ratio=RATIO_TEST,
                    runs=TIMING_RUNS, warmup=TIMING_WARMUP):
    """
    Run `warmup` runs that are not counted (the first run is usually slower),
    then `runs` timed runs.
    Returns (result of last run, array of the timed runs in seconds).
    I report the median: one slow run (e.g. something else running on the
    computer) changes the mean a lot but the median almost not at all.
    """
    inputs = (load_detector_input(idx1, method), load_detector_input(idx2, method))
    for _ in range(warmup):
        match_pair(idx1, idx2, method, strategy, ratio, inputs=inputs)

    times, result = [], None
    for _ in range(runs):
        result = match_pair(idx1, idx2, method, strategy, ratio, inputs=inputs)
        times.append(result.elapsed)
    return result, np.array(times)
