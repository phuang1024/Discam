"""
Utils for computing difference between frames.

Procedure:

1. Frame difference:
Let img1, img2 be two frames.
Align img1 and img2 with ORB feature matching.
diff = abs(img1 - img2)

2. Temporal filters:
   There are two goals with this:
   - Reduce transient noise.
   - Amplify changes over longer spaces.
   We keep an excitability / inhibition map.
diff = EMA(diff)
inhibit = EMA(diff)  # Double EMA, this one with higher alpha.
diff = diff - inhibit

3. Bounding box:
diff = downsample(diff)
diff = median_blur(diff)
diff = diff > threshold
bbox = bbox(diff)
"""

import cv2
import numpy as np

from utils import EMA


def align_frame(img1, img2):
    """
    Aligh img1 to img2 using ORB feature matching.

    Returns aligned img1.
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe’s ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    h, w, _ = img2.shape
    aligned = cv2.warpAffine(img1, M, (w, h))
    return aligned


def frame_diff(img1, img2, mult=5, add=0):
    """
    Aligns frames and computes absolute difference.
    Adds linear scaling/offset, and clips.

    return: Grayscale image of magnitude of difference.
        ndarray float32 (H, W) [0, 1]
    """
    aligned = align_frame(img1, img2)

    diff = cv2.absdiff(aligned, img2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    diff = (diff + add) * mult
    diff = np.clip(diff, 0, 1)

    return diff


class DiffFilter:
    """
    Temporal filter for frame diffs.
    """

    def __init__(self, inhibit_weight=1):
        self.diff_ema = EMA(0.6)
        self.inhibit_ema = EMA(0.9)
        self.inhibit_weight = inhibit_weight

    def process(self, diff, mult=2):
        """
        Process new diff frame.

        diff: ndarray float32 (H, W) [0, 1] image.
        return: Filtered diff.
            ndarray float32 (H, W) [0, 1] image.
        """
        diff = self.diff_ema.update(diff)
        inhibit = self.inhibit_ema.update(diff)

        # Dialate inhibit
        kernel = np.ones((5, 5), np.uint8)
        inhibit = cv2.dilate(inhibit, kernel, iterations=2)

        diff = diff - inhibit * self.inhibit_weight

        diff = diff * mult
        diff = np.clip(diff, 0, 1)

        return diff


def compute_bbox(diff, thres, downsample=4, blur=3, padding=50):
    """
    Compute bounding box of salient areas
    using techniques to reduce noise.

    diff: ndarray float32 (H, W) [0, 1] image.
    thres: Threshold, between 0 and 1.
    downsample: Downsample factor.
    blur: Box blur kernel size.
    padding: Padding in pixels.
    return: (x1, y1, x2, y2) bbox.
        None if no salient area found.
    """
    diff = (diff * 255).astype(np.uint8)
    h, w = diff.shape
    diff = cv2.resize(diff, (w // downsample, h // downsample), cv2.INTER_AREA)

    diff = cv2.blur(diff, (blur, blur), 0)

    thres = int(thres * 255)
    diff = diff > thres

    ys, xs = np.where(diff)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 * downsample - padding)
    y1 = max(0, y1 * downsample - padding)
    x2 = min(w, x2 * downsample + padding)
    y2 = min(h, y2 * downsample + padding)

    return (x1, y1, x2, y2)
