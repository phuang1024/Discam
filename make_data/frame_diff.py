"""
Utils for computing difference between frames.

Procedure:
1. Align frames using ORB feature matching.
2. Compute absolute difference, and apply threshold and mult.
3. Blur and downsample to reduce noise.
4. Compute bounding box of salient areas.
"""

import cv2
import numpy as np


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


def frame_diff(img1, img2, floor=0.05, mult=5):
    """
    Aligns frames and computes absolute difference.

    return: Grayscale image of magnitude of difference.
        ndarray float32 (H, W) [0, 1]
    """
    aligned = align_frame(img1, img2)

    diff = cv2.absdiff(aligned, img2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    diff = ((diff - floor) * mult).clip(0, 1)

    return diff


def compute_bbox(diff, scale_mask, thres=0.15, downsample=4, blur=5, padding=50):
    """
    Compute bounding box of salient areas
    using techniques to reduce noise.

    diff: ndarray float32 (H, W) [0, 1] image.
    scale_mask: See make_bbox_data.py
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

    scale_mask = scale_mask[::downsample, ::downsample]
    diff = diff * scale_mask

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
