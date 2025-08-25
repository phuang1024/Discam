"""
Utils for computing difference between frames,
using frame alignment.
"""

import cv2
import numpy as np


def frame_diff(img1, img2):
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

    diff = cv2.absdiff(aligned, img2)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow("Diff", diff)
    cv2.waitKey(100)

