"""
Dense optical flow.
"""

import cv2
import numpy as np

video = "../data/videos/BoomNov10_part.mp4"
cap = cv2.VideoCapture(video)

ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (960, 540))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    for _ in range(5):
        ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Frame", frame)
    cv2.imshow("Optical Flow", bgr)
    if cv2.waitKey(1) == ord("q"):
        break
    prev_gray = gray

cap.release()
