import argparse

import cv2


parser = argparse.ArgumentParser()
parser.add_argument("video")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

cap.release()
