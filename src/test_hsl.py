"""
Script to test HSL dynamic range compression.
"""

import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("original", frame)

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)

        l_clahe = clahe.apply(l)
        hls = cv2.merge((h, l_clahe, s))
        frame = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
        cv2.imshow("clahe", frame)

        l_uniform = np.full_like(l, 128)
        hls = cv2.merge((h, l_uniform, s))
        frame = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
        cv2.imshow("uniform", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
