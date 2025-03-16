"""
Script to test HSL dynamic range compression.
"""

import argparse

import cv2


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

        # Convert to HLS
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Split channels
        h, l, s = cv2.split(hls)

        # Apply CLAHE to L channel
        cl = clahe.apply(l)

        # Merge channels
        hls = cv2.merge((h, cl, s))

        # Convert back to BGR
        frame = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

        cv2.imshow("adjusted", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
