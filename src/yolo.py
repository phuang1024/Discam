import argparse

import cv2
from ultralytics import YOLO

yolo = YOLO("yolo11n.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    video = cv2.VideoCapture(args.file)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        preds = yolo.predict(source=frame, save=True)
        print(type(preds))
        print(preds)

        cv2.imshow("frame", frame)
        if cv2.waitKey(0) == ord("q"):
            break


if __name__ == "__main__":
    main()
