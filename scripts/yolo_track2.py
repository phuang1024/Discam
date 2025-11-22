"""
Frame by frame tracking.
Test frame step, camera jitter, etc..
"""

from hashlib import sha256

import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = "../data/videos/BoomNov10_part.mp4"
cap = cv2.VideoCapture(video)

while True:
    for _ in range(4):
        ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, conf=0.02, persist=True)[0]

    boxes = result.boxes.xywh.cpu()
    track_ids = result.boxes.id.int().cpu()

    for box, track_id in zip(boxes, track_ids):
        digest = int(sha256(str(track_id.item()).encode()).hexdigest(), 16)
        color = (
            digest % 256,
            (digest // 256) % 256,
            (digest // 65536) % 256,
        )
        x, y, w, h = box
        cv2.rectangle(
            frame,
            (int(x - w / 2), int(y - h / 2)),
            (int(x + w / 2), int(y + h / 2)),
            color,
            2,
        )

    cv2.imshow("a", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
