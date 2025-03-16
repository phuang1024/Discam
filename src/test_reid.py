"""
Stream a video, and track a single person.
"""

import argparse
import random

import cv2
import torch

from features import run_yolo
from reid import reid_model


def prepare_reid_input(frame, preds):
    inputs = []
    for i in range(len(preds.cls)):
        x1, y1, x2, y2 = map(int, preds.xyxy[i])
        person = frame[y1:y2, x1:x2]
        person = cv2.resize(person, (128, 256))
        person = torch.tensor(person, dtype=torch.float32)
        person = person.permute(2, 0, 1) / 128.0 - 1.0
        inputs.append(person)

    return torch.stack(inputs).to("cuda") if inputs else None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)

    target_embed = None
    i = 0
    color = None
    while True:
        for _ in range(args.interval):
            ret, frame = video.read()
            if not ret:
                return

        preds = run_yolo(frame)
        preds = preds[0].boxes

        persons = prepare_reid_input(frame, preds)
        if persons is not None:
            results = reid_model(persons)

            if i % 400 == 0:
                res_i = random.randint(0, len(results) - 1)
                target_embed = results[res_i].unsqueeze(0)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                print("Recompute target")

            else:
                dists = torch.nn.functional.cosine_similarity(target_embed, results)
                best_i = torch.argmax(dists).item()

                x1, y1, x2, y2 = map(int, preds.xyxy[best_i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("frame", frame)
        cv2.waitKey(10)

        i += 1


if __name__ == "__main__":
    main()
