"""
Test RT-DETR model for person detection.
"""

import argparse

import cv2
import torch
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

torch.set_grad_enabled(False)

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd", device_map="auto")


def pred_frame(frame):
    """
    frame: cv2 format.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = image_processor(images=frame, return_tensors="pt").to(model.device)
    outupts = model(**inputs)
    results = image_processor.post_process_object_detection(
        outupts,
        target_sizes=torch.tensor([(frame.shape[0], frame.shape[1])]),
        threshold=0.2,
    )
    return results


def anno_frame(frame, results):
    """
    frame: cv2 format.
    results: output of pred_frame.
    """
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            if label == 0:
                box = [round(i, 2) for i in box.tolist()]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    while True:
        for _ in range(5):
            ret, frame = video.read()
        if not ret:
            break

        results = pred_frame(frame)
        anno_frame(frame, results)
        cv2.imshow("RT-DETR", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
