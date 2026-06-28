"""
Connects all CV components together.
Also visualization utils.
"""

import cv2
import numpy as np
from tqdm import tqdm

from cv.classify import Classifier
from cv.detect import Detector
from cv.perspective import ComputePersp
from utils.constants import *


class Pipeline:
    def __init__(self, mask_path):
        self.detector = Detector()
        self.perspective = ComputePersp(mask_path)
        self.classifier = Classifier(mask_path)

    def update(self, frame, frame_i):
        """
        frame: cv2 format.
        """
        boxes = self.detector.update(frame)
        self.perspective.update(boxes)
        active_inds = self.classifier.update(boxes, frame_i)
        #vis_pipeline(frame, boxes, active_inds, self.classifier.field_mask)

        active_boxes = boxes[active_inds]
        return active_boxes


def post_run_pipeline(video_path, mask_path):
    """
    Run pipeline on video file.
    Respects FPS and RES constants.
    return: (pipe_out, frame_is)
        pipe_out: List of pipeline outputs.
        frame_is: List of frame indices in original video coord.
    """
    video = cv2.VideoCapture(video_path)
    orig_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = int(video.get(cv2.CAP_PROP_FPS))
    orig_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_scale = int(orig_fps / DET_FPS)

    pipeline = Pipeline(mask_path)
    pipe_out = []
    frame_is = []
    frame_i = 0
    pbar = tqdm(total=orig_len // fps_scale, desc="Detector")
    while True:
        for _ in range(fps_scale):
            ret, frame = video.read()
        frame_i += fps_scale
        if not ret:
            break

        if orig_w != DET_RES[0] or orig_h != DET_RES[1]:
            frame = cv2.resize(frame, DET_RES)

        pipe_out.append(pipeline.update(frame, frame_i))
        frame_is.append(frame_i)
        pbar.update(1)

    pbar.close()
    video.release()
    return pipe_out, frame_is


def vis_pipeline(frame, boxes, active_inds, field_mask):
    """
    frame: cv2 format.
    boxes: tracked boxes format.
    active_inds: List of indices of active player boxes.
    tracks: Classifier.tracks
    field_mask: Classifier.field_mask
    """
    frame = frame.copy()

    # Draw tracks.
    """
    for track in tracks.values():
        for i in range(len(track.points) - 1):
            cv2.line(frame, track.points[i], track.points[i+1], (255, 0, 0), 2)
    """

    # Draw boxes.
    for i, box in enumerate(boxes):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

    # Overlay field mask
    field_mask = field_mask / 2 + 0.5
    field_mask = (field_mask * 255).astype(np.uint8)
    field_mask = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.addWeighted(frame, 1.0, field_mask, 0.3, 0)

    cv2.imshow("Classifier", frame)
    cv2.waitKey(1)
