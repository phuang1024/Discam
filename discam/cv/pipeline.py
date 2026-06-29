"""
Connects all CV components together.
Also visualization utils.
"""

import cv2
import numpy as np
from tqdm import tqdm

from .classify import Classifier
from .detect import Detector
from .perspective import ComputePersp
from ..utils.constants import *


class Pipeline:
    def __init__(self, mask_path):
        self.detector = Detector()
        self.perspective = ComputePersp(mask_path)
        self.classifier = Classifier()

    def update(self, frame, frame_i):
        """
        frame: cv2 format.
        frame_i: Frame index number.
        """
        boxes = self.detector.update(frame)
        player_locs, mask_locs = self.perspective.update(boxes)
        active_inds = self.classifier.update(player_locs, mask_locs)

        vis_frame(frame, boxes, active_inds)
        vis_locations(player_locs, mask_locs, active_inds, self.perspective)

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


def vis_frame(frame, boxes, active_inds):
    """
    Visualize image frame detections.
    frame: cv2 format.
    boxes: boxes format.
    active_inds: Bool array of whether each box is active.
    """
    frame = frame.copy()
    # Draw boxes.
    for i, box in enumerate(boxes):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

    cv2.imshow("Pipeline", frame)
    cv2.waitKey(1)


def vis_locations(locs, mask_locs, active_inds, persp: ComputePersp):
    """
    Visualize physical locations.
    """
    RES = 800

    def exterp(value, from_min, from_max, to_min, to_max):
        """Linear interp that supports extrap."""
        return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

    def interp_coords(coords):
        """From XY location to vis image pixel pos."""
        return (
            exterp(coords[:, 0], -50, 50, 0, RES),
            exterp(coords[:, 1], 0, 100, RES, 0),
        )

    img = np.full((RES, RES, 3), 255, dtype=np.uint8)

    # Overall camera FOV cone.
    view_points = np.array(((0, 0), (DET_RES[0], 0), (DET_RES[0], DET_RES[1]), (0, DET_RES[1])), dtype=float)
    view_locs = persp.compute_locations(view_points)
    view_locs = interp_coords(view_locs)
    view_locs = np.array(view_locs, dtype=int).swapaxes(0, 1)
    cv2.fillPoly(img, [view_locs], (200, 200, 200))

    # Field mask.
    mask_locs = interp_coords(mask_locs)
    mask_locs = np.array(mask_locs, dtype=int).swapaxes(0, 1)
    cv2.polylines(img, [mask_locs], True, (255, 0, 0), 4)

    # Players.
    xs, ys = interp_coords(locs)
    for i, (x, y) in enumerate(zip(xs, ys)):
        color = (0, 255, 0) if active_inds[i] else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), 5, color, -1)

    cv2.imshow("Locations", img)
    cv2.waitKey(1)
