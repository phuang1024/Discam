"""
Pipeline running for Post Processing.
"""

import cv2
import numpy as np
from tqdm import tqdm

from ..cv.classify import Classifier
from ..cv.detect import Detector
from ..cv.perspective import ComputePersp
from ..cv.vis import vis_frame, vis_locations
from ..utils import logger
from ..utils.constants import *


class Pipeline:
    def __init__(self, mask_path):
        self.detector = Detector()
        self.perspective = ComputePersp(mask_path)
        self.classifier = Classifier()

    def update(self, frame, frame_i):
        """
        frame: cv2 format.
        frame_i: Index of frame.
        """
        boxes = self.detector.update(frame)
        player_locs, mask_locs = self.perspective.update(boxes)
        active_inds = self.classifier.update(player_locs, mask_locs)
        active_boxes = boxes[active_inds]

        # Vis and logging.
        if logger.enabled:
            det_vis = vis_frame(frame, boxes, active_inds)
            locs_vis = vis_locations(player_locs, mask_locs, active_inds, self.perspective)
            logger.add_image("detections_vis", det_vis, frame_i)
            logger.add_image("locations_vis", locs_vis, frame_i)
            cv2.imshow("Detections", det_vis)
            cv2.imshow("Locations", locs_vis)
            cv2.waitKey(1)

            logger.add_scalar("num_active", np.sum(active_inds), frame_i)
            logger.add_scalar("persp_vanishing", self.perspective.vanishing, frame_i)
            logger.add_scalar("persp_min_dist", self.perspective.dist_min, frame_i)

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
