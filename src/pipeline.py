"""
Complete pipeline that integrates all components:
Detector, motion analysis, vidstab, warp.
"""

from vidstab.VidStab import VidStab

from detector import Detector, vis_detector
from motion import Motion, vis_motion
from utils import *

# TODO vidstab, warp, tiled infer?


class Pipeline:
    def __init__(self):
        self.stab = VidStab()
        self.detector = Detector()
        self.motion = Motion()

        self.frame_i = 0

    def update(self, frame):
        """
        frame: cv2 format.
        """
        print("Pipeline update frame", self.frame_i)

        # Stabilization.
        stab_frame = self.stab.stabilize_frame(input_frame=frame, smoothing_window=STAB_WINDOW)
        if self.frame_i >= STAB_WINDOW:
            frame = stab_frame

        # Run detector.
        if self.frame_i % DETECT_INTERVAL == 0:
            detect_out = self.detector.update(frame)
            #vis_detector(frame, detect_out)

        # Run motion analysis.
        motion_out = self.motion.update(frame)
        vis_motion(frame, motion_out)

        self.frame_i += 1

    def init_warp(self):
        """
        Make perspective correction.
        Shrink lower edge by factor, forming a trapezoid shape.
        """
        length = int(RES[0] * WARP_CORRECTION)
        rect1 = np.array([
            [0, 0],
            [RES[0], 0],
            [RES[0], RES[1]],
            [0, RES[1]],
        ], dtype=np.float32)
        rect2 = np.array([
            [0, 0],
            [RES[0], 0],
            [RES[0] / 2 + length / 2, RES[1]],
            [RES[0] / 2 - length / 2, RES[1]],
        ], dtype=np.float32)

        self.warp_mat = cv2.getPerspectiveTransform(rect1, rect2)
        self.inv_warp_mat = cv2.getPerspectiveTransform(rect2, rect1)
