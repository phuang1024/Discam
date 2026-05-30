"""
Pipeline that integrates all computer vision components.
Input: Video stream that's already scaled.
Output: Bounding boxes at select frames.
"""

from vidstab.VidStab import VidStab

from static_bbox import StaticBBox, vis_static_bbox
from detect import Detector, vis_detector
from motion import Motion, vis_motion
from utils import *

# TODO vidstab, warp, tiled infer?


class Pipeline:
    def __init__(self, field_mask_path):
        self.stab = VidStab()
        self.detector = Detector()
        self.motion = Motion()
        self.static_bbox = StaticBBox(field_mask_path)

        self.detect_out = None

        self.frame_i = 0

    def update(self, frame):
        """
        frame: cv2 format.
        return {
            frame_i: int,
            static_bbox: xyxy bbox, tuple of ints.
                Pixel values are with respect to RES.
        }
        """
        # Stabilization.
        stab_frame = self.stab.stabilize_frame(input_frame=frame, smoothing_window=STAB_WINDOW)
        if self.frame_i >= STAB_WINDOW:
            frame = stab_frame

        # Run detector.
        if self.detect_out is None or self.frame_i % DETECT_INTERVAL == 0:
            self.detect_out = self.detector.update(frame)
            vis_detector(frame, self.detect_out)

        # Run motion analysis.
        motion_out = self.motion.update(frame)
        #vis_motion(frame, motion_out)

        # Run static bbox.
        static_bbox_out = self.static_bbox.update(self.detect_out, motion_out)
        #vis_static_bbox(frame, static_bbox_out)

        self.frame_i += 1

        return {
            "frame_i": self.frame_i - 1,
            "static_bbox": static_bbox_out["bbox"],
        }

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
