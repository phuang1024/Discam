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
        self.detector = Detector(field_mask_path)
        self.motion = Motion(field_mask_path)
        self.static_bbox = StaticBBox()

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

        # Run motion analysis.
        motion_out = self.motion.update(frame)
        #vis_motion(frame, motion_out)

        # Run detector.
        if self.detect_out is None or self.frame_i % DETECT_INTERVAL == 0:
            self.detect_out = self.detector.update(frame, motion_out)
            vis_detector(frame, self.detect_out, motion_out)

        # Run static bbox.
        static_bbox_out = self.static_bbox.update(self.detect_out)#, motion_out)
        #vis_static_bbox(frame, static_bbox_out)

        self.frame_i += 1

        return {
            "frame_i": self.frame_i - 1,
            "static_bbox": static_bbox_out["bbox"],
        }
