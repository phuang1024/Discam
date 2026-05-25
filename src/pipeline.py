"""

"""

from detector import Detector, vis_detector
from utils import *


class Pipeline:
    """
    Full NN to bounding box pipeline.
    """

    def __init__(self):
        self.detector = Detector()

        self.frame_i = 0

    def update(self, frame):
        """
        frame: cv2 format.
        """
        print("Pipeline update frame", self.frame_i)

        if self.frame_i % DETECT_INTERVAL == 0:
            detect_out = self.detector.update(frame)
            vis_detector(frame, detect_out)

        self.frame_i += 1
