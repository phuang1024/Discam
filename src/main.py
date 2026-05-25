import argparse

import cv2
import torch

from constants import *
from bounding_box import StaticBBox, vis_static_bbox
from pipeline import CVPipeline, vis_pipeline
from video_read import ScaledReader

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    video = ScaledReader(args.video)
    pipe = CVPipeline()
    static_bbox = StaticBBox()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, RES)

        pipe.update(frame)
        static_bbox.update(pipe)
        vis_pipeline(pipe)
        #vis_static_bbox(static_bbox, frame)


if __name__ == "__main__":
    main()
