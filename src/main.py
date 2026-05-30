import argparse

import torch
from tqdm import tqdm

from pipeline import Pipeline
from video_read import ScaledReader

torch.set_grad_enabled(False)


def run_cv_pipeline(args):
    """
    Create and run pipeline on all frames.
    return: Sequential list of dict.
        Each dict is a return value from Pipeline.update
    """
    video = ScaledReader(args.video)
    pipe = Pipeline(args.field_mask)

    pipe_outputs = []
    pbar = tqdm(total=video.get_len(), desc="CV pipeline")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        pipe_outputs.append(pipe.update(frame))
        pbar.update(1)

    return pipe_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--field_mask")
    args = parser.parse_args()

    pipe_outputs = run_cv_pipeline(args)


if __name__ == "__main__":
    main()
