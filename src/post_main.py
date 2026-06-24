"""
Entry point for post-recording video processing.
Crops and trims video.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

#from bounding_box import compute_final_boxes
from cv.pipeline import post_run_pipeline
#from trim import find_trim_sections, gen_timestamps
from utils.constants import *

torch.set_grad_enabled(False)


def write_output(in_path, out_path, bboxes, trim_sections):
    """
    Write output video with crop and trim.
    """
    trim_sections = trim_sections.tolist()

    in_video = cv2.VideoCapture(in_path)
    orig_fps = in_video.get(cv2.CAP_PROP_FPS)
    orig_w = in_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_video = FFmpegWriter(out_path, orig_fps, OUT_RES)

    frame_i = 0
    pbar = tqdm(total=len(bboxes), desc="Writing output")
    while True:
        # Increment at beginning.
        frame_i += 1
        pbar.update(1)
        ret, frame = in_video.read()
        if not ret:
            break

        # Check trim.
        if len(trim_sections) > 0:
            curr_time = (frame_i - 1) / orig_fps
            if curr_time > trim_sections[0][1]:
                trim_sections.pop(0)
            if trim_sections[0][0] <= curr_time <= trim_sections[0][1]:
                continue

        # Get bbox.
        bbox = bboxes[frame_i - 1]
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * orig_w / NN_RES[0])
        x2 = int(x2 * orig_w / NN_RES[0])
        y1 = int(y1 * orig_h / NN_RES[1])
        y2 = int(y2 * orig_h / NN_RES[1])
        # Crop frame
        frame_crop = frame[y1:y2, x1:x2]
        frame_crop = cv2.resize(frame_crop, OUT_RES)
        out_video.write(frame_crop)

        # Draw vis
        """
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("box", vis_frame)
        cv2.imshow("crop", frame_crop)
        cv2.waitKey(1)
        """

    pbar.close()
    in_video.release()
    out_video.release()


def check_file_exists(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)


def get_file_paths(video_path):
    """
    Get file paths based on input video path.
    """
    def with_suffix(suffix):
        return video_path.parent / (video_path.stem + suffix)
    paths = {
        "in": video_path,
        "out": with_suffix(".discout.mp4"),
        "field_mask": with_suffix(".npy"),
        "cache": with_suffix(".discache"),
    }
    check_file_exists(paths["in"])
    check_file_exists(paths["field_mask"])
    paths["cache"].mkdir(exist_ok=True)
    return paths


def run_pipe_wrapper(args, paths):
    """
    Load from cache, or
    run pipeline and save to cache.
    """
    print("Run CV pipeline:")
    cache_file = paths["cache"] / "pipeline_post.pkl"
    if args.no_cache or not cache_file.exists():
        data = post_run_pipeline(paths["in"], paths["field_mask"])
        print(f"    Saving to cache {cache_file}.")
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    else:
        print(f"    Loading from cache {cache_file}.")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--no_cache", action="store_true", help="Don't read from cache.")
    args = parser.parse_args()

    # Get paths.
    paths = get_file_paths(args.video)
    print(f"Discpost: Discam version {VERSION}",
          f"    Input video: {paths['in']}",
          f"    Output video: {paths['out']}",
          f"    Field mask: {paths['field_mask']}", sep="\n")

    # Get video info.
    cap = cv2.VideoCapture(str(paths["in"]))
    orig_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Input video:",
          f"    Frame count: {orig_len}",
          f"    FPS: {orig_fps}", sep="\n")

    # Run pipeline.
    pipe_out, frame_is = run_pipe_wrapper(args, paths)
    stop

    print("Compute bounding boxes.")
    boxes = compute_final_boxes(detector_outs, frame_count, out_fps)

    trim_sections = find_trim_sections(detect_out)
    print(f"Found trim sections: ", end="")
    for x in trim_sections:
        print(x, end=" ")
    print()

    ts_string = gen_timestamps(trim_sections)
    ts_file = args.video.parent / (args.video.stem + ".ts.txt")
    print(f"    Writing timestamps to {ts_file}")
    with open(ts_file, "w") as f:
        f.write(ts_string)

    print("Write output video.")
    write_output(in_path, out_path, boxes, trim_sections)


if __name__ == "__main__":
    main()
