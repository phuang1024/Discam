"""
Utilities for parsing the points.txt file, used in crop.py
"""

from dataclasses import dataclass

import cv2


@dataclass
class Game:
    points: list["Point"]
    title: list[str]
    """Each element on new line."""


@dataclass
class Point:
    # Frame start in original video.
    start: int = 0
    # Frame start after cropping out non-points, in the output file.
    start_post_crop: int = 0
    # Frame end in original video.
    end: int = 0
    line: str = ""
    score: int = 0
    result: str = ""


def parse_file(path: str, fps: float):
    game = Game(points=[], title=[])

    with open(path, "r") as f:
        frame_post_crop = 0
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            key, value = line.split(maxsplit=1)

            if key == "title":
                game.title = value.split("\\n")
            elif key == "point":
                start, end = value.split()
                start = int(parse_timestamp(start) * fps)
                end = int(parse_timestamp(end) * fps)
                game.points.append(Point(start=start, start_post_crop=frame_post_crop, end=end))
                frame_post_crop += end - start
            elif key == "line":
                game.points[-1].line = value
            elif key == "score":
                game.points[-1].score = int(value)
            elif key == "result":
                game.points[-1].result = value

    check_no_overlap(game)
    return game


def parse_timestamp(ts: str) -> int:
    """
    Returns the seconds equivalent of a timestamp in one of the formats:
    - S
    - M:S
    - H:M:S
    """
    parts = list(map(int, ts.split(":")))
    assert 1 <= len(parts) <= 3, "Invalid timestamp format"
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def check_no_overlap(game):
    """
    Ensure a point ends before the next begins.
    """
    end_frame = -1
    for p in game.points:
        if p.start < end_frame:
            raise ValueError(f"Point overlap detected at {p}")
        end_frame = p.end


def get_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
