"""
Utilities for parsing the points.txt file, used in crop.py
"""

from dataclasses import dataclass


@dataclass
class Game:
    points: list["Point"]
    title: list[str]
    """Each element on new line."""


@dataclass
class Point:
    start: int = 0
    start_post_crop: int = 0
    """Frame start after cropping out non-points."""
    end: int = 0
    line: str = ""
    score: int = 0
    result: str = ""


def parse_file(path: str):
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
                start, end = map(int, value.split())
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


def check_no_overlap(game):
    """
    Ensure a point ends before the next begins.
    """
    end_frame = -1
    for p in game.points:
        if p.start < end_frame:
            raise ValueError(f"Point overlap detected at {p}")
        end_frame = p.end
