"""
Utilities for parsing the points.txt file, used in crop.py
"""

from dataclasses import dataclass


@dataclass
class Game:
    points: list["Point"]
    title: str = ""


@dataclass
class Point:
    start: int = 0
    end: int = 0
    line: str = ""
    score: int = 0
    result: str = ""


def parse_file(path: str):
    game = Game(points=[])

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            key, value = line.split(maxsplit=1)
            value = value.replace("\\n", "\n")

            if key == "title":
                game.title = value
            elif key == "point":
                start, end = map(int, value.split())
                game.points.append(Point(start=start, end=end))
            elif key == "line":
                game.points[-1].line = value
            elif key == "score":
                game.points[-1].score = int(value)
            elif key == "result":
                game.points[-1].result = value

    return game
