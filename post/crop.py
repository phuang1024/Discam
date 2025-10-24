import argparse

import cv2

from pointstxt import parse_file

TITLE_TIME = 24 * 3
LINE_TIME = 24 * 3
RESULT_TIME = 24 * 3


def check_no_overlap(game):
    """
    Ensure a point ends before the next begins.
    """
    end_frame = -1
    for p in game.points:
        if p.start < end_frame:
            raise ValueError(f"Point overlap detected at {p}")
        end_frame = p.end


def crop(args, game):
    """
    Crop and annotate.
    """
    def put_text(frame, text, pos):
        cv2.putText(frame, text, pos,
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    in_video = cv2.VideoCapture(args.input)
    out_video = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        in_video.get(cv2.CAP_PROP_FPS),
        (
            int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    score_us = 0
    score_them = 0
    for point_i, point in enumerate(game.points):
        if point.score == 1:
            score_us += 1
        elif point.score == -1:
            score_them += 1
        else:
            print("Warning: Point has invalid score", point)

        in_video.set(cv2.CAP_PROP_POS_FRAMES, point.start)
        for t in range(point.end - point.start):
            ret, frame = in_video.read()
            if not ret:
                print("Warning: Reached end of video early on point", point)
                break

            # Annotate frame
            if point_i == 0 and t < TITLE_TIME:
                put_text(frame, game.title, (50, 100))

            if t < LINE_TIME:
                put_text(frame, point.line, (50, frame.shape[0] - 100))

            if t >= (point.end - point.start) - RESULT_TIME:
                put_text(frame, f"Score: {score_us} - {score_them}", (50, frame.shape[0] - 150))
                put_text(frame, point.result, (50, frame.shape[0] - 100))

            out_video.write(frame)

    if point_i + 1 < len(game.points):
        print("Warning: Did not reach end of all points. Stopped at point index", point_i)

    in_video.release()
    out_video.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("points")
    args = parser.parse_args()

    game = parse_file(args.points)
    check_no_overlap(game)

    crop(args, game)


if __name__ == "__main__":
    main()
