# Synopsis

Programmatic utilities for post processing film.

Capabilities and pipeline:
1. Video stabilization with FFmpeg.
2. Video crop to game points (i.e. remove footage in between points).
   Also add annotations.
3. Color improvement.

Other features:
- Generate YT chapter markings of points.
- Text annotate on video.


# Detailed pipeline description

All three steps take input and output videos.

It is recommended to run the three steps in the given sequence.

## Stabilization

Usage:

```bash
./stab.sh input.mp4 stab.mp4
# Output is saved as stab.mp4
```

- Uses FFmpeg video stabilization. See script for hard coded parameters.
- Generates `transforms.trf` as an intermediate file.
- Converts video to 1080p24. Compresses video.

## Crop

Temporal crop to points of the game.

### File format

First, write a points file, e.g. `points.txt`.
Use a video viewer with frame numbers like Blender to help.
It is recommended to leave a bit of time (e.g. 2s, or when the pull starts) at the crop of each point.
This accounts for time taken by annotations.

Points text file format:

```
title <title>

point <frame_start> <frame_end>
line <line>
score <score>
result <result>

point <frame_start> <frame_end>
line <line>
score <score>
result <result>

...
```

- Empty lines do not matter.
- Each line is a single word key (e.g. `title`), then any number of spaces, then the value.
  Write the text `\n` to indicate a newline in the value.
- Keys:
    - `title`: Title annotated at beginning.
    - `point`: Signifies a new point. Value should be two integers space separated,
      indicating the start and end frame of the point.
    - `line`: Annotation put at beginning of point.
    - `score`: Numerical score as `1`, `-1`. `+1` is OK too.
      `1` means "we" score a point, `-1` means "they" score a point.
      Score is annotated throughout the video.
    - `result`: Annotation put at end of point.

Example points text file:

```
title No Wisco 2025\nSaturday Game 1

point 240 720
line Player1 Player2 ...
score 1
result Clean hold

point 900 1500
line Player1 Player2 ...
score 1
result Break
```

### Usage

```bash
python crop.py stab.mp4 cropped.mp4 points.txt
# Output is saved as cropped.mp4
```

## Color adjustment

### Features

TODO

### Usage

```bash
python color.py cropped.mp4 final.mp4
# Output is saved as final.mp4
```
