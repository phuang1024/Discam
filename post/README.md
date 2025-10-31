# Synopsis

Programmatic utilities for post processing film.

Capabilities and pipeline:
1. Video crop and stabilization with FFmpeg.
2. Add annotations to video and improve color.
   Also generates YT chapter markings of points.

# Points file format

The points file specifies the timestamps of the start and end of each point of the game,
as well as annotations to add to the video.

Use a video viewer with frame numbers to help.

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
  Write the text `\n` to indicate a newline in the title value only.
- Keys:
    - `title`: Title annotated at beginning.
    - `point`: Signifies a new point.
      Value should be the start and end timestamps of the point in `H:M:S` format.
    - `line`: Annotation put at beginning of point.
    - `score`: Numerical score as `1`, `-1`. `+1` is OK too.
      `1` means "we" score a point, `-1` means "they" score a point.
      Score is annotated throughout the video.
    - `result`: Annotation put at end of point.

Example points text file:

```
title No Wisco 2025\nSaturday Game 1

point 10.5 50
line Player1 Player2 ...
score 1
result Clean hold

point 67.8 135.6
line Player1 Player2 ...
score 1
result Break
```

# Crop and stabilization

```bash
python crop_stab.py input.mp4 crop_stab.mp4 points.txt
# Output is saved as crop_stab.mp4
```

- Uses FFmpeg video crop and stabilization. See script for hard coded parameters.
- Generates `cropped.input.mp4` as an intermediate file.
- Generates `transforms.trf` as an intermediate file.
- Converts video to 1080p24. Compresses video.

# Annotations and color correction

Add text annotations to the video.
Assumes video has been cropped.

```bash
python color_text.py crop_stab.mp4 final.mp4 points.txt
# Output is saved as final.mp4
# Also saves final.mp4_chapters.txt which has YT chapter markings.
```
