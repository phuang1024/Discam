# Timestamp-er

The goal of this module is to automatically create timestamps of points, for ease of viewing.

I.e. detect when each point starts.

The NN classifies whether each moment (a short segment of video) is in play.
Querying the NN on clips of the entire game allows us to determine overall sections in play.
Apply some temporal filter for stability, and find the beginning of each section for the timestamps.

## NN format

Input:

- Video clip (3D tensor).
- Sampled at some low FPS.
- A few seconds in length.

Output:

- Binary clasification of whether this clip is game in play.

## Training and usage

Generate data: Requires a video and list of timestamps.

Timestamps file: Each line is two timestamps, which means that a point is occuring between them.
Use `hh:mm:ss` or `mm:ss` or `s` format.

```
hh:mm:ss hh:mm:ss
...
```

```bash
python make_data.py video.mp4 timestamps.txt data_dir/
```

Train:

```bash
python train.py data_dir/ output_dir/
```
