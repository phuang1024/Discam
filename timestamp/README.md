# Timestamp

Goal: Automatically generate timestamps of the start of each point
given unedited film of a game.

## NN format

- Input: A video clip. Length can be quite long (TODO).
- Output: Probability per frame of a point starting at that frame.

First, each frame is encoded with a fine tuned ResNet.
Then, the embedding sequence is fed into a 1D CNN which outputs probability.

## Training and usage

### Data

Requires videos and list of timestamps.

Video should be unedited film of a game.

Videos should be:

- 480x270.
- 3fps.

Each video has an associated timestamps file, which is manually labeled data
of the start ane end of each point.

Timestamps file: Each line is two timestamps, the start and end of a point.
Use `hh:mm:ss` or `mm:ss` or `s` format.

```
hh:mm:ss hh:mm:ss
hh:mm:ss hh:mm:ss
...
```

### Training
