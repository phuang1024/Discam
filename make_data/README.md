# Make data

This directory contains scripts to process and create training data.

## Usage

First, run optical flow to find salient points in each frame.

This saves `output/points.npy`.

```bash
python optical_flow.py video.mp4 output/

# Optional
python vis_optical_flow.py video.mp4 output/
```
