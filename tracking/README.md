# Introduction

This module implements person tracking, and person classification.

Tracking is done with YOLO detection and tracking.

Classification is a 1D convolutional NN, which classifies each trajectory as an
active or inactive player.

## Components

The inference pipeline consists of:

- `track.py`: YOLO detection and tracking.
- `model.py`: Classifier model.

The model training pipeline consists of:

- `make_data.py`: Script to generate and label data.
- `train.py`: Training script.
- `test.py`: Test and visualize results.

## Data

Data generation is a three step process: Tracking, labeling, distilling. See
`make_data.py`.

Tracks are saved as NN input format.

Tracking and labeled data file structure:

```
data/
|__ 0.track.json  # Trajectory, raw format.
|__ 0.meta.json   # Metadata.
|__ 0.label.txt   # Generated during labeling.
|   ...
```

Distilled data file structure:

```
data/
|__ 0.track.pt   # Trajectory, max length NN input size, NN input format.
|__ 0.label.txt  # Label.
|   ...
```

Usage:

```bash
python make_data.py track --data ... --video ...
python make_data.py label --data ... --video ...
python make_data.py distill --data ... --output ...
```
