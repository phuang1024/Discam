#!/bin/bash

# Run both post processing scripts in sequence.
# Usage: ./post.sh input.mp4 output.mp4 points.txt
# Saves intermediate file crop_stab.mp4

set -e

python crop_stab.py "$1" crop_stab.mp4 "$3"
python color_text.py crop_stab.mp4 "$2" "$3"
