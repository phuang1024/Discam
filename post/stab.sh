#!/bin/bash

ffmpeg -i "$1" -vf vidstabdetect=shakiness=6 -f null -

ffmpeg -i "$1" -vf vidstabtransform=smoothing=30:zoom=10:input="transforms.trf" \
    -c:v libx264 -c:a mp3 -s 1920x1080 -r 24 -preset slow -crf 27 "stab_$1"
