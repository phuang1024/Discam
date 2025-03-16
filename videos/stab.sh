ffmpeg -i upres.mp4 -vf vidstabdetect=shakiness=7 -f null -
ffmpeg -i upres.mp4 -vf vidstabtransform=smoothing=30:zoom=5:input="transforms.trf" stab.mp4
