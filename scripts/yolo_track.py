from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = "../data/videos/BoomNov10_part.mp4"
results = model.track(source=video, show=True)
