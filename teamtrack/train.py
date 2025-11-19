from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, name="teamtrack-yolo11n")
