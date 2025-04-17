from ultralytics import YOLO

# Load the lightweight YOLOv8 Nano model
model = YOLO("yolov8n.pt")  # use 'yolov8s.pt' later for more accuracy

# Train the model (FAST MODE 💨)
model.train(
    data="data.yaml",    # path to your dataset
    epochs=10,           # lower = faster, but less accurate
    imgsz=416,           # smaller images = faster training
    batch=4,             # reduce batch size for CPU-friendliness
    workers=0,           # prevent crashing on Mac
    device="cpu",        # use CPU explicitly
    verbose=True         # show all logs
)
