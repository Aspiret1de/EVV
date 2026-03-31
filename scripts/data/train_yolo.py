from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
data_yaml = PROJECT_ROOT / "data" / "kitti_yolo" / "kitti.yaml"

print("PROJECT_ROOT =", PROJECT_ROOT)
print("data_yaml =", data_yaml)
print("yaml exists:", data_yaml.exists())

model = YOLO("yolov8n.pt")

model.train(
    data=str(data_yaml),
    epochs=1,
    imgsz=640,
    batch=4,
    device="cpu"
)