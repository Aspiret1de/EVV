from pathlib import Path
from ultralytics import YOLO


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    data_yaml = PROJECT_ROOT / "data" / "kitti_yolo" / "kitti.yaml"

    runs_dir = PROJECT_ROOT / "runs"

    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("data_yaml =", data_yaml)
    print("yaml exists:", data_yaml.exists())
    print("runs_dir =", runs_dir)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(runs_dir),
        name="yolo_kitti_baseline",
        exist_ok=True,
        pretrained=True,
        verbose=True
    )


if __name__ == "__main__":
    main()