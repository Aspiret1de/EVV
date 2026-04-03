from ultralytics import YOLO
from pathlib import Path

def evaluate_model():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_PATH = PROJECT_ROOT / "runs" / "yolo_kitti_baseline" / "weights" / "best.pt"
    DATA_YAML = PROJECT_ROOT / "data" / "kitti_yolo" / "kitti.yaml"

    model = YOLO(MODEL_PATH)

    # Run validation mode
    # iou=0.7 is the standard KITTI threshold for 'Car'
    # iou=0.5 is the standard for 'Pedestrian' and 'Cyclist'
    metrics = model.val(
        data=str(DATA_YAML),
        split='val',
        iou=0.6, # A middle-ground threshold for evaluation
        project=str(PROJECT_ROOT / "runs" / "evaluation"),
        name="kitti_metrics"
    )

    print("--- Evaluation Results ---")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    
    # Class-specific Average Precision
    for i, name in enumerate(metrics.names.values()):
        ap50 = metrics.box.class_result(i)[2]
        print(f"Class {name} AP@50: {ap50:.4f}")

if __name__ == "__main__":
    evaluate_model()