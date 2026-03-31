from pathlib import Path
import random
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "kitti_yolo" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "data" / "kitti_yolo" / "labels" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kitti_yolo" / "vis_samples"

# 类别名称
CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist"
}

# 想随机可视化多少张
NUM_SAMPLES = 5

# 是否保存图片
SAVE_IMAGES = True


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h

            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)

            # 边界裁剪
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w - 1, xmax)
            ymax = min(img_h - 1, ymax)

            boxes.append((class_id, xmin, ymin, xmax, ymax))

    return boxes


def draw_boxes(image, boxes):
    for class_id, xmin, ymin, xmax, ymax in boxes:
        class_name = CLASS_NAMES.get(class_id, str(class_id))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            class_name,
            (xmin, max(ymin - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.png"))

    if not image_files:
        print("No images found in:", IMAGE_DIR)
        return

    sample_files = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

    print("IMAGE_DIR =", IMAGE_DIR)
    print("LABEL_DIR =", LABEL_DIR)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print(f"Visualizing {len(sample_files)} sample images...")

    for img_path in sample_files:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue

        img_h, img_w = image.shape[:2]
        boxes = load_yolo_labels(label_path, img_w, img_h)
        vis_image = draw_boxes(image.copy(), boxes)

        print(f"{img_path.name}: {len(boxes)} boxes")

        if SAVE_IMAGES:
            out_path = OUTPUT_DIR / f"{img_path.stem}_vis.png"
            cv2.imwrite(str(out_path), vis_image)

    print("Visualization finished.")
    print("Check saved images in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()