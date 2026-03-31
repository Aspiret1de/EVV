from pathlib import Path
import os
import random
import shutil
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = PROJECT_ROOT / "data" / "kitti_raw" / "training"
IMG_DIR = RAW_ROOT / "image_2"
LBL_DIR = RAW_ROOT / "label_2"

# 输出 YOLO 数据目录
YOLO_ROOT = PROJECT_ROOT / "data" / "kitti_yolo"
TRAIN_IMG_DIR = YOLO_ROOT / "images" / "train"
VAL_IMG_DIR = YOLO_ROOT / "images" / "val"
TRAIN_LBL_DIR = YOLO_ROOT / "labels" / "train"
VAL_LBL_DIR = YOLO_ROOT / "labels" / "val"

CLASS_MAP = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2
}

VAL_RATIO = 0.2
RANDOM_SEED = 42

print("PROJECT_ROOT =", PROJECT_ROOT)
print("IMG_DIR =", IMG_DIR)
print("LBL_DIR =", LBL_DIR)
print("IMG exists:", IMG_DIR.exists())
print("LBL exists:", LBL_DIR.exists())
print("PNG count:", len(list(IMG_DIR.glob("*.png"))))
print("TXT count:", len(list(LBL_DIR.glob("*.txt"))))

def make_dirs():
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def convert_bbox(size, box):
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return x_center, y_center, width, height


def parse_kitti_label(label_path, img_size):
    yolo_lines = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            cls_name = parts[0]

            if cls_name not in CLASS_MAP:
                continue

            xmin = float(parts[4])
            ymin = float(parts[5])
            xmax = float(parts[6])
            ymax = float(parts[7])

            if xmax <= xmin or ymax <= ymin:
                continue

            x_center, y_center, width, height = convert_bbox(
                img_size, (xmin, ymin, xmax, ymax)
            )

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)

            class_id = CLASS_MAP[cls_name]
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

    return yolo_lines


def main():
    make_dirs()

    image_files = sorted([p for p in IMG_DIR.glob("*.png")])
    stems = [p.stem for p in image_files]

    random.seed(RANDOM_SEED)
    random.shuffle(stems)

    val_count = int(len(stems) * VAL_RATIO)
    val_stems = set(stems[:val_count])
    train_stems = set(stems[val_count:])

    print(f"Total images: {len(stems)}")
    print(f"Train: {len(train_stems)}")
    print(f"Val: {len(val_stems)}")

    for stem in stems:
        img_path = IMG_DIR / f"{stem}.png"
        lbl_path = LBL_DIR / f"{stem}.txt"

        if not img_path.exists() or not lbl_path.exists():
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = parse_kitti_label(lbl_path, (img_w, img_h))

        if stem in train_stems:
            out_img = TRAIN_IMG_DIR / img_path.name
            out_lbl = TRAIN_LBL_DIR / f"{stem}.txt"
        else:
            out_img = VAL_IMG_DIR / img_path.name
            out_lbl = VAL_LBL_DIR / f"{stem}.txt"

        shutil.copy2(img_path, out_img)

        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    print("Conversion finished.")


if __name__ == "__main__":
    main()