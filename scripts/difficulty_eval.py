import shutil
import tempfile
from pathlib import Path
from ultralytics import YOLO


def filter_kitti_by_difficulty(raw_lbl_dir, yolo_lbl_dir, output_root, level):
    output_dir = Path(output_root) / f"labels_{level}"
    output_dir.mkdir(parents=True, exist_ok=True)

    criteria = {
        "easy": {"h": 40, "occ": 0, "trunc": 0.15},
        "moderate": {"h": 25, "occ": 1, "trunc": 0.30},
        "hard": {"h": 25, "occ": 2, "trunc": 0.50},
    }
    c = criteria[level]

    for lbl_file in Path(yolo_lbl_dir).glob("*.txt"):
        raw_lbl_path = Path(raw_lbl_dir) / lbl_file.name
        if not raw_lbl_path.exists():
            continue

        valid_lines = []

        with open(raw_lbl_path, "r") as f_raw, open(lbl_file, "r") as f_yolo:
            raw_lines = [
                line for line in f_raw.readlines()
                if not line.startswith("DontCare")
            ]
            yolo_lines = f_yolo.readlines()

            if len(raw_lines) != len(yolo_lines):
                print(
                    f"Warning: Skipping {lbl_file.name} (mismatch: raw={len(raw_lines)}, yolo={len(yolo_lines)})"
                )
                continue

            for r_line, y_line in zip(raw_lines, yolo_lines):
                parts = r_line.split()

                if len(parts) < 8:
                    continue  # malformed line

                try:
                    trunc = float(parts[1])
                    occ = int(parts[2])
                    h = float(parts[7]) - float(parts[5])
                except ValueError:
                    continue  # bad numeric conversion

                if h >= c["h"] and occ <= c["occ"] and trunc <= c["trunc"]:
                    valid_lines.append(y_line)

        # Always write the file, even if it's empty, to prevent YOLO "missing label" warnings
        with open(output_dir / lbl_file.name, "w") as f_out:
            if valid_lines:
                f_out.writelines(valid_lines)

    return output_dir


def clear_yolo_cache(label_dir):
    for cache_file in Path(label_dir).parent.glob("*.cache"):
        try:
            cache_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete cache {cache_file}: {e}")


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    MODEL_PATH = PROJECT_ROOT / "runs/yolo_kitti_baseline/weights/best.pt"
    RAW_LBL_DIR = PROJECT_ROOT / "data/kitti_raw/training/label_2"
    VAL_YOLO_LBL = PROJECT_ROOT / "data/kitti_yolo/labels/val"
    DATA_YAML = PROJECT_ROOT / "data/kitti_yolo/kitti.yaml"

    model = YOLO(MODEL_PATH)
    results_summary = {}

    backup_dir = VAL_YOLO_LBL.parent / "val_backup"

    for level in ["easy", "moderate", "hard"]:
        print(f"\n--- Evaluating {level.upper()} Difficulty ---")

        temp_root = Path(tempfile.mkdtemp())
        temp_label_dir = filter_kitti_by_difficulty(
            RAW_LBL_DIR, VAL_YOLO_LBL, temp_root, level
        )

        try:
            # Ensure clean backup location
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            # Move original labels → backup
            if VAL_YOLO_LBL.exists():
                shutil.move(str(VAL_YOLO_LBL), str(backup_dir))

            # Move filtered labels → validation dir
            shutil.move(str(temp_label_dir), str(VAL_YOLO_LBL))

            # Clear YOLO cache
            clear_yolo_cache(VAL_YOLO_LBL)

            # Run validation
            metrics = model.val(
                data=str(DATA_YAML),
                split="val",
                plots=False,
                verbose=False,
            )

            results_summary[level] = metrics.box.map50

        finally:
            # Restore original labels safely
            try:
                if VAL_YOLO_LBL.exists():
                    shutil.rmtree(VAL_YOLO_LBL)

                if backup_dir.exists():
                    shutil.move(str(backup_dir), str(VAL_YOLO_LBL))
            except Exception as e:
                print(f"ERROR restoring labels: {e}")

            # Cleanup temp directory
            shutil.rmtree(temp_root, ignore_errors=True)

    print("\n" + "=" * 30)
    print("KITTI DIFFICULTY STATISTICS (mAP@50)")
    print("=" * 30)
    for lvl, score in results_summary.items():
        print(f"{lvl.capitalize()}: {score:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()