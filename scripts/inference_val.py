import random
from pathlib import Path
from ultralytics import YOLO

def run_selective_inference(num_samples=10):
    # 1. Setup paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_PATH = PROJECT_ROOT / "runs" / "yolo_kitti_baseline" / "weights" / "best.pt"
    VAL_IMG_DIR = PROJECT_ROOT / "data" / "kitti_yolo" / "images" / "val"
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Did you finish training?")

    # 2. Get all image paths and pick 10 random ones
    all_images = list(VAL_IMG_DIR.glob("*.png"))
    if len(all_images) < num_samples:
        num_samples = len(all_images)
    
    sampled_images = random.sample(all_images, num_samples)
    # Convert Path objects to strings for the model
    image_list = [str(img) for img in sampled_images]

    # 3. Load the model
    model = YOLO(MODEL_PATH)

    # 4. Perform Inference on the sample
    print(f"Running inference on {num_samples} random images...")
    results = model.predict(
        source=image_list,
        conf=0.25,      # Adjust this if you see too many or too few boxes
        save=True,      # This saves the visual overlays
        save_txt=True,  # This saves the labels for your table/analysis
        project=str(PROJECT_ROOT / "runs" / "inference"),
        name="kitti_random_samples",
        exist_ok=True
    )

    print(f"Done! Check your results here: runs/inference/kitti_random_samples")

if __name__ == "__main__":
    run_selective_inference(10)