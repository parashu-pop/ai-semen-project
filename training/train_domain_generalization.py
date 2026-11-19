
from ultralytics import YOLO
from pathlib import Path
import shutil

# Paths
baseline = Path("runs/detect/train-9/weights/best.pt")
data_yaml = Path("dataset/visem_tracking/data.yaml")
project_dir = Path("runs/detect")
run_name = "dg_exp1"
run_dir = project_dir / run_name
weights_dir = run_dir / "weights"

if not baseline.exists():
    raise FileNotFoundError(f"Baseline not found: {baseline}")

if not data_yaml.exists():
    raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

# Preserve the original baseline
backup = Path("runs/detect/baseline_best.pt")
backup.parent.mkdir(parents=True, exist_ok=True)

if not backup.exists():
    shutil.copy2(baseline, backup)

# Resume from last.pt if a previous DG run exists.
# Otherwise, start from the original baseline.
last_checkpoint = weights_dir / "last.pt"

if last_checkpoint.exists():
    print(f"Resuming interrupted DG training: {last_checkpoint}")
    model = YOLO(str(last_checkpoint))
    resume_training = True
else:
    print(f"Starting DG training from baseline: {baseline}")
    model = YOLO(str(baseline))
    resume_training = False

# Train or resume
results = model.train(
    data=str(data_yaml),
    epochs=10,
    imgsz=256,
    batch=1,
    device=0,
    workers=0,

    project=str(project_dir),
    name=run_name,
    exist_ok=True,
    resume=resume_training,

    # Domain-randomization augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.6,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=3,

    # Save checkpoints and plots
    save=True,
    save_period=1,
    plots=True,
    val=True,
    seed=42
)

print("Training completed.")
print(f"Best checkpoint: {weights_dir / 'best.pt'}")
print(f"Latest checkpoint: {weights_dir / 'last.pt'}")
