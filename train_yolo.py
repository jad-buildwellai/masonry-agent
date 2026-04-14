#!/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
"""
YOLO26-l training on masonry crack/brick dataset
Apple M3 Max — MPS (Metal) backend via PyTorch, MLX for class-weight analysis
Dataset: 7794 train / 600 val / 600 test — 3 classes: brick, broken_brick, crack
YOLO26-l: 26.3M params, 392 layers, end2end (no NMS), 93.8 GFLOPs

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 train_yolo.py
"""

import time
from pathlib import Path

import mlx.core as mx
import torch
from ultralytics import YOLO


# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_YAML = BASE / "masonry.v1i.yolo26" / "data_abs.yaml"
RUNS_DIR = BASE / "runs"

# ── M3 Max device check ──────────────────────────────────────────────────────
assert torch.backends.mps.is_available(), "MPS not available — check PyTorch build"
DEVICE = "mps"
print(f"[device] PyTorch → {DEVICE}  |  MLX → {mx.default_device()}")

# ── Training config (M3 Max optimised for YOLO26-l) ─────────────────────────
TRAIN_ARGS = dict(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=16,           # YOLO26-l bigger than v8l; 16 safe on M3 Max, try 32 if no OOM
    workers=0,          # MPS forces 0 anyway (Ultralytics overrides)
    device=DEVICE,
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,           # final lr = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    cos_lr=True,
    amp=True,           # mixed precision — big speed gain on Apple Silicon
    cache=False,        # set True to cache images in RAM (needs ~15 GB for full train set)
    project=str(RUNS_DIR),
    name="masonry_yolo26l",
    save=True,
    save_period=10,     # checkpoint every N epochs
    val=True,
    plots=True,
    verbose=True,
    # Augmentation — mirror Roboflow augment config
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    shear=10.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.0,
)

# ── MLX helper: compute class weights from label files ───────────────────────
def compute_class_weights_mlx(labels_dir: Path, nc: int = 3) -> mx.array:
    """Inverse-frequency class weights via MLX on Apple GPU."""
    counts = mx.zeros([nc])
    for lf in labels_dir.glob("*.txt"):
        raw = lf.read_text().strip()
        if not raw:
            continue
        classes = mx.array([int(line.split()[0]) for line in raw.splitlines()])
        for c in range(nc):
            counts[c] += mx.sum(classes == c).item()
    total = mx.sum(counts).item()
    weights = total / (nc * counts + 1e-6)
    weights = weights / mx.sum(weights) * nc
    mx.eval(weights)
    return weights


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Masonry YOLO26-l Training — M3 Max ===\n")

    # Inspect class balance with MLX before training
    train_labels = BASE / "masonry.v1i.yolo26" / "train" / "labels"
    print("[mlx] computing class weights …")
    t0 = time.perf_counter()
    weights = compute_class_weights_mlx(train_labels, nc=3)
    elapsed = time.perf_counter() - t0
    names = ["brick", "broken_brick", "crack"]
    for i, (n, w) in enumerate(zip(names, weights.tolist())):
        print(f"  class {i} ({n:14s}): weight = {w:.4f}")
    print(f"  [mlx] done in {elapsed:.2f}s\n")

    # Load pretrained YOLO26-l and fine-tune on masonry dataset
    model = YOLO("yolo26l.pt")
    print(f"[model] YOLO26-l loaded — parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"[data ] {DATA_YAML}")
    print(f"[runs ] {RUNS_DIR}\n")

    results = model.train(**TRAIN_ARGS)

    print("\n=== Training complete ===")
    best_path = RUNS_DIR / "masonry_yolo26l" / "weights" / "best.pt"
    print(f"Best weights: {best_path}")

    # ── Validate best model on test split ───────────────────────────────────
    print("\n[val] running validation on test split …")
    val_model = YOLO(str(best_path))
    metrics = val_model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        device=DEVICE,
        split="test",
        plots=True,
        save_json=True,
    )
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    for i, name in enumerate(names):
        print(f"  {name:14s} AP50: {metrics.box.ap50[i]:.4f}")

    # ── Export to CoreML for on-device Mac/iOS inference ─────────────────────
    print("\n[export] CoreML …")
    val_model.export(format="coreml", imgsz=640)
    print("  saved: best.mlpackage")


if __name__ == "__main__":
    main()
