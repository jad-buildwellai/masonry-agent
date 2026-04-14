#!/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
"""
Brick counter — real-time bricklaying progress monitoring
Based on: Magdy et al. (2025) "Real-Time Progress Monitoring of Bricklaying",
          Buildings 15(14):2456, doi:10.3390/buildings15142456

Usage:
  # Count bricks in one image
  python3.11 count_bricks.py image.jpg

  # Count bricks in a directory
  python3.11 count_bricks.py /path/to/images/

  # Compare against as-planned quantity (progress %)
  python3.11 count_bricks.py /path/to/images/ --planned 200

  # Use specific trained model
  python3.11 count_bricks.py /path/to/images/ --model runs/masonry_yolo26l/weights/best.pt

  # Save annotated images to a folder
  python3.11 count_bricks.py /path/to/images/ --output-dir output/
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO
import cv2

# ── Config (mirrors paper: Magdy et al., 2025) ───────────────────────────────
CONF_THRESHOLD   = 0.60   # §4.3 — detections below this flagged for review
DISCREPANCY_WARN = 0.05   # §4.3 — alert if |as-built − as-planned| / planned > 5%
BRICK_CLASSES    = {0: "brick", 1: "broken_brick"}   # count as bricklaying progress
DEFECT_CLASSES   = {2: "crack"}                       # quality flag, not progress
IMG_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASE     = Path(__file__).parent
RUNS_DIR = BASE / "runs"

# Colour palette for bounding boxes (BGR)
COLOURS = {
    0: (0,   200,  50),   # brick        — green
    1: (0,   140, 255),   # broken_brick — orange
    2: (0,   0,   220),   # crack        — red
}


def find_best_model() -> Path:
    """Return masonry.pt, else latest best.pt, else yolov8l.pt fallback."""
    primary = BASE / "masonry.pt"
    if primary.exists():
        return primary
    candidates = sorted(RUNS_DIR.glob("*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    fallback = BASE / "yolov8l.pt"
    if fallback.exists():
        print(f"[model] masonry.pt not found — using pretrained {fallback.name}")
        return fallback
    sys.exit("[model] ERROR: masonry.pt not found. Train the model first.")


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTENSIONS else []
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def draw_detections(img, results, conf_thresh: float) -> tuple:
    """Draw boxes on img. Returns (annotated_img, counts_dict)."""
    counts = {c: 0 for c in list(BRICK_CLASSES) + list(DEFECT_CLASSES)}
    low_conf_count = 0

    for box in results[0].boxes:
        conf   = float(box.conf)
        cls_id = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label  = results[0].names[cls_id]
        colour = COLOURS.get(cls_id, (200, 200, 200))

        if conf < conf_thresh:
            # Below threshold — draw dashed grey box (flag for review per paper §4.3)
            low_conf_count += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(img, f"? {label} {conf:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
            continue

        counts[cls_id] = counts.get(cls_id, 0) + 1
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

    return img, counts, low_conf_count


def progress_report(as_built: int, as_planned: int | None) -> str:
    if as_planned is None:
        return ""
    pct = (as_built / as_planned * 100) if as_planned > 0 else 0.0
    disc = abs(as_built - as_planned) / as_planned if as_planned > 0 else 0.0
    warn = " ⚠  DISCREPANCY > 5% — manual review required" if disc > DISCREPANCY_WARN else ""
    return f"  Progress  : {as_built}/{as_planned} bricks ({pct:.1f}%){warn}"


def run(args, conf_threshold: float = CONF_THRESHOLD):
    source   = Path(args.source)
    model_pt = Path(args.model) if args.model else find_best_model()
    out_dir  = Path(args.output_dir) if args.output_dir else None

    print(f"\n=== Brick Counter — masonry progress monitor ===")
    print(f"  Model     : {model_pt}")
    print(f"  Source    : {source}")
    print(f"  Conf thr  : {conf_threshold}  (paper: Magdy et al., 2025 §4.3)")
    if args.planned:
        print(f"  As-planned: {args.planned} bricks")
    print()

    model  = YOLO(str(model_pt))
    images = collect_images(source)

    if not images:
        sys.exit(f"No images found at: {source}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_bricks  = 0   # brick + broken_brick across all images
    total_cracks  = 0
    total_flagged = 0   # below-threshold, flagged for review

    header = f"{'Image':<45} {'brick':>6} {'broken':>7} {'crack':>6} {'flagged':>8}"
    print(header)
    print("-" * len(header))

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] Cannot read: {img_path.name}")
            continue

        results = model(img, conf=0.01, verbose=False)  # get all, filter manually
        img_ann, counts, flagged = draw_detections(img.copy(), results, conf_threshold)

        n_brick   = counts.get(0, 0)
        n_broken  = counts.get(1, 0)
        n_crack   = counts.get(2, 0)
        n_progress = n_brick + n_broken   # both count as bricklaying progress

        total_bricks  += n_progress
        total_cracks  += n_crack
        total_flagged += flagged

        # Overlay summary on annotated image
        summary_txt = [
            f"brick: {n_brick}  broken: {n_broken}  crack: {n_crack}",
            f"progress units: {n_progress}  (flagged: {flagged})",
        ]
        for i, txt in enumerate(summary_txt):
            cv2.putText(img_ann, txt, (10, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3)
            cv2.putText(img_ann, txt, (10, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

        if out_dir:
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), img_ann)

        name_short = img_path.name[:44]
        print(f"  {name_short:<44} {n_brick:>6} {n_broken:>7} {n_crack:>6} {flagged:>8}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("-" * len(header))
    print(f"  {'TOTAL':<44} {total_bricks:>6}  {'':>5} {total_cracks:>6} {total_flagged:>8}")
    print()
    print(f"  As-built bricks (brick + broken_brick) : {total_bricks}")
    print(f"  Cracks detected (defect flag)          : {total_cracks}")
    print(f"  Low-confidence detections (for review) : {total_flagged}")
    prog = progress_report(total_bricks, args.planned)
    if prog:
        print(prog)
    if out_dir:
        print(f"\n  Annotated images saved → {out_dir}/")
    print()


def main():
    parser = argparse.ArgumentParser(description="Brick counter — bricklaying progress monitor")
    parser.add_argument("source",
                        help="Image file or directory of images")
    parser.add_argument("--model",  default=None,
                        help="Path to YOLO .pt weights (default: latest best.pt or yolov8l.pt)")
    parser.add_argument("--planned", type=int, default=None,
                        help="As-planned brick count for progress %% calculation")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save annotated images")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"Confidence threshold (default {CONF_THRESHOLD})")
    args = parser.parse_args()

    run(args, conf_threshold=args.conf)


if __name__ == "__main__":
    main()
