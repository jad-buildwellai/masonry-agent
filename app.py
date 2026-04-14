#!/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
"""
Brick Counter — Streamlit UI
Upload images → count bricks → progress report
Based on: Magdy et al. (2025), Buildings 15(14):2456
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE     = Path(__file__).parent
RUNS_DIR = BASE / "runs"

CONF_THRESHOLD   = 0.60
DISCREPANCY_WARN = 0.05

COLOURS = {
    0: (0,   200,  50),   # brick        — green
    1: (0,   140, 255),   # broken_brick — orange
    2: (0,   0,   220),   # crack        — red
}
CLASS_NAMES = {0: "brick", 1: "broken_brick", 2: "crack"}


@st.cache_resource
def load_model():
    primary = BASE / "masonry.pt"
    if primary.exists():
        return YOLO(str(primary)), str(primary)
    candidates = sorted(RUNS_DIR.glob("*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return YOLO(str(candidates[0])), str(candidates[0])
    fallback = BASE / "yolov8l.pt"
    if fallback.exists():
        return YOLO(str(fallback)), str(fallback) + " (pretrained fallback)"
    st.error("No model found. Place masonry.pt in the project folder.")
    st.stop()


def detect(model, img_bgr, conf_thresh):
    results   = model(img_bgr, conf=0.01, verbose=False)
    counts    = {0: 0, 1: 0, 2: 0}
    flagged   = 0
    annotated = img_bgr.copy()

    for box in results[0].boxes:
        conf   = float(box.conf)
        cls_id = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        colour = COLOURS.get(cls_id, (200, 200, 200))
        label  = CLASS_NAMES.get(cls_id, str(cls_id))

        if conf < conf_thresh:
            flagged += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(annotated, f"? {label} {conf:.2f}", (x1, max(0, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        else:
            counts[cls_id] += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

    return annotated, counts, flagged


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brick Counter",
    page_icon="🧱",
    layout="wide",
)

st.title("🧱 Brick Counter")
st.caption("Real-time bricklaying progress monitor · Magdy et al. (2025)")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    conf_thresh = st.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.95, value=CONF_THRESHOLD, step=0.05,
        help="Detections below this are flagged for manual review (paper §4.3: 0.60)"
    )
    planned = st.number_input(
        "As-planned brick count (optional)",
        min_value=0, value=0, step=10,
        help="Set > 0 to calculate progress % against as-planned BIM quantity"
    )
    st.divider()
    model, model_path = load_model()
    st.success(f"Model loaded")
    st.code(Path(model_path).name, language=None)

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload one or more brick wall images",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Upload images above to begin counting.")
    st.stop()

# ── Process images ────────────────────────────────────────────────────────────
total_brick  = 0
total_broken = 0
total_crack  = 0
total_flag   = 0

rows = []

for f in uploaded:
    pil_img  = Image.open(f).convert("RGB")
    img_bgr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ann_bgr, counts, flagged = detect(model, img_bgr, conf_thresh)
    ann_rgb  = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

    n_brick   = counts[0]
    n_broken  = counts[1]
    n_crack   = counts[2]
    progress  = n_brick + n_broken

    total_brick  += n_brick
    total_broken += n_broken
    total_crack  += n_crack
    total_flag   += flagged

    rows.append({
        "file":     f.name,
        "brick":    n_brick,
        "broken":   n_broken,
        "crack":    n_crack,
        "progress": progress,
        "flagged":  flagged,
        "ann_img":  ann_rgb,
    })

# ── Per-image results ─────────────────────────────────────────────────────────
st.subheader(f"Results — {len(rows)} image{'s' if len(rows) > 1 else ''}")

for row in rows:
    with st.expander(f"📷 {row['file']}  —  {row['progress']} bricks detected", expanded=len(rows) == 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(row["ann_img"], caption="Annotated", use_container_width=True)
        with col2:
            st.metric("🟩 Brick",         row["brick"])
            st.metric("🟠 Broken brick",  row["broken"])
            st.metric("🔴 Crack",         row["crack"])
            st.metric("⚫ Flagged (<conf)", row["flagged"])

# ── Summary ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary")

total_progress = total_brick + total_broken

c1, c2, c3, c4 = st.columns(4)
c1.metric("As-built bricks", total_progress, help="brick + broken_brick")
c2.metric("Cracks (defect)", total_crack)
c3.metric("Flagged for review", total_flag, help=f"conf < {conf_thresh:.2f}")
c4.metric("Images processed", len(rows))

if planned and planned > 0:
    pct  = total_progress / planned * 100
    disc = abs(total_progress - planned) / planned

    st.divider()
    st.subheader("Progress vs As-Planned")

    prog_col, warn_col = st.columns([2, 1])
    with prog_col:
        st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% complete  ({total_progress} / {planned})")
    with warn_col:
        if disc > DISCREPANCY_WARN:
            st.warning(f"⚠ Discrepancy {disc*100:.1f}% > 5%  — manual review required (paper §4.3)")
        else:
            st.success(f"✅ Within tolerance ({disc*100:.1f}% discrepancy)")

# ── Legend ─────────────────────────────────────────────────────────────────────
with st.expander("Legend"):
    st.markdown("""
| Colour | Class | Role |
|--------|-------|------|
| 🟩 Green | `brick` | Progress unit |
| 🟠 Orange | `broken_brick` | Progress unit |
| 🔴 Red | `crack` | Defect flag |
| ⚫ Grey | any (low conf) | Flagged for manual review |

Confidence threshold default: **0.60** (Magdy et al., 2025 §4.3)
Progress % = `(brick + broken_brick) / as-planned × 100`
Alert fires when discrepancy > **5%**
""")
