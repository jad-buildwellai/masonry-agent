"""
Masonry Brick Counter — FastAPI Pod Server
Bulk image upload → YOLO detection → brick counts + progress report
Based on: Magdy et al. (2025), Buildings 15(14):2456

Run:
  pip install fastapi uvicorn ultralytics opencv-python-headless
  python pod_server.py

Endpoint:
  POST /count
  Content-Type: application/json

Input:
  {
    "images": [
      {"name": "wall1.jpg", "data": "<base64-encoded-image>"},
      ...
    ],
    "planned": 500,          // optional
    "conf_threshold": 0.6    // optional, default 0.6
  }

Output: same schema as runpod_handler.py
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
VOLUME_DIR = Path("/runpod-volume")
MODEL_DIR  = VOLUME_DIR / "models"

CONF_DEFAULT     = 0.60
DISCREPANCY_WARN = 0.05
COLOURS = {
    0: (0,   200,  50),   # brick        green
    1: (0,   140, 255),   # broken_brick orange
    2: (0,   0,   220),   # crack        red
}
CLASS_NAMES = {0: "brick", 1: "broken_brick", 2: "crack"}

app = FastAPI(title="Masonry Brick Counter", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model() -> YOLO:
    global _model
    if _model is not None:
        return _model

    # Priority: volume cache → baked image → local
    candidates = [
        MODEL_DIR / "masonry.pt",
        Path("/masonry.pt"),
        BASE / "masonry.pt",
    ]
    for p in candidates:
        if p.exists():
            logger.info(f"Loading model from {p}")
            _model = YOLO(str(p))
            # Cache to volume for faster future loads
            if p != MODEL_DIR / "masonry.pt":
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(p, MODEL_DIR / "masonry.pt")
                logger.info("Cached model to volume")
            return _model

    raise RuntimeError("masonry.pt not found — upload to /runpod-volume/models/masonry.pt")


# ── Request / Response models ─────────────────────────────────────────────────
class ImageItem(BaseModel):
    name: str
    data: str  # base64-encoded image bytes


class CountRequest(BaseModel):
    images: list[ImageItem]
    planned: Optional[int] = None
    conf_threshold: float = CONF_DEFAULT


# ── Detection logic ───────────────────────────────────────────────────────────
def detect_image(model: YOLO, img_bgr, conf_thresh: float) -> dict:
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
            cv2.putText(annotated, f"?{label} {conf:.2f}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        else:
            counts[cls_id] += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

    return {
        "brick":         counts[0],
        "broken_brick":  counts[1],
        "crack":         counts[2],
        "flagged":       flagged,
        "annotated_bgr": annotated,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        model = get_model()
        return {"status": "ok", "model_loaded": model is not None}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/count")
def count_bricks(req: CountRequest):
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided")

    results       = []
    total_brick   = 0
    total_broken  = 0
    total_crack   = 0
    total_flagged = 0

    for img_item in req.images:
        name = img_item.name
        try:
            img_bytes = base64.b64decode(img_item.data)
            arr       = np.frombuffer(img_bytes, np.uint8)
            img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                results.append({"name": name, "error": "Could not decode image"})
                continue

            det = detect_image(model, img_bgr, req.conf_threshold)

            _, buf = cv2.imencode(".jpg", det["annotated_bgr"],
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
            ann_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            n_brick   = det["brick"]
            n_broken  = det["broken_brick"]
            n_crack   = det["crack"]
            n_flagged = det["flagged"]
            progress  = n_brick + n_broken

            total_brick   += n_brick
            total_broken  += n_broken
            total_crack   += n_crack
            total_flagged += n_flagged

            results.append({
                "name":            name,
                "brick":           n_brick,
                "broken_brick":    n_broken,
                "crack":           n_crack,
                "flagged":         n_flagged,
                "progress_units":  progress,
                "annotated_image": ann_b64,
            })

        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            results.append({"name": name, "error": str(e)})

    total_progress = total_brick + total_broken
    summary = {
        "images_processed":     len(results),
        "total_brick":          total_brick,
        "total_broken_brick":   total_broken,
        "total_crack":          total_crack,
        "total_flagged":        total_flagged,
        "total_progress_units": total_progress,
    }

    if req.planned and req.planned > 0:
        pct  = total_progress / req.planned * 100
        disc = abs(total_progress - req.planned) / req.planned
        summary["planned"]             = req.planned
        summary["progress_pct"]        = round(pct, 2)
        summary["discrepancy_pct"]     = round(disc * 100, 2)
        summary["discrepancy_warning"] = disc > DISCREPANCY_WARN

    return {"results": results, "summary": summary, "success": True}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting brick counter API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
