"""
RunPod Serverless Handler — Masonry Brick Counter
Bulk image upload → YOLO detection → brick counts + progress report
Based on: Magdy et al. (2025), Buildings 15(14):2456

Input format:
{
  "input": {
    "images": [
      {"name": "wall1.jpg", "data": "<base64-encoded-image>"},
      ...
    ],
    "planned": 500,          // optional: as-planned brick count for progress %
    "conf_threshold": 0.6    // optional: confidence threshold (default 0.6)
  }
}

Output format:
{
  "results": [
    {
      "name": "wall1.jpg",
      "brick": 12,
      "broken_brick": 2,
      "crack": 1,
      "flagged": 5,
      "progress_units": 14,
      "annotated_image": "<base64-encoded-annotated-jpg>"
    },
    ...
  ],
  "summary": {
    "images_processed": 3,
    "total_brick": 30,
    "total_broken_brick": 5,
    "total_crack": 3,
    "total_flagged": 12,
    "total_progress_units": 35,
    "planned": 500,
    "progress_pct": 7.0,
    "discrepancy_warning": false
  }
}
"""

import os
import sys
import logging
import subprocess
import base64
from pathlib import Path
from typing import Any

# ── Cache dirs BEFORE any ML imports ─────────────────────────────────────────
VOLUME_DIR = Path('/runpod-volume')
PIP_CACHE  = VOLUME_DIR / 'pip_cache'
HF_CACHE   = VOLUME_DIR / 'huggingface_cache'
MODEL_DIR  = VOLUME_DIR / 'models'

for d in [PIP_CACHE, HF_CACHE, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ['PIP_CACHE_DIR']       = str(PIP_CACHE)
os.environ['HF_HOME']             = str(HF_CACHE)
os.environ['TRANSFORMERS_CACHE']  = str(HF_CACHE)
os.environ['XDG_CACHE_HOME']      = str(VOLUME_DIR / '.cache')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CONF_DEFAULT     = 0.60
DISCREPANCY_WARN = 0.05
COLOURS = {
    0: (0,   200,  50),   # brick        green
    1: (0,   140, 255),   # broken_brick orange
    2: (0,   0,   220),   # crack        red
}
CLASS_NAMES = {0: "brick", 1: "broken_brick", 2: "crack"}


class BrickCounter:

    def __init__(self):
        self.model = None
        self.ready = False

    def install_deps(self) -> bool:
        pkgs = [
            ['numpy<2.0'],
            ['torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu121'],
            ['ultralytics', 'opencv-python-headless'],
        ]
        for group in pkgs:
            cmd = [sys.executable, '-m', 'pip', 'install',
                   '--cache-dir', str(PIP_CACHE), '--quiet'] + group
            logger.info(f"pip install {' '.join(group)}")
            try:
                subprocess.run(cmd, check=True, timeout=600)
            except Exception as e:
                logger.error(f"Install failed: {e}")
                return False
        return True

    def load_model(self) -> bool:
        try:
            from ultralytics import YOLO

            # Prefer model cached on network volume
            cached = MODEL_DIR / 'masonry.pt'
            if cached.exists():
                logger.info(f"Loading from volume cache: {cached}")
                self.model = YOLO(str(cached))
            else:
                # First boot: model must be baked into image or downloaded
                # Place masonry.pt at /masonry.pt in the Docker image
                baked = Path('/masonry.pt')
                if baked.exists():
                    import shutil
                    shutil.copy(baked, cached)
                    logger.info(f"Copied baked model to volume cache")
                    self.model = YOLO(str(cached))
                else:
                    logger.error("masonry.pt not found at /masonry.pt or volume cache")
                    return False

            self.ready = True
            logger.info("Model ready")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

    def setup(self) -> bool:
        if self.ready:
            return True
        if not self.install_deps():
            return False
        if not self.load_model():
            return False
        return True

    def detect_image(self, img_bgr, conf_thresh: float) -> dict:
        import cv2

        results  = self.model(img_bgr, conf=0.01, verbose=False)
        counts   = {0: 0, 1: 0, 2: 0}
        flagged  = 0
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
                cv2.putText(annotated, f"?{label} {conf:.2f}", (x1, max(0, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
            else:
                counts[cls_id] += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

        return {
            "brick":        counts[0],
            "broken_brick": counts[1],
            "crack":        counts[2],
            "flagged":      flagged,
            "annotated_bgr": annotated,
        }

    def process(self, job_input: dict) -> dict:
        import cv2
        import numpy as np

        images       = job_input.get("images", [])
        planned      = job_input.get("planned", None)
        conf_thresh  = float(job_input.get("conf_threshold", CONF_DEFAULT))

        if not images:
            return {"error": "No images provided. Send 'images': [{name, data}]", "success": False}

        results          = []
        total_brick      = 0
        total_broken     = 0
        total_crack      = 0
        total_flagged    = 0

        for img_item in images:
            name = img_item.get("name", "unknown")
            b64  = img_item.get("data", "")

            try:
                img_bytes = base64.b64decode(b64)
                arr       = np.frombuffer(img_bytes, np.uint8)
                img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    results.append({"name": name, "error": "Could not decode image"})
                    continue

                det = self.detect_image(img_bgr, conf_thresh)

                # Encode annotated image back to base64 JPEG
                _, buf = cv2.imencode('.jpg', det["annotated_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                ann_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

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
                    "name":             name,
                    "brick":            n_brick,
                    "broken_brick":     n_broken,
                    "crack":            n_crack,
                    "flagged":          n_flagged,
                    "progress_units":   progress,
                    "annotated_image":  ann_b64,
                })

            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                results.append({"name": name, "error": str(e)})

        total_progress = total_brick + total_broken
        summary = {
            "images_processed":  len(results),
            "total_brick":       total_brick,
            "total_broken_brick": total_broken,
            "total_crack":       total_crack,
            "total_flagged":     total_flagged,
            "total_progress_units": total_progress,
        }

        if planned and int(planned) > 0:
            planned = int(planned)
            pct  = total_progress / planned * 100
            disc = abs(total_progress - planned) / planned
            summary["planned"]              = planned
            summary["progress_pct"]         = round(pct, 2)
            summary["discrepancy_pct"]      = round(disc * 100, 2)
            summary["discrepancy_warning"]  = disc > DISCREPANCY_WARN

        return {"results": results, "summary": summary, "success": True}


# ── Global handler (persists across warm requests) ────────────────────────────
_counter = None


def handler(event: dict) -> dict:
    global _counter

    job_input = event.get("input", {})
    if not job_input:
        return {"error": "No input provided", "success": False}

    if _counter is None:
        _counter = BrickCounter()
        if not _counter.setup():
            return {"error": "Setup failed — check logs", "success": False}

    return _counter.process(job_input)


if __name__ == '__main__':
    import runpod
    logger.info("Starting RunPod serverless brick counter...")
    runpod.serverless.start({'handler': handler})
