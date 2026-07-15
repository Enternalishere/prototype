from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise ImportError("ultralytics is required for YOLO inference. Install it with pip install ultralytics")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(weights_path)


def detect_regions_yolo(
    image: np.ndarray,
    model,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 100,
) -> list[dict]:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = model.predict(source=image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    boxes = []
    for res in results:
        if res.boxes is None:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for coords, score in zip(xyxy, confs):
            x1, y1, x2, y2 = coords.astype(int)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            boxes.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(w),
                "h": int(h),
                "score": float(score),
            })
    return boxes
