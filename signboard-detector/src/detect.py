from __future__ import annotations

import cv2
import numpy as np

from src.merge_split import merge_boxes


def detect_regions(
    image: np.ndarray,
    threshold: int = 150,
    bridge_gap: int = 3,
    min_piece_size: int = 50,
    ignore_margin: float = 0.02,
    merge_mode: str = "letter",
):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    if threshold is not None:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        binary = adaptive

    if bridge_gap > 0:
        kernel_size = max(1, int(bridge_gap) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_piece_size:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= ignore_margin * max(image.shape[1], 1) or y <= ignore_margin * max(image.shape[0], 1):
            continue
        if x + w >= (1 - ignore_margin) * max(image.shape[1], 1) or y + h >= (1 - ignore_margin) * max(image.shape[0], 1):
            continue
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area)})

    if merge_mode == "word":
        boxes = merge_boxes(boxes, gap_px=max(3, bridge_gap * 2))
    boxes = sorted(boxes, key=lambda b: (b["y"], b["x"]))
    return boxes
