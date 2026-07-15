from __future__ import annotations

import os
from pathlib import Path
from typing import List


def write_yolo_labels(image_path: str, boxes: List[dict], class_id: int = 0) -> None:
    image_path = Path(image_path)
    label_path = image_path.with_suffix('.txt')
    img = __import__('cv2').imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    lines = []
    for box in boxes:
        cx = (box['x'] + box['w'] / 2) / w
        cy = (box['y'] + box['h'] / 2) / h
        bw = box['w'] / w
        bh = box['h'] / h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding='utf-8')


def generate_yolo_dataset(root_dir: str, output_dir: str, labels: List[dict]) -> None:
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / 'images'
    labels_out = output_dir / 'labels'
    images_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)

    for item in labels:
        image_src = root_dir / item['image']
        image_dst = images_out / image_src.name
        label_dst = labels_out / image_src.with_suffix('.txt').name
        image_dst.write_bytes(image_src.read_bytes())
        lines = []
        for box in item['boxes']:
            cx = (box['x'] + box['w'] / 2) / item['image_width']
            cy = (box['y'] + box['h'] / 2) / item['image_height']
            bw = box['w'] / item['image_width']
            bh = box['h'] / item['image_height']
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_dst.write_text("\n".join(lines), encoding='utf-8')
