from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


def render_text_on_background(text: str, image_size=(1024, 512), background_color=(255, 255, 255)):
    img = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 4.0
    thickness = 8
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = max(20, (image_size[0] - size[0]) // 2)
    y = max(size[1] + 20, (image_size[1] + size[1]) // 2)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img, [dict(x=x, y=y-size[1], w=size[0], h=size[1])]


def create_synthetic_dataset(output_dir: str, samples: List[str], image_size=(1024, 512)) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / 'images'
    label_dir = output_dir / 'labels'
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    for idx, text in enumerate(samples, start=1):
        name = f'synth_{idx:03d}.jpg'
        img, boxes = render_text_on_background(text, image_size=image_size)
        cv2.imwrite(str(image_dir / name), img)
        h, w = image_size[1], image_size[0]
        label_lines = []
        for box in boxes:
            cx = (box['x'] + box['w'] / 2) / w
            cy = (box['y'] + box['h'] / 2) / h
            bw = box['w'] / w
            bh = box['h'] / h
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (label_dir / f'synth_{idx:03d}.txt').write_text('\n'.join(label_lines), encoding='utf-8')


if __name__ == '__main__':
    create_synthetic_dataset('synthetic_dataset', ['PREMO STUDIO', 'SENSATION ENGINEERING', 'LUNA CAFE', 'MITHUNA SOLUTIONS'])
