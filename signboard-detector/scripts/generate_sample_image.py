from pathlib import Path
import cv2
import numpy as np


def create_sample_image(path: str) -> None:
    h, w = 720, 1280
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (80, 120), (1180, 620), (220, 220, 220), -1)
    cv2.putText(img, 'MITHUNA', (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (20, 20, 20), 8)
    cv2.putText(img, 'SOLUTIONS', (140, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (20, 20, 20), 8)
    cv2.putText(img, 'LUNA', (140, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 8)
    cv2.putText(img, 'CAFE', (140, 590), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (20, 20, 20), 8)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    out = Path(__file__).resolve().parent.parent / 'sample_inputs' / 'mithuna_solutions.jpg'
    out.parent.mkdir(parents=True, exist_ok=True)
    create_sample_image(str(out))
