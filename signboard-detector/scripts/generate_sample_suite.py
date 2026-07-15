import os
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.run_pipeline import run_pipeline


SAMPLES = [
    ("mithuna_solutions", "MITHUNA SOLUTIONS", 236),
    ("monsil_lubricants", "MONSIL LUBRICANTS", 236),
    ("luna_cafe", "LUNA CAFE", 236),
    ("premo_studio", "PREMO STUDIO", 236),
    ("sensation_engineering", "SENSATION ENGINEERING", 236),
]


def create_sample_image(path: Path, label: str) -> None:
    h, w = 720, 1280
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (60, 100), (1220, 640), (220, 220, 220), -1)
    cv2.putText(img, label[:10], (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 7)
    cv2.putText(img, label[10:], (120, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 7)
    cv2.putText(img, 'SIGNBOARD', (120, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 5)
    cv2.imwrite(str(path), img)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_dir = root / 'sample_inputs'
    output_dir = root / 'sample_outputs'
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, label, board_width in SAMPLES:
        input_path = input_dir / f'{name}.jpg'
        create_sample_image(input_path, label)
        letter_result = run_pipeline(
            image_path=str(input_path),
            board_width_inches=board_width,
            output_dir=str(output_dir),
            threshold=140,
            bridge_gap=3,
            min_piece_size=40,
            ignore_margin=0.02,
            upscale=1.0,
            merge_mode='letter',
        )
        word_result = run_pipeline(
            image_path=str(input_path),
            board_width_inches=board_width,
            output_dir=str(output_dir),
            threshold=140,
            bridge_gap=6,
            min_piece_size=40,
            ignore_margin=0.02,
            upscale=1.0,
            merge_mode='word',
        )
        summary.append({
            'image': name,
            'piece_count_letter': letter_result['piece_count'],
            'piece_count_word': word_result['piece_count'],
            'pieces_letter': letter_result['pieces'],
            'pieces_word': word_result['pieces'],
        })

    (output_dir / 'results.json').write_text(__import__('json').dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
