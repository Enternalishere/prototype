from __future__ import annotations

import json
import sys
from pathlib import Path
import shutil
import cv2

ROOT = Path(__file__).resolve().parents[1]
ROOT = ROOT.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess_image
from src.run_pipeline import run_pipeline

TEST_ROOT = ROOT.parent / 'test_images'
OUT_DIR = ROOT / 'sample_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

images = ['4.JPG','6.JPG','19.JPG','29.JPG','90.JPG','92.JPG','test2.JPG','test3.JPG','test4.JPG']

summary = []
print('=== SMOKE TEST ===')
for name in images:
    image_path = TEST_ROOT / name
    metadata = {'name': name, 'exists': image_path.exists(), 'error': None, 'piece_count': None, 'bboxes_valid': False}
    if not image_path.exists():
        metadata['error'] = 'missing image file'
        summary.append(metadata)
        print('MISSING', name)
        continue
    try:
        payload = run_pipeline(str(image_path), board_width_inches=236, output_dir=str(OUT_DIR))
        result_json = OUT_DIR / f'{image_path.stem}_results.json'
        annotated = OUT_DIR / f'{image_path.stem}_annotated.jpg'
        metadata['piece_count'] = payload.get('piece_count')
        metadata['exists'] = result_json.exists() and annotated.exists()
        metadata['bboxes_valid'] = isinstance(payload.get('pieces'), list) and all(
            isinstance(piece.get('bbox'), list) and len(piece['bbox']) == 4 for piece in payload.get('pieces', [])
        )
        print('OK', name, 'count=', metadata['piece_count'], 'exists=', metadata['exists'], 'bboxes=', metadata['bboxes_valid'])
    except Exception as exc:
        metadata['error'] = repr(exc)
        print('ERROR', name, repr(exc))
    summary.append(metadata)

print('\n=== EDGE CASES ===')
edge_base = TEST_ROOT / '4.JPG'
rot_path = TEST_ROOT / '4_rotated.JPG'
lowres_path = TEST_ROOT / '4_lowres.JPG'
if edge_base.exists():
    img = cv2.imread(str(edge_base))
    if img is not None:
        if not rot_path.exists():
            h, w = img.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(rot_path), rotated)
        if not lowres_path.exists():
            low = cv2.resize(img, (max(16, img.shape[1] // 4), max(16, img.shape[0] // 4)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(lowres_path), low)

for label, case_path in [('rotated', rot_path), ('lowres', lowres_path)]:
    if not case_path.exists():
        print('missing edge case file', case_path)
        continue
    pre, gray, board_box = preprocess_image(cv2.imread(str(case_path)), upscale=1.0)
    cv2.imwrite(str(OUT_DIR / f'{case_path.stem}_preprocessed.jpg'), pre)
    cv2.imwrite(str(OUT_DIR / f'{case_path.stem}_preprocessed_gray.jpg'), gray)
    try:
        payload = run_pipeline(str(case_path), board_width_inches=236, output_dir=str(OUT_DIR))
        print('EDGE', label, 'board_box=', board_box, 'piece_count=', payload.get('piece_count'))
    except Exception as exc:
        print('EDGE ERROR', label, repr(exc))

print('\n=== MERGE/SPLIT CHECK ===')
base_name = '4'
base_path = TEST_ROOT / '4.JPG'
letter_path = TEST_ROOT / '4_letter.JPG'
word_path = TEST_ROOT / '4_word.JPG'
if base_path.exists():
    shutil.copyfile(str(base_path), str(letter_path))
    shutil.copyfile(str(base_path), str(word_path))
    letter_payload = run_pipeline(str(letter_path), board_width_inches=236, output_dir=str(OUT_DIR), merge_mode='letter')
    word_payload = run_pipeline(str(word_path), board_width_inches=236, output_dir=str(OUT_DIR), merge_mode='word')
    print('MERGE/SPLIT letter_count=', letter_payload.get('piece_count'), 'word_count=', word_payload.get('piece_count'))
    if letter_payload.get('piece_count') == word_payload.get('piece_count'):
        print('WARNING: letter and word mode piece counts are identical')
else:
    print('MERGE/SPLIT base image missing', base_path)

print('\n=== SIZE ACCURACY SANITY CHECK ===')
known_path = TEST_ROOT / '19.JPG'
if known_path.exists():
    payload = run_pipeline(str(known_path), board_width_inches=236, output_dir=str(OUT_DIR))
    if payload.get('pieces'):
        piece = payload['pieces'][0]
        board_only = preprocess_image(cv2.imread(str(known_path)), upscale=1.0)[2]
        board_px = max(board_only.get('w', 1), 1)
        expected_in = piece['bbox'][2] * 236 / board_px
        print('SIZE CHECK piece 1 width_in=', piece['width_in'], 'computed expected=', round(expected_in,2), 'board_px=', board_px)
    else:
        print('SIZE CHECK no pieces found for', known_path)
else:
    print('SIZE CHECK image missing', known_path)

print('\n=== SUMMARY ===')
print(json.dumps(summary, indent=2))
