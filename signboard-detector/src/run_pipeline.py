from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.detect import detect_regions
from src.preprocess import preprocess_image
from src.size_estimate import estimate_piece_sizes
from src.yolo_detect import detect_regions_yolo, load_yolo_model


def draw_annotations(image: np.ndarray, boxes: list[dict], output_path: str) -> None:
    canvas = image.copy()
    for idx, box in enumerate(boxes, start=1):
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, str(idx), (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(output_path, canvas)


def export_yolo_labels(image: np.ndarray, boxes: list[dict], label_path: Path) -> None:
    h, w = image.shape[:2]
    lines = []
    for box in boxes:
        cx = (box["x"] + box["w"] / 2) / w
        cy = (box["y"] + box["h"] / 2) / h
        bw = box["w"] / w
        bh = box["h"] / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    image_path: str,
    board_width_inches: float = 236,
    output_dir: str | None = None,
    threshold: int = 140,
    bridge_gap: int = 3,
    min_piece_size: int = 50,
    ignore_margin: float = 0.02,
    upscale: float = 1.0,
    merge_mode: str = "letter",
    yolo_weights: str | None = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    export_yolo: bool = False,
):
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = str(image_path.parent)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    processed, _, board_box = preprocess_image(image, upscale=upscale)
    if yolo_weights:
        model = load_yolo_model(yolo_weights)
        boxes = detect_regions_yolo(processed, model, conf=yolo_conf, iou=yolo_iou)
    else:
        boxes = detect_regions(
            processed,
            threshold=threshold,
            bridge_gap=bridge_gap,
            min_piece_size=min_piece_size,
            ignore_margin=ignore_margin,
            merge_mode=merge_mode,
        )

    estimates = estimate_piece_sizes(boxes, board_width_inches, board_box, image.shape)

    stem = image_path.stem
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    json_path = output_dir / f"{stem}_results.json"
    # Ultralytics matches labels to images by filename stem, e.g. image.jpg
    # requires labels/image.txt.  Keeping the same stem makes these exports
    # immediately usable as a YOLO training dataset.
    label_path = output_dir / f"{stem}.txt"

    draw_annotations(image, boxes, str(annotated_path))
    if export_yolo:
        export_yolo_labels(image, boxes, label_path)

    payload = {
        "image": str(image_path.name),
        "piece_count": len(estimates),
        "pieces": estimates,
        "yolo_label_file": str(label_path.name) if export_yolo else None,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect signboard letters and logos")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--board-width", type=float, default=236.0, help="Known board width in inches")
    parser.add_argument("--output-dir", default="sample_outputs", help="Directory for outputs")
    parser.add_argument("--threshold", type=int, default=140, help="Adaptive threshold cutoff")
    parser.add_argument("--bridge-gap", type=int, default=3, help="Gap-bridging dilation size")
    parser.add_argument("--min-piece-size", type=int, default=50, help="Minimum contour area")
    parser.add_argument("--ignore-margin", type=float, default=0.02, help="Margin to ignore around the border")
    parser.add_argument("--upscale", type=float, default=1.0, help="Optional super-resolution-like upscale")
    parser.add_argument("--merge-mode", choices=["letter", "word"], default="letter", help="Group letters into words or keep them separate")
    parser.add_argument("--yolo-weights", default=None, help="Path to YOLO model weights for inference")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--export-yolo-labels", action="store_true", help="Export YOLO-format label file for detected pieces")
    args = parser.parse_args()
    run_pipeline(
        image_path=args.image,
        board_width_inches=args.board_width,
        output_dir=args.output_dir,
        threshold=args.threshold,
        bridge_gap=args.bridge_gap,
        min_piece_size=args.min_piece_size,
        ignore_margin=args.ignore_margin,
        upscale=args.upscale,
        merge_mode=args.merge_mode,
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        export_yolo=args.export_yolo_labels,
    )


if __name__ == "__main__":
    main()
