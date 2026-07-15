from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def train_yolo(
    data: str,
    weights: str = 'yolov8n.pt',
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    project: str = 'models',
    name: str = 'signboard',
    device: str = '0',
):
    if YOLO is None:
        raise ImportError('ultralytics is required to train YOLO models. Install with pip install ultralytics')
    model = YOLO(weights)
    model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a YOLOv8 signboard detector')
    parser.add_argument('--data', required=True, help='Path to YOLO data.yaml file')
    parser.add_argument('--weights', default='yolov8n.pt', help='Base YOLO weights to start from')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--project', default='models', help='Project folder for training results')
    parser.add_argument('--name', default='signboard', help='Name for this training run')
    parser.add_argument('--device', default='0', help='Device identifier for training (e.g. 0 or cpu)')
    args = parser.parse_args()
    train_yolo(
        data=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == '__main__':
    main()
