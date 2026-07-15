from __future__ import annotations

import argparse
from pathlib import Path

from src.yolo_train import generate_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate YOLO dataset from labeled images')
    parser.add_argument('--root', required=True, help='Root directory containing source images')
    parser.add_argument('--output', default='yolo_dataset', help='Output dataset directory')
    parser.add_argument('--labels', required=True, help='Path to JSON labels file')
    args = parser.parse_args()

    import json
    labels = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    generate_yolo_dataset(args.root, args.output, labels)


if __name__ == '__main__':
    main()
