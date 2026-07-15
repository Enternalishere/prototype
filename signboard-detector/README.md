# Signboard Letter/Logo Detector

A lightweight OpenCV-based signboard detector that finds letter and logo regions on boards, corrects uneven lighting and perspective, estimates piece size in inches from a known board width, and writes annotated images plus JSON output.

## Features

- CLAHE-based lighting correction and denoising
- Basic perspective/angle correction using contour-based warping
- Adaptive-threshold connected-component detection for letters and logo pieces
- Gap-bridging and minimum-size tuning for word-level vs letter-level grouping
- Pixel-to-inch size estimation from a known board width
- CLI workflow that runs end to end for one image at a time

## Setup

```bash
cd signboard-detector
python -m pip install -r requirements.txt
```

## Run on a new image

```bash
python src/run_pipeline.py --image path/to/signboard.jpg --board-width 236 --output-dir sample_outputs
```

## Generate sample outputs

To generate outputs for the built-in sample files and compare letter vs word grouping:

```bash
python scripts/generate_sample_suite.py
```

### Parameters

- `--threshold`: adaptive-threshold cutoff value for foreground extraction.
- `--bridge-gap`: dilation size used to bridge nearby pieces into a larger word-level object.
- `--min-piece-size`: minimum contour area to keep; smaller regions are ignored as noise.
- `--ignore-margin`: fraction of image width/height to ignore around borders.
- `--upscale`: optional resize factor for low-resolution images.
- `--merge-mode`: `letter` keeps pieces separate; `word` merges nearby pieces into larger groups.
- `--yolo-weights`: optional YOLO weights path to switch from OpenCV contour detection to model inference.
- `--yolo-conf`: confidence threshold when running YOLO inference.
- `--yolo-iou`: NMS IoU threshold when running YOLO inference.

## How this handles lighting, angle, and image quality issues

- Uneven lighting is corrected with CLAHE in Lab color space.
- Rotated or angled boards are corrected with a contour-based perspective transform and optional deskew rotation.
- Noise and low-resolution inputs are reduced with bilateral filtering and a simple resize upscaling step.
- Adaptive thresholding and contour filtering make the method robust on signs with cluttered backgrounds.

## Phase 2 / production path

Once thousands of labeled signboard images are available, the same workflow can be upgraded to a trained YOLOv8 segmentation model.

Suggested production path:

1. Collect a large dataset of signboards with masks for letters and logos.
2. Use synthetic augmentation to expand the dataset: render text and logos in multiple fonts, colors, angles, and lighting onto varied backgrounds.
3. Train a YOLOv8 segmentation model to predict word and logo masks directly.
4. Use the classical OpenCV pipeline as a fast baseline and fallback when no model is available.

This hybrid approach allows a robust production system without waiting for a very large manually labeled dataset.

## Optional model weights

The `models/` folder is reserved for YOLOv8 weights or segmentation models.

## YOLO training and inference

### Generate a YOLO dataset

There are two paths:

1. Manual labels:
   - Place each image in a folder
   - Create a `.txt` file for each image with normalized YOLO box coordinates

2. Synthetic sample dataset:

```bash
python src/synthetic_data.py
```

This writes a dataset to `synthetic_dataset/images` and `synthetic_dataset/labels`.

3. Export a dataset from a manifest:

```bash
python src/make_yolo_dataset.py --root . --labels labels_manifest.json --output yolo_dataset
```

### Train a YOLOv8 model

```bash
python src/train_yolo.py --data data.yaml --weights yolov8n.pt --epochs 50 --batch 16 --imgsz 640 --project models --name signboard
```

This will save `models/signboard/weights/best.pt`.

### Run inference with YOLO

```bash
python src/run_pipeline.py --image sample_inputs/premo_studio.jpg --board-width 236 --yolo-weights models/signboard/weights/best.pt --export-yolo-labels --output-dir sample_outputs
```

## Output files

Each run writes:

- an annotated image with numbered boxes
- a JSON file with piece ID, bbox, and estimated width/height in inches

## Evaluation

A notebook is included at [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) for IoU, precision, and recall comparisons against a small hand-labeled set.
