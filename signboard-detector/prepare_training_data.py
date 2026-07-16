#!/usr/bin/env python
"""
Data preparation script for signboard training.
Converts raw signboard images into YOLO format training data.
"""

import os
import sys
import cv2
import json
from pathlib import Path

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.run_pipeline import run_pipeline


def prepare_training_data(
    raw_images_dir: str = "training_data/raw_images",
    output_images_dir: str = "training_data/images",
    output_labels_dir: str = "training_data/labels",
    board_width: float = 236,
    threshold: int = 140,
    bridge_gap: int = 3,
    min_piece_size: int = 50,
):
    """
    Process raw signboard images and generate YOLO training data.
    
    Args:
        raw_images_dir: Directory containing raw signboard images
        output_images_dir: Directory to save processed images
        output_labels_dir: Directory to save YOLO labels
        board_width: Board width in inches for size estimation
        threshold: Adaptive threshold value
        bridge_gap: Gap bridging size
        min_piece_size: Minimum piece size in pixels
    """
    
    raw_dir = Path(raw_images_dir)
    images_dir = Path(output_images_dir)
    labels_dir = Path(output_labels_dir)
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print(f"❌ Raw images directory not found: {raw_dir}")
        print(f"📁 Please save your signboard images to: {raw_dir.absolute()}")
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(raw_dir.glob(f'*{ext}'))
    
    if not image_files:
        print(f"❌ No images found in {raw_dir}")
        return
    
    print(f"✅ Found {len(image_files)} images to process")
    print("-" * 60)
    
    processed_count = 0
    
    for idx, image_path in enumerate(image_files, 1):
        image_name = image_path.stem
        
        try:
            print(f"[{idx}/{len(image_files)}] Processing: {image_name}")
            
            # Run detection pipeline
            run_pipeline(
                str(image_path),
                board_width_inches=board_width,
                output_dir=str(labels_dir),  # Will save JSON and labels here
                threshold=threshold,
                bridge_gap=bridge_gap,
                min_piece_size=min_piece_size,
                merge_mode="letter",
                export_yolo=True,
            )
            
            # Copy processed image to training images directory
            output_image_path = images_dir / f"{image_name}.jpg"
            image = cv2.imread(str(image_path))
            if image is not None:
                cv2.imwrite(str(output_image_path), image)
                processed_count += 1
                print(f"  ✓ Saved to {output_image_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_name}: {e}")
    
    print("-" * 60)
    print(f"\n📊 Training Data Summary:")
    print(f"   Images processed: {processed_count}")
    print(f"   Images saved to: {images_dir.absolute()}")
    print(f"   Labels saved to: {labels_dir.absolute()}")
    print(f"\n✅ Ready to train! Run:")
    print(f"   python src/train_yolo.py --data data.yaml --epochs 50")


if __name__ == "__main__":
    prepare_training_data()
