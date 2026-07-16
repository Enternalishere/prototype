# Training Data Setup Guide

## 📁 Directory Structure

Your training workspace is organized as follows:

```
signboard-detector/
├── training_data/                 ← Your training data folder
│   ├── raw_images/               ← SAVE YOUR IMAGES HERE
│   │   ├── signboard_1.jpg
│   │   ├── signboard_2.jpg
│   │   └── ... (more images)
│   ├── images/                   ← Processed images (auto-generated)
│   └── labels/                   ← YOLO labels (auto-generated)
├── prepare_training_data.py      ← Data preparation script
├── src/
│   ├── train_yolo.py            ← Training script
│   └── (other source files)
└── ...
```

## 🎯 Step-by-Step Guide

### Step 1: Save Your Images
1. Take screenshots or save the signboard images from the attachments
2. Place them in this directory: **`training_data/raw_images/`**
3. Supported formats: `.jpg`, `.jpeg`, `.png`

Example:
```
training_data/raw_images/
├── senson_engineering.jpg
├── spro_hair_studio.jpg
├── interior_design.jpg
├── anytime_fitness.jpg
├── luna_cafe.jpg
├── mithuna_solutions.jpg
├── moisii_lubricants.jpg
├── premo_studio.jpg
├── globa_art.jpg
└── (and more...)
```

### Step 2: Prepare Training Data
Run the preparation script to process all images and generate YOLO labels:

```bash
python prepare_training_data.py
```

This will:
- ✅ Detect all letter/logo pieces in each image
- ✅ Generate YOLO format labels (`.txt` files)
- ✅ Copy processed images to `training_data/images/`
- ✅ Save detection metadata as JSON

### Step 3: Review Generated Data
After processing, check:
- **Images**: `training_data/images/`
- **Labels**: `training_data/labels/`

Each image should have a corresponding `.txt` file with YOLO format bounding boxes.

### Step 4: Train the Model
Run the training script:

```bash
python src/train_yolo.py --data data.yaml --epochs 50 --batch 16
```

**Key parameters:**
- `--epochs`: Number of training iterations (default: 50)
- `--batch`: Batch size (default: 16, reduce if out of memory)
- `--imgsz`: Image size (default: 640)
- `--project`: Output directory for results (default: `models`)

### Step 5: Monitor Training
Training results will be saved to `models/signboard/` with:
- Trained weights
- Training curves (plots)
- Validation results

## 📊 Data Format Details

### YOLO Label Format (`.txt`)
Each image has a corresponding `.txt` file with one line per detection:

```
0 0.505 0.169 0.629 0.349
0 0.234 0.456 0.123 0.234
```

Format: `<class> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`

Where:
- `class`: 0 (all pieces are class 0)
- Coordinates are normalized to [0, 1]

### Image Requirements
- **Resolution**: 640×640 recommended (auto-resized by YOLO)
- **Formats**: JPG, PNG
- **Quality**: High quality, well-lit images
- **Count**: At least 50-100 images for decent model

## 🔧 Troubleshooting

### No images found?
```bash
# Check that images exist
dir training_data\raw_images\
```

### Permission errors?
```bash
# Run as administrator if needed
# Or check file permissions
```

### Out of memory during training?
```bash
# Reduce batch size
python src/train_yolo.py --data data.yaml --epochs 50 --batch 8
```

### Poor detection results?
- Use higher quality images
- Ensure consistent lighting
- Vary image angles and perspectives
- Add 50+ more images to training set

## 📝 Next Steps

1. **Save images** to `training_data/raw_images/`
2. **Run preparation**: `python prepare_training_data.py`
3. **Start training**: `python src/train_yolo.py --data data.yaml --epochs 50`
4. **Evaluate results** in `models/signboard/`

Good luck! 🚀
