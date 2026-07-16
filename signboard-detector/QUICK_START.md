# 🚀 Quick Start - Training Your Model

## What You Need to Do

### 1️⃣ Save Your Signboard Images

Save all the signboard images you provided (the attachments) to this folder:

```
d:\prototype\signboard-detector\training_data\raw_images\
```

**Supported formats:** `.jpg`, `.jpeg`, `.png`

**Example images to save:**
- senson_engineering.jpg
- spro_hair_studio.jpg
- interior_design.jpg
- anytime_fitness.jpg
- luna_cafe.jpg
- mithuna_solutions.jpg
- moisii_lubricants.jpg
- premo_studio.jpg
- globa_art.jpg
- And any other signboard images you have

### 2️⃣ Run the Training Pipeline

Choose ONE of these options:

#### Option A: PowerShell (Recommended for Windows)
```powershell
cd d:\prototype\signboard-detector
.\train.ps1
```

#### Option B: Command Prompt
```cmd
cd d:\prototype\signboard-detector
train.bat
```

#### Option C: Manual Steps
```powershell
# Step 1: Prepare data
python prepare_training_data.py

# Step 2: Start training
python src/train_yolo.py --data data.yaml --epochs 50
```

---

## What Happens During Training

1. **Data Preparation** (1-2 minutes)
   - Scans all images in `training_data/raw_images/`
   - Detects signboard pieces (letters, logos)
   - Generates YOLO format labels
   - Saves to `training_data/images/` and `training_data/labels/`

2. **Training** (5-30 minutes depending on image count and GPU)
   - Trains YOLOv8 model on detected pieces
   - Validates on 20% of images
   - Saves results to `models/signboard/`

3. **Results**
   - Best model: `models/signboard/weights/best.pt`
   - Training graphs: `models/signboard/results.png`
   - Validation metrics: Shown in console

---

## 📁 Directory Structure After Training

```
signboard-detector/
├── training_data/
│   ├── raw_images/              ← YOUR IMAGES GO HERE
│   │   ├── senson_engineering.jpg
│   │   ├── spro_hair_studio.jpg
│   │   └── (more images)
│   ├── images/                  ← Processed images (auto-generated)
│   └── labels/                  ← YOLO labels (auto-generated)
├── models/                      ← Training results
│   └── signboard/
│       ├── weights/
│       │   ├── best.pt          ← Use this model!
│       │   └── last.pt
│       ├── results.png          ← Training curves
│       └── confusion_matrix.png
├── data.yaml                    ← Dataset config
├── train.ps1                    ← Training script (PowerShell)
├── train.bat                    ← Training script (Batch)
├── prepare_training_data.py     ← Data prep script
└── src/
    ├── train_yolo.py           ← Training code
    └── (other files)
```

---

## 🎯 Customization

### Training Parameters

Edit the command to adjust training:

```powershell
python src/train_yolo.py `
    --data data.yaml `
    --epochs 100 `
    --batch 8 `
    --imgsz 640 `
    --device 0 `
    --project models `
    --name signboard
```

**Key parameters:**
- `--epochs`: Number of training iterations (default: 50, more = longer training)
- `--batch`: Batch size (default: 16, reduce if out of memory)
- `--imgsz`: Input image size (default: 640)
- `--device`: GPU device ID (0 for first GPU, -1 for CPU)

### Data Preparation Parameters

Edit `prepare_training_data.py` line 46 to adjust:

```python
prepare_training_data(
    board_width=236,        # Known board width in inches
    threshold=140,          # Detection sensitivity
    bridge_gap=3,           # Gap bridging for word grouping
    min_piece_size=50,      # Minimum piece size in pixels
)
```

---

## ⚠️ Troubleshooting

### "No images found"
- Check images are in: `training_data\raw_images\`
- Use supported formats: `.jpg`, `.jpeg`, `.png`

### "Out of memory" during training
```powershell
# Use smaller batch size
python src/train_yolo.py --data data.yaml --epochs 50 --batch 8
```

### Training takes too long
```powershell
# Reduce epochs
python src/train_yolo.py --data data.yaml --epochs 20 --batch 16
```

### Poor detection accuracy
- Add more training images (50+)
- Ensure images have good lighting
- Vary image angles and perspectives
- Increase epochs: `--epochs 100`

---

## ✅ Testing Your Model

After training completes, use the trained model:

```python
from src.yolo_detect import detect_regions_yolo, load_yolo_model

# Load the trained model
model = load_yolo_model("models/signboard/weights/best.pt")

# Run inference
from src.run_pipeline import run_pipeline
run_pipeline(
    "path/to/test_image.jpg",
    yolo_weights="models/signboard/weights/best.pt"
)
```

---

## 📊 Expected Performance

**With 10-20 images:**
- Training time: 2-5 minutes
- Accuracy: 60-75%

**With 50+ images:**
- Training time: 10-30 minutes
- Accuracy: 80-95%

**With 100+ images:**
- Training time: 30-60 minutes
- Accuracy: 90-98%

---

## 🎓 Next Steps

1. ✅ Save images to `training_data/raw_images/`
2. ✅ Run `.\train.ps1` or `train.bat`
3. ✅ Wait for training to complete
4. ✅ Check `models/signboard/weights/best.pt`
5. ✅ Use the model for inference!

**Happy training! 🚀**
