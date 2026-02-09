# ASV-INSPECT - Complete Workflow Guide

This document provides a step-by-step workflow for using ASV-INSPECT.

## 📋 Table of Contents

1. [Initial Setup](#initial-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Phase](#training-phase)
4. [Inspection Phase](#inspection-phase)
5. [API Deployment](#api-deployment)
6. [Example Commands](#example-commands)

---

## Initial Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python quick_start.py
```

This script will:
- Check project structure
- Verify dependencies
- Count dataset files
- Guide you through next steps

---

## Dataset Preparation

### 1. Collect Training Images

- Capture 50-200+ images of correct assemblies
- Ensure consistent lighting and angle
- Cover natural variations (slight position differences are OK)

### 2. Label Images (YOLO Format)

Use labeling tools like:
- [LabelImg](https://github.com/heartexlabs/labelImg)
- [CVAT](https://github.com/opencv/cvat)
- [Roboflow](https://roboflow.com/)

Export as **YOLO format**

### 3. Organize Dataset

```
dataset/
├── images/
│   ├── assembly_001.jpg
│   ├── assembly_002.jpg
│   └── ...
├── labels/
│   ├── assembly_001.txt
│   ├── assembly_002.txt
│   └── ...
└── obj.names
```

### 4. Create obj.names

List component classes (one per line):

```
bolt
bearing
screw
washer
nut
```

---

## Training Phase

### Step 1: Train YOLO Detector

**Your Dataset:**
- 148 training images (gearbox assemblies)
- 3 component classes: bolt, bearing, oil jet
- Images: `dataset/images/*.JPG`
- Labels: `dataset/labels/*.txt` (YOLO format)

**Training Command:**

```bash
cd src
python train_detector.py --dataset ../dataset --epochs 100 --device cpu
```

**Recommended Options for Your Dataset:**
- `--model-size s`: Small model (good balance for 148 images)
- `--epochs 100`: Good starting point for 148 images
- `--batch 16`: Adjust based on your RAM (8-32)
- `--device cpu`: Use 'cuda' if you have a GPU

**Example with options:**
```bash
python train_detector.py --dataset ../dataset --epochs 100 --model-size s --batch 16 --device cpu
```

**Expected time:** 15-45 minutes on CPU (depends on hardware)

**Output:** `models/detector/train/weights/best.pt`

### Step 2: Build Golden Model

**What This Does:**
Analyzes your 148 training images to learn where bolts, bearings, and oil jets are typically located in a correctly assembled gearbox.

**Command:**
```bash
python build_golden_model.py --dataset ../dataset
```

**Recommended Options for Your Gearbox Dataset:**
- `--eps 0.05`: Default works well for most assemblies
- `--min-samples 2`: At least 2 occurrences to be considered expected
- `--min-occurrence 0.5`: Component must appear in 50% of images (74+ images)

**Example with custom parameters:**
```bash
python build_golden_model.py --dataset ../dataset --eps 0.05 --min-samples 2 --min-occurrence 0.5
```

**Expected time:** 5-10 seconds

**Output:** `models/golden_model/golden_model.json`

**This file contains:** Expected locations for each bolt, bearing, and oil jet in your gearbox assembly

---

## Inspection Phase

### Option 1: Command Line (Single Image)

```bash
python inspect_assembly.py --image path/to/test_image.jpg
```

### Option 2: Command Line (Batch)

```bash
python inspect_assembly.py --image path/to/test_folder/ --batch
```

### Option 3: Python Script

```bash
python example_inference.py
```

Edit the script to change test image path.

### Option 4: Batch Processing Script

```bash
python example_batch.py
```

Processes entire directory and generates summary report.

---

## API Deployment

### 1. Start API Server

```bash
cd api
python app.py --host 0.0.0.0 --port 8000
```

### 2. Access API Documentation

Open browser: `http://localhost:8000/`

### 3. Test API (curl)

```bash
# Health check
curl http://localhost:8000/health

# Inspect image
curl -X POST http://localhost:8000/inspect \
  -F "file=@test_image.jpg" \
  -F "save_visualization=true"
```

### 4. Test API (Python)

```python
import requests

url = "http://localhost:8000/inspect"
files = {"file": open("test_image.jpg", "rb")}
params = {"save_visualization": True}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"Status: {result['status']}")
print(f"Compliance: {result['compliance_score']:.1%}")
```

---

## Example Commands

### Full Workflow Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python quick_start.py

# 3. Train detector (assumes dataset/ is ready)
cd src
python train_detector.py --dataset ../dataset --epochs 100

# 4. Build golden model
python build_golden_model.py --dataset ../dataset

# 5. Inspect test image
python inspect_assembly.py --image ../test_images/test_001.jpg

# 6. Start API server
cd ../api
python app.py
```

### Advanced Training Options

```bash
# Train with GPU
python train_detector.py \
  --dataset ../dataset \
  --model-size m \
  --epochs 200 \
  --batch 32 \
  --device cuda

# Custom golden model parameters
python build_golden_model.py \
  --dataset ../dataset \
  --eps 0.03 \
  --min-samples 3 \
  --min-occurrence 0.7
```

### Custom Inspection Parameters

```bash
# Lower tolerance for stricter matching
python inspect_assembly.py \
  --image test.jpg \
  --tolerance 0.03 \
  --confidence 0.4
```

---

## Troubleshooting

### Issue: Training shows "No images found"

**Solution:** Ensure images are in `dataset/images/` with supported extensions (.jpg, .png)

### Issue: "RuntimeError: CUDA out of memory"

**Solutions:**
- Reduce batch size: `--batch 8`
- Use smaller model: `--model-size n`
- Use CPU: `--device cpu`

### Issue: Golden model finds no expected components

**Solutions:**
- Lower `--min-occurrence` (e.g., 0.3)
- Lower `--eps` for tighter clusters (e.g., 0.03)
- Check labels are correct and consistent

### Issue: Too many false positives (missing components)

**Solutions:**
- Increase inspection tolerance: `--tolerance 0.08`
- Check golden model has correct expected locations
- Verify detector is accurate on test images

---

## Performance Tips

### For Better Accuracy
- Use 100+ training images
- Ensure consistent lighting and camera angle
- Use data augmentation (automatic in YOLO)
- Use larger model (yolov8m or yolov8l)

### For Faster Inference
- Use GPU: `--device cuda`
- Use smaller model (yolov8n)
- Reduce image size: `--imgsz 416`
- Export to ONNX for deployment

### For Production
- Set up monitoring/logging
- Use FastAPI with multiple workers
- Cache golden model in memory
- Implement retry logic for failures

---

## Next Steps

1. ✅ **Setup Complete** - If you've followed this guide
2. 🔄 **Iterate** - Adjust parameters based on results
3. 📊 **Evaluate** - Test on validation set
4. 🚀 **Deploy** - Use API for production integration
5. 🔧 **Maintain** - Retrain periodically with new data

---

## Additional Resources

- **YOLOv8 Documentation:** https://docs.ultralytics.com/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **DBSCAN Clustering:** https://scikit-learn.org/stable/modules/clustering.html#dbscan

---

**Need Help?**

1. Run `python quick_start.py` for automated checks
2. Review main README.md for detailed documentation
3. Check example scripts for usage patterns

---

*ASV-INSPECT - Making assembly inspection intelligent and automated.*
