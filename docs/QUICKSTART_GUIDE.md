# Quick Start Guide - Your Gearbox Assembly Inspection

## ✅ Current Status

Your dataset is ready!
- **148 images** of gearbox assemblies
- **157 label files** (YOLO format)
- **3 component classes**: bolt, bearing, oil jet

---

## 🎯 Step-by-Step Training & Inspection

### STEP 1: Install Dependencies (If Not Done)

```bash
# Navigate to project directory
cd C:\Users\Asba\Desktop\Projects\ASV-Inspect

# Install requirements
pip install -r requirements.txt
```

**Expected time:** 2-5 minutes

---

### STEP 2: Train the Detector

This trains the AI to recognize bolts, bearings, and oil jets in your gearbox images.

```bash
cd src
python train_detector.py --dataset ../dataset --epochs 100 --model-size s --batch 16 --device cpu
```

**What this does:**
- Trains YOLOv8 small model on your 148 images
- Learns to detect bolts, bearings, and oil jets
- Saves best model to `models/detector/train/weights/best.pt`

**Expected time:** 20-45 minutes (CPU)

**Options to adjust:**
- Use `--device cuda` if you have an NVIDIA GPU (much faster!)
- Use `--batch 8` if you get memory errors
- Use `--epochs 150` for potentially better accuracy

**Output to look for:**
- Training progress bar
- Metrics: precision, recall, mAP
- Final model saved message

---

### STEP 3: Build Golden Model

This creates a "reference" of where components should be located in a correct assembly.

```bash
# Stay in src directory
python build_golden_model.py --dataset ../dataset
```

**What this does:**
- Analyzes all 148 training images
- Finds common positions for each component type
- Creates statistical model of expected locations
- Saves to `models/golden_model/golden_model.json`

**Expected time:** 5-10 seconds

**Output to look for:**
- Message: "Golden model saved successfully"
- Number of expected components found

---

### STEP 4: Test Inspection

Now you can inspect new gearbox images to check for missing components!

**Option A: Inspect a single test image**

```bash
# Stay in src directory
python inspect_assembly.py --image PATH_TO_YOUR_TEST_IMAGE.jpg
```

Replace `PATH_TO_YOUR_TEST_IMAGE.jpg` with actual path, for example:
```bash
python inspect_assembly.py --image C:\Users\Asba\Desktop\test_gearbox.jpg
```

**Option B: Inspect a batch of images**

```bash
python inspect_assembly.py --image C:\Users\Asba\Desktop\test_images\ --batch
```

**What you'll get:**
- Console output showing PASS/FAIL status
- Annotated image in `outputs/images/` with:
  - Green boxes around detected components
  - Red dashed boxes with X for missing components
  - PASS/FAIL banner at top
- JSON report in `outputs/reports/` with detailed results

---

### STEP 5: (Optional) Start Web API

For production use, you can run the inspection system as a web service.

```bash
cd ../api
python app.py --host 0.0.0.0 --port 8000
```

Then open browser: `http://localhost:8000/`

You'll see interactive API documentation where you can upload images for inspection.

---

## 📋 Complete Command Sequence

Here's everything in one place:

```bash
# 1. Navigate to project
cd C:\Users\Asba\Desktop\Projects\ASV-Inspect

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Train detector
cd src
python train_detector.py --dataset ../dataset --epochs 100 --model-size s --batch 16 --device cpu

# 4. Build golden model
python build_golden_model.py --dataset ../dataset

# 5. Test on an image
python inspect_assembly.py --image YOUR_TEST_IMAGE_PATH.jpg

# 6. (Optional) Start API
cd ../api
python app.py
```

---

## 🎨 Understanding the Results

### PASS Status
- ✅ All expected components detected
- Green banner at top
- Compliance: 100%
- All boxes are colored (detected components)

### FAIL Status
- ❌ One or more components missing
- Red banner at top
- Compliance: < 100%
- Red dashed boxes with X show missing components
- Console shows which components are missing and where

### JSON Report Example
```json
{
  "status": "FAIL",
  "compliance_score": 0.85,
  "expected_count": 20,
  "detected_count": 17,
  "missing_count": 3,
  "missing_components": [
    {
      "class_name": "bolt",
      "expected_x": 0.45,
      "expected_y": 0.67
    }
  ]
}
```

---

## ⚙️ Adjusting Parameters

### If you're getting too many false positives (says missing when parts are there):

```bash
python inspect_assembly.py --image test.jpg --tolerance 0.08
```

Increase tolerance from 0.05 (default) to 0.08 or 0.10

### If detector is not finding components:

```bash
python inspect_assembly.py --image test.jpg --confidence 0.15
```

Lower confidence threshold from 0.25 (default) to 0.15 or 0.20

### If golden model has too few expected components:

```bash
python build_golden_model.py --dataset ../dataset --min-occurrence 0.3
```

Lower min-occurrence from 0.5 (default) to 0.3 (component must be in 30% of images)

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'ultralytics'"
**Solution:** Run `pip install -r requirements.txt`

### Error: "CUDA out of memory"
**Solution:** Use `--device cpu` or reduce `--batch 8`

### Error: "No images found in dataset"
**Solution:** Check that images are in `dataset/images/` with .JPG extension

### Training is very slow
**Solution:** 
- Use GPU if available: `--device cuda`
- Use smaller model: `--model-size n`
- Reduce epochs: `--epochs 50`

### Too many missing components detected
**Solution:**
- Increase tolerance: `--tolerance 0.08`
- Lower confidence: `--confidence 0.2`
- Check golden model has correct expected locations

---

## 📊 Next Steps After Training

1. **Validate accuracy**: Test on 10-20 images you didn't train on
2. **Adjust parameters**: Fine-tune tolerance and confidence based on results
3. **Deploy**: Use API for production integration
4. **Monitor**: Track false positives/negatives over time
5. **Retrain**: Add new images and retrain periodically

---

## 💡 Tips for Best Results

1. **Training images should be "correct" assemblies** - all components present
2. **Test images can be correct or faulty** - system will detect what's missing
3. **Consistent lighting and camera angle** helps accuracy
4. **More training data = better accuracy** (you have 148, which is good!)
5. **GPU training is 10-20x faster** if you have NVIDIA GPU

---

## 🆘 Need Help?

Run this to check your setup:
```bash
python quick_start.py
```

This will verify:
- Project structure
- Dataset completeness
- Dependencies installed
- Models trained
- Next steps to take

---

**You're ready to start training! Begin with STEP 1 above. Good luck! 🚀**
