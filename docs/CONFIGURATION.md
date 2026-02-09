# ASV-INSPECT Configuration Guide

## Overview

This document explains all configuration parameters for ASV-INSPECT system.

---

## 1. YOLO Detector Training

### Model Size Selection

| Size | Parameters | Speed | Accuracy | Use Case |
|------|------------|-------|----------|----------|
| n    | 3.2M       | ⚡⚡⚡ | ⭐⭐    | CPU, Real-time |
| s    | 11.2M      | ⚡⚡  | ⭐⭐⭐  | Balanced |
| m    | 25.9M      | ⚡    | ⭐⭐⭐⭐ | GPU, Better accuracy |
| l    | 43.7M      | 🐌   | ⭐⭐⭐⭐⭐ | GPU, Best accuracy |
| x    | 68.2M      | 🐌🐌 | ⭐⭐⭐⭐⭐ | Research, Maximum accuracy |

**Recommendation:** Start with `n` for CPU, `m` for GPU

### Training Parameters

```bash
--epochs: 50-200 (default: 100)
```
- More epochs = better learning, but risk of overfitting
- Use early stopping (automatic) for optimal results
- 50-100 epochs usually sufficient for small datasets

```bash
--batch: 8-32 (default: 16)
```
- Larger batch = faster training, more GPU memory
- Reduce if getting CUDA out of memory errors
- CPU: 8-16, GPU: 16-32

```bash
--imgsz: 320-1280 (default: 640)
```
- Higher resolution = more details, slower training
- Must be multiple of 32
- Common: 416, 640, 1280

```bash
--device: 'cpu' or 'cuda'
```
- Use GPU for 5-10x faster training
- Automatic GPU detection

---

## 2. Golden Model Building

### Clustering Parameters

```bash
--eps: 0.01-0.15 (default: 0.05)
```
**What it does:** Maximum distance for points to be in same cluster

**Interpretation:**
- 0.05 = 5% of image dimension
- For 640px image: 0.05 = 32 pixels

**Guidelines:**
- **Tight assemblies** (precise positioning): 0.02-0.03
- **Normal assemblies** (moderate variation): 0.04-0.06
- **Loose assemblies** (high variation): 0.07-0.10

**Example scenarios:**
- Circuit board (very precise): `--eps 0.02`
- Car engine (moderate precision): `--eps 0.05`
- Furniture assembly (less precise): `--eps 0.08`

```bash
--min-samples: 1-5 (default: 2)
```
**What it does:** Minimum detections required to form a cluster

**Guidelines:**
- Small dataset (<30 images): `--min-samples 1`
- Medium dataset (30-100 images): `--min-samples 2`
- Large dataset (>100 images): `--min-samples 3`

**Warning:** Setting to 1 may create clusters from outliers

```bash
--min-occurrence: 0.0-1.0 (default: 0.5)
```
**What it does:** Minimum fraction of images where component must appear

**Guidelines:**
- **Always present components:** 0.7-0.9
- **Usually present components:** 0.5-0.7
- **Sometimes present (variations):** 0.3-0.5

**Example:**
- 100 training images, component appears in 80 → occurrence ratio = 0.8
- If `--min-occurrence 0.5` → included ✓
- If `--min-occurrence 0.9` → excluded ✗

---

## 3. Inspection Parameters

### Tolerance Settings

```bash
--tolerance: 0.01-0.20 (default: 0.05)
```
**What it does:** Maximum distance for detecting matching components

**Guidelines:**
- **Strict inspection:** 0.02-0.03 (catching small deviations)
- **Normal inspection:** 0.04-0.06 (standard tolerance)
- **Lenient inspection:** 0.07-0.10 (allowing more variation)

**Adaptive tolerance:**
- System automatically increases tolerance based on training variance
- Formula: `adaptive = base_tolerance + (2 × std_deviation)`
- Disable with `use_adaptive_tolerance=False`

### Confidence Threshold

```bash
--confidence: 0.1-0.9 (default: 0.25)
```
**What it does:** Minimum YOLO detection confidence

**Guidelines:**
- **High precision needed:** 0.4-0.6 (fewer false positives)
- **Balanced:** 0.25-0.35 (recommended)
- **High recall needed:** 0.1-0.2 (catch all possible detections)

**Trade-off:**
- Higher → fewer false detections, may miss real components
- Lower → detect more, but more false positives

---

## 4. API Configuration

### Environment Variables

Create `.env` file:

```bash
# Model Paths
DETECTOR_PATH=models/detector/train/weights/best.pt
GOLDEN_MODEL_PATH=models/golden_model/golden_model.json

# Detection Parameters
BASE_TOLERANCE=0.05
CONFIDENCE_THRESHOLD=0.25

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Server Options

```bash
--host: IP address to bind (default: 0.0.0.0)
```
- `0.0.0.0` = accessible from network
- `127.0.0.1` = localhost only

```bash
--port: Port number (default: 8000)
```
- Choose unused port
- Common: 8000, 8080, 5000

```bash
--reload: Enable auto-reload (development only)
```
- Automatically restart on code changes
- Don't use in production

---

## 5. Parameter Selection Guide

### Scenario 1: High-Precision Manufacturing

```bash
# Training
python train_detector.py \
  --model-size m \
  --epochs 150 \
  --device cuda

# Golden model
python build_golden_model.py \
  --eps 0.02 \
  --min-samples 3 \
  --min-occurrence 0.8

# Inspection
python inspect_assembly.py \
  --tolerance 0.03 \
  --confidence 0.4
```

**Use case:** Circuit boards, aerospace components

### Scenario 2: Standard Manufacturing

```bash
# Training
python train_detector.py \
  --model-size s \
  --epochs 100 \
  --device cpu

# Golden model
python build_golden_model.py \
  --eps 0.05 \
  --min-samples 2 \
  --min-occurrence 0.5

# Inspection
python inspect_assembly.py \
  --tolerance 0.05 \
  --confidence 0.25
```

**Use case:** Automotive, consumer electronics

### Scenario 3: Variable Assemblies

```bash
# Training
python train_detector.py \
  --model-size n \
  --epochs 80 \
  --device cpu

# Golden model
python build_golden_model.py \
  --eps 0.08 \
  --min-samples 1 \
  --min-occurrence 0.3

# Inspection
python inspect_assembly.py \
  --tolerance 0.08 \
  --confidence 0.2
```

**Use case:** Furniture, modular assemblies

---

## 6. Tuning Tips

### If detector misses components:
1. Lower `--confidence` (e.g., 0.15)
2. Train longer (`--epochs 150`)
3. Use larger model (`--model-size m`)
4. Add more training data

### If too many false positives:
1. Raise `--confidence` (e.g., 0.35)
2. Use larger model for better discrimination
3. Ensure training data quality

### If golden model has wrong expected locations:
1. Adjust `--eps` (try ±0.02)
2. Check training labels are consistent
3. Visualize clusters (add debug output)

### If inspection reports too many missing:
1. Increase `--tolerance` (e.g., 0.07)
2. Lower `--min-occurrence` in golden model
3. Check if detector is accurate

### If inspection misses real problems:
1. Decrease `--tolerance` (e.g., 0.03)
2. Increase `--min-occurrence` in golden model
3. Ensure golden model represents "correct" assemblies

---

## 7. Performance Optimization

### Training Speed
- Use GPU: ~10x faster
- Reduce `--imgsz`: ~2x faster
- Use smaller model: ~3x faster
- Multi-GPU: Use `--device 0,1`

### Inference Speed
- Use YOLOv8n: ~5x faster than YOLOv8x
- Export to ONNX: ~2x faster
- Use GPU: ~10x faster
- Batch processing: More efficient

### Memory Usage
- Reduce batch size if OOM errors
- Use smaller model
- Process images sequentially instead of batch

---

## 8. Quality Metrics

### Detector Performance
- **mAP50** (mean Average Precision): >0.90 excellent
- **mAP50-95**: >0.70 excellent
- **Precision**: >0.95 for production
- **Recall**: >0.95 for production

Check after training in `models/detector/train/results.png`

### Inspection Performance
- **Compliance score**: Percentage of components present
- **False positive rate**: Missing alarms for present components
- **False negative rate**: Missed defects
- **Pass rate**: Percentage of assemblies passing inspection

Monitor over time and adjust parameters if needed.

---

## 9. Example Configurations

### config_high_precision.json
```json
{
  "training": {
    "model_size": "m",
    "epochs": 150,
    "batch": 32,
    "imgsz": 640
  },
  "golden_model": {
    "eps": 0.02,
    "min_samples": 3,
    "min_occurrence": 0.8
  },
  "inspection": {
    "base_tolerance": 0.03,
    "confidence_threshold": 0.4,
    "use_adaptive_tolerance": true
  }
}
```

### config_fast_cpu.json
```json
{
  "training": {
    "model_size": "n",
    "epochs": 80,
    "batch": 8,
    "imgsz": 416
  },
  "golden_model": {
    "eps": 0.05,
    "min_samples": 2,
    "min_occurrence": 0.5
  },
  "inspection": {
    "base_tolerance": 0.05,
    "confidence_threshold": 0.25,
    "use_adaptive_tolerance": true
  }
}
```

---

**Summary:** Start with default values, then adjust based on your specific needs and the quality of results you observe. Monitor performance metrics and iterate!
