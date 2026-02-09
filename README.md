# 🔍 ASV-INSPECT - Automated Assembly Verification System

An AI-powered computer vision system for automated inspection of mechanical assemblies using YOLOv8 object detection.

## 📋 Overview

ASV-INSPECT is a complete assembly inspection solution that detects missing components in mechanical assemblies. It uses deep learning (YOLOv8) to identify components and compares them against a golden model to verify assembly completeness.

### Key Features

- ✅ **Real-time Component Detection** - YOLOv8 nano model (94% mAP50)
- 🎯 **Multiple Detection Modes** - Position-based or count-based verification
- 🖥️ **Interactive Web Interface** - Streamlit-based UI for easy operation
- 📊 **Detailed Reporting** - Visual annotations and JSON reports
- ⚙️ **Adjustable Parameters** - Confidence threshold and position tolerance controls
- 🔄 **Batch Processing** - Inspect multiple assemblies efficiently

### Detected Components

- **Bolts** (12 expected per assembly)
- **Bearings** (4 expected per assembly)
- **Oil Jets** (2 expected per assembly)

**Total Expected: 18 components per assembly**

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows/Linux/macOS
- 4GB RAM minimum (8GB recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ASV-Inspect.git
   cd ASV-Inspect
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Web Interface (Recommended)**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**Option 2: Command Line**
```bash
python src/inspect_assembly.py --image path/to/image.jpg
```

## 📖 Usage

### Web Interface

1. **Upload Image** - Drag and drop or browse for assembly image
2. **Adjust Settings** (optional):
   - **Confidence Threshold** (0.01-1.0): Lower = detect more components
   - **Position Tolerance** (0.05-0.80): Higher = accept components further from expected positions
   - **Count Only Mode**: Enable to ignore positions (recommended for varying camera angles)

3. **View Results**:
   - Green boxes: Detected components
   - Red circles: Missing component locations (position-based mode)
   - Status banner: PASS/FAIL with compliance percentage

4. **Take Action**:
   - ✓ Mark as Verified
   - 🔄 Check Another Image

### Detection Modes

**Position-Based Mode** (Default disabled)
- Matches detections to expected component positions
- Shows WHERE missing components should be
- ⚠️ Requires consistent camera positioning (same angle/distance as training images)

**Count-Only Mode** (Default enabled)
- Simply counts components by type
- Works with any camera angle/position
- Can't show specific missing component locations
- ✅ Recommended for production use with varying camera setups

## 📁 Project Structure

```
ASV-Inspect/
├── app.py                      # Streamlit web interface
├── src/
│   ├── inspect_assembly.py     # Core inspection logic
│   ├── train_detector.py       # YOLO model training
│   ├── build_golden_model.py   # Golden model creation
│   ├── visualize.py            # Result visualization
│   ├── utils.py                # Utility functions
│   └── data_loader.py          # Data loading utilities
├── models/
│   ├── detector/               # Trained YOLO model
│   └── golden_model/           # Reference component positions
├── dataset/
│   ├── images/                 # Training images
│   ├── labels/                 # YOLO annotations
│   └── data.yaml               # Dataset configuration
├── outputs/
│   ├── images/                 # Annotated result images
│   └── reports/                # JSON inspection reports
├── docs/                       # Documentation
├── example_batch.py            # Batch processing example
└── example_inference.py        # Single image example
```

## 🎓 Training Your Own Model

### 1. Prepare Dataset

Place images in `dataset/images/` and YOLO labels in `dataset/labels/`

Format: `class_id x_center y_center width height`

### 2. Train Detector

```bash
python src/train_detector.py --epochs 100 --batch 16 --device cpu
```

For GPU training:
```bash
python src/train_detector.py --epochs 100 --batch 16 --device 0
```

### 3. Build Golden Model

```bash
python src/build_golden_model.py
```

Uses k-means clustering to determine expected component positions and counts.

## 📊 Model Performance

- **Model**: YOLOv8 nano
- **Parameters**: 3,011,433
- **Overall mAP50**: 94.0%
- **Per-Class Performance**:
  - Bolt: 99.4% mAP50
  - Bearing: 99.5% mAP50
  - Oil Jet: 83.1% mAP50

Training Details:
- 100 epochs
- 139 training images
- 2,454 total detections
- CPU training time: ~11.3 hours

## 🔧 Configuration

### Key Parameters

- `confidence_threshold`: Minimum detection confidence (default: 0.05)
- `base_tolerance`: Position matching tolerance (default: 0.50)
- `use_adaptive_tolerance`: Adjust tolerance based on component variance

### Adjusting for Your Use Case

**High False Positives** (detects non-existent components):
- Increase confidence threshold (0.15-0.30)

**Missing Visible Components**:
- Lower confidence threshold (0.01-0.05)
- Check lighting and image quality

**Position Matching Issues**:
- Increase position tolerance (0.50-0.70)
- Or enable Count Only Mode

## 🐛 Troubleshooting

### Red circles in wrong locations
**Cause**: Assembly positioned differently than training images  
**Solution**: Enable "Count Only Mode" or ensure consistent camera positioning

### Not detecting all visible components
**Cause**: Confidence threshold too high  
**Solution**: Lower confidence threshold slider to 0.01-0.05

### False missing components on complete assemblies
**Cause**: Position tolerance too strict  
**Solution**: Increase position tolerance or use Count Only Mode

## 📚 Documentation

Detailed documentation available in `/docs`:
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Quick Start Guide](docs/QUICKSTART_GUIDE.md)
- [Workflow Documentation](docs/WORKFLOW.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit for the web framework
- OpenCV for image processing

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ for automated quality inspection**
