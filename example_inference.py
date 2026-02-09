#!/usr/bin/env python3
"""
Simple inference example for ASV-INSPECT
Demonstrates how to use the system programmatically
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inspect_assembly import AssemblyInspector
from utils import format_inspection_summary


def main():
    print("\n" + "="*60)
    print("  ASV-INSPECT - Simple Inference Example")
    print("="*60 + "\n")
    
    # Configuration
    DETECTOR_PATH = "models/detector/train/weights/best.pt"
    GOLDEN_MODEL_PATH = "models/golden_model/golden_model.json"
    TEST_IMAGE = "dataset/images/test_001.jpg"  # Change this to your test image
    
    # Check if files exist
    if not Path(DETECTOR_PATH).exists():
        print(f"❌ Detector model not found: {DETECTOR_PATH}")
        print("Please train the detector first:")
        print("  cd src && python train_detector.py --dataset ../dataset")
        return
    
    if not Path(GOLDEN_MODEL_PATH).exists():
        print(f"❌ Golden model not found: {GOLDEN_MODEL_PATH}")
        print("Please build the golden model first:")
        print("  cd src && python build_golden_model.py --dataset ../dataset")
        return
    
    if not Path(TEST_IMAGE).exists():
        print(f"⚠️  Test image not found: {TEST_IMAGE}")
        print("Please update TEST_IMAGE variable with path to a valid image")
        return
    
    # Initialize inspector
    print("Loading models...")
    inspector = AssemblyInspector(
        detector_path=DETECTOR_PATH,
        golden_model_path=GOLDEN_MODEL_PATH,
        base_tolerance=0.05,
        confidence_threshold=0.25
    )
    
    print("Models loaded successfully!\n")
    
    # Perform inspection
    print(f"Inspecting image: {TEST_IMAGE}")
    report = inspector.inspect(
        image_path=TEST_IMAGE,
        output_dir="outputs",
        save_visualization=True,
        save_report=True
    )
    
    # Display results
    print(f"\n{format_inspection_summary(report)}")
    
    # Show output locations
    if 'output_image_path' in report:
        print(f"\n📷 Annotated image saved: {report['output_image_path']}")
    
    if 'report_path' in report:
        print(f"📄 JSON report saved: {report['report_path']}")
    
    # Example: Access specific data
    print("\n" + "="*60)
    print("  Programmatic Access Example")
    print("="*60 + "\n")
    
    print(f"Status: {report['status']}")
    print(f"Compliance Score: {report['compliance_score']:.1%}")
    print(f"Total Detections: {len(report['detections'])}")
    
    if report['missing_components']:
        print(f"\nMissing Components:")
        for missing in report['missing_components']:
            print(f"  - {missing['class_name']} at "
                  f"({missing['expected_x']:.3f}, {missing['expected_y']:.3f})")
    else:
        print("\n✓ All components present!")
    
    print("\n" + "="*60)
    print("  Inspection Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInspection cancelled.")
    except Exception as e:
        print(f"\n❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()
