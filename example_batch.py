#!/usr/bin/env python3
"""
Batch Processing Example for ASV-INSPECT
Process multiple images and generate summary report
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inspect_assembly import AssemblyInspector


def generate_batch_summary(reports, output_file):
    """Generate comprehensive batch summary"""
    
    total = len(reports)
    passed = sum(1 for r in reports if r['status'] == 'PASS')
    failed = total - passed
    
    # Calculate statistics
    avg_compliance = sum(r['compliance_score'] for r in reports) / total if total > 0 else 0
    total_missing = sum(r['missing_count'] for r in reports)
    
    # Find most common missing components
    missing_by_class = {}
    for report in reports:
        for missing in report['missing_components']:
            cls = missing['class_name']
            missing_by_class[cls] = missing_by_class.get(cls, 0) + 1
    
    # Create summary
    summary = {
        'batch_info': {
            'timestamp': datetime.now().isoformat(),
            'total_images': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0
        },
        'statistics': {
            'average_compliance': avg_compliance,
            'total_missing_components': total_missing,
            'average_missing_per_image': total_missing / total if total > 0 else 0
        },
        'missing_components_breakdown': missing_by_class,
        'detailed_results': [
            {
                'image_name': r['image_name'],
                'status': r['status'],
                'compliance_score': r['compliance_score'],
                'missing_count': r['missing_count']
            }
            for r in reports
        ]
    }
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def print_batch_summary(summary):
    """Print formatted batch summary"""
    
    info = summary['batch_info']
    stats = summary['statistics']
    
    print("\n" + "="*60)
    print("  BATCH PROCESSING SUMMARY")
    print("="*60 + "\n")
    
    print(f"Total Images Processed: {info['total_images']}")
    print(f"Passed: {info['passed']} ({info['pass_rate']:.1%})")
    print(f"Failed: {info['failed']} ({(1-info['pass_rate']):.1%})")
    print(f"\nAverage Compliance: {stats['average_compliance']:.1%}")
    print(f"Total Missing Components: {stats['total_missing_components']}")
    print(f"Average Missing per Image: {stats['average_missing_per_image']:.2f}")
    
    if summary['missing_components_breakdown']:
        print("\nMissing Components Breakdown:")
        for cls, count in sorted(summary['missing_components_breakdown'].items(), 
                                  key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count}")
    
    print("\n" + "="*60)


def main():
    print("\n" + "="*60)
    print("  ASV-INSPECT - Batch Processing Example")
    print("="*60 + "\n")
    
    # Configuration
    DETECTOR_PATH = "models/detector/train/weights/best.pt"
    GOLDEN_MODEL_PATH = "models/golden_model/golden_model.json"
    
    # Change this to your test images directory
    TEST_IMAGES_DIR = "test_images"
    
    # Check if directory exists
    if not Path(TEST_IMAGES_DIR).exists():
        print(f"⚠️  Test images directory not found: {TEST_IMAGES_DIR}")
        print("Please create directory and add test images, or update TEST_IMAGES_DIR variable")
        
        # Try using dataset images as fallback
        fallback = "dataset/images"
        if Path(fallback).exists():
            print(f"\nUsing fallback directory: {fallback}")
            TEST_IMAGES_DIR = fallback
        else:
            return
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(TEST_IMAGES_DIR).glob(f"*{ext}"))
        image_paths.extend(Path(TEST_IMAGES_DIR).glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print(f"❌ No images found in: {TEST_IMAGES_DIR}")
        return
    
    print(f"Found {len(image_paths)} images to process\n")
    
    # Initialize inspector
    print("Loading models...")
    try:
        inspector = AssemblyInspector(
            detector_path=DETECTOR_PATH,
            golden_model_path=GOLDEN_MODEL_PATH,
            base_tolerance=0.05,
            confidence_threshold=0.25
        )
        print("Models loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease ensure you have:")
        print("  1. Trained the detector: cd src && python train_detector.py")
        print("  2. Built golden model: cd src && python build_golden_model.py")
        return
    
    # Process each image
    print("="*60)
    print("Processing images...")
    print("="*60 + "\n")
    
    reports = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {image_path.name}")
        
        try:
            report = inspector.inspect(
                str(image_path),
                output_dir="outputs/batch",
                save_visualization=True,
                save_report=False  # Will save batch summary instead
            )
            reports.append(report)
            
            # Quick status
            status_icon = "✓" if report['status'] == 'PASS' else "✗"
            print(f"  {status_icon} {report['status']} - "
                  f"Compliance: {report['compliance_score']:.1%} - "
                  f"Missing: {report['missing_count']}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Generate and save summary
    if reports:
        print("\nGenerating batch summary...")
        
        summary_file = "outputs/batch/batch_summary.json"
        Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
        
        summary = generate_batch_summary(reports, summary_file)
        print_batch_summary(summary)
        
        print(f"\n📄 Batch summary saved: {summary_file}")
        print(f"📷 Annotated images saved: outputs/batch/images/")
    
    print("\n" + "="*60)
    print("  Batch Processing Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBatch processing cancelled.")
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
