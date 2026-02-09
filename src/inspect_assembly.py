"""
Assembly Inspection Module for ASV-INSPECT
Performs component detection and missing component analysis
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

from utils import (
    euclidean_distance,
    calculate_adaptive_tolerance,
    generate_timestamp,
    save_json_report,
    format_inspection_summary
)
from visualize import InspectionVisualizer
from data_loader import YOLODataLoader


class AssemblyInspector:
    """Perform assembly inspection using YOLO detector and golden model"""
    
    def __init__(
        self,
        detector_path: str,
        golden_model_path: str,
        base_tolerance: float = 0.50,
        confidence_threshold: float = 0.05,
        use_adaptive_tolerance: bool = True
    ):
        """
        Initialize assembly inspector
        
        Args:
            detector_path: Path to YOLO model weights
            golden_model_path: Path to golden model JSON
            base_tolerance: Base spatial tolerance for matching (normalized)
            confidence_threshold: Minimum confidence for detections
            use_adaptive_tolerance: Use adaptive tolerance based on variance
        """
        self.detector_path = detector_path
        self.golden_model_path = golden_model_path
        self.base_tolerance = base_tolerance
        self.confidence_threshold = confidence_threshold
        self.use_adaptive_tolerance = use_adaptive_tolerance
        
        # Load YOLO detector
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. Install with: pip install ultralytics")
        
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Detector model not found: {detector_path}")
        
        print(f"Loading YOLO detector from: {detector_path}")
        self.detector = YOLO(detector_path)
        
        # Load golden model
        self.golden_model = self._load_golden_model()
        
        # Initialize visualizer
        self.visualizer = InspectionVisualizer()
        
        print(f"Inspector initialized with {len(self.golden_model['expected_components'])} expected components")
    
    def _load_golden_model(self) -> Dict:
        """Load golden model from JSON file"""
        if not os.path.exists(self.golden_model_path):
            raise FileNotFoundError(f"Golden model not found: {self.golden_model_path}")
        
        with open(self.golden_model_path, 'r') as f:
            model = json.load(f)
        
        print(f"Loaded golden model with {len(model['expected_components'])} expected components")
        return model
    
    def detect_components(self, image_path: str) -> List[Dict]:
        """
        Run YOLO detection on image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detection dictionaries
        """
        # Run inference
        results = self.detector(image_path, conf=self.confidence_threshold, verbose=False)
        
        # Parse results
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[class_id]
                    
                    # Calculate normalized center coordinates
                    img_height, img_width = result.orig_shape
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    }
                    detections.append(detection)
        
        return detections
    
    def find_missing_components(
        self,
        detections: List[Dict],
        expected_components: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Match detections to expected positions and identify unmatched locations
        
        Args:
            detections: List of detected components
            expected_components: List of expected component locations
            
        Returns:
            (matched_components, missing_components)
        """
        missing_components = []
        matched_detections = set()
        
        # Try to match each expected component to a detection
        for expected in expected_components:
            # Calculate tolerance based on variance
            if self.use_adaptive_tolerance:
                tolerance = calculate_adaptive_tolerance(
                    expected.get('std_x', 0.05),
                    expected.get('std_y', 0.05),
                    self.base_tolerance
                )
            else:
                tolerance = self.base_tolerance
            
            # Find nearest detection of same class within tolerance
            best_match = None
            best_distance = float('inf')
            
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                
                if det['class_id'] != expected['class_id']:
                    continue
                
                distance = euclidean_distance(
                    det['x_center'],
                    det['y_center'],
                    expected['x_center'],
                    expected['y_center']
                )
                
                if distance < best_distance and distance <= tolerance:
                    best_distance = distance
                    best_match = i
            
            # If no match, this position is missing a component
            if best_match is None:
                missing = {
                    'class_id': expected['class_id'],
                    'class_name': expected['class_name'],
                    'expected_x': expected['x_center'],
                    'expected_y': expected['y_center'],
                    'expected_width': expected.get('avg_width', 0.05),
                    'expected_height': expected.get('avg_height', 0.05),
                    'tolerance_used': tolerance
                }
                missing_components.append(missing)
            else:
                matched_detections.add(best_match)
        
        # Get matched detections
        matched = [det for i, det in enumerate(detections) if i in matched_detections]
        
        return matched, missing_components
    
    def find_missing_components_count_only(
        self,
        detections: List[Dict],
        expected_components: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Count-only mode: Compare counts by type, ignore positions
        
        Args:
            detections: List of detected components
            expected_components: List of expected component locations
            
        Returns:
            (all_detections, missing_components_list)
        """
        from collections import Counter
        
        # Count detections by class
        detected_counts = Counter([d['class_id'] for d in detections])
        
        # Count expected by class
        expected_counts = Counter([e['class_id'] for e in expected_components])
        
        # Build missing list
        missing_components = []
        for class_id, expected_count in expected_counts.items():
            detected_count = detected_counts.get(class_id, 0)
            shortage = expected_count - detected_count
            
            if shortage > 0:
                # Get class name
                class_name = next((e['class_name'] for e in expected_components if e['class_id'] == class_id), 'unknown')
                # Add placeholders for missing count
                for i in range(shortage):
                    missing = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'expected_x': 0.5,  # Dummy position
                        'expected_y': 0.5,
                        'expected_width': 0.05,
                        'expected_height': 0.05,
                        'count_only': True
                    }
                    missing_components.append(missing)
        
        # All detections are "matched" in count-only mode
        return detections, missing_components
    
    def compute_compliance_score(
        self,
        expected_count: int,
        detected_count: int,
        missing_count: int
    ) -> float:
        """
        Calculate compliance score
        
        Args:
            expected_count: Number of expected components
            detected_count: Number of matched detections
            missing_count: Number of missing components
            
        Returns:
            Compliance score between 0 and 1
        """
        if expected_count == 0:
            return 1.0
        
        # Score based on matched components
        score = (expected_count - missing_count) / expected_count
        
        return max(0.0, min(1.0, score))
    
    def inspect(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_visualization: bool = True,
        save_report: bool = True
    ) -> Dict:
        """
        Perform complete inspection on an image
        
        Args:
            image_path: Path to input image
            output_dir: Directory for outputs (default: outputs/)
            save_visualization: Save annotated image
            save_report: Save JSON report
            
        Returns:
            Inspection report dictionary
        """
        print(f"\n{'='*60}")
        print(f"Inspecting: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Detect components
        print("Running detection...")
        detections = self.detect_components(image_path)
        print(f"Detected {len(detections)} components")
        
        # Find missing components
        print("Checking for missing components...")
        expected_components = self.golden_model['expected_components']
        
        # Use count-only mode if tolerance is very high (>= 500)
        if self.base_tolerance >= 500:
            print("Using COUNT-ONLY mode (ignoring positions)")
            matched, missing = self.find_missing_components_count_only(detections, expected_components)
        else:
            print("Using POSITION-BASED matching")
            matched, missing = self.find_missing_components(detections, expected_components)
        
        # Compute compliance
        expected_count = len(expected_components)
        detected_count = len(matched)
        missing_count = len(missing)
        
        # Get expected counts by class for report
        expected_counts_by_class = {}
        for exp in expected_components:
            class_name = exp['class_name']
            expected_counts_by_class[class_name] = expected_counts_by_class.get(class_name, 0) + 1
        
        # Get matched counts by class
        matched_counts_by_class = {}
        for det in matched:
            class_name = det['class_name']
            matched_counts_by_class[class_name] = matched_counts_by_class.get(class_name, 0) + 1
        
        compliance_score = self.compute_compliance_score(
            expected_count,
            detected_count,
            missing_count
        )
        
        # Determine status
        status = "PASS" if missing_count == 0 else "FAIL"
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_name': Path(image_path).name,
            'image_path': str(image_path),
            'status': status,
            'compliance_score': compliance_score,
            'expected_count': expected_count,
            'detected_count': detected_count,
            'missing_count': missing_count,
            'expected_counts_by_class': expected_counts_by_class,
            'matched_counts_by_class': matched_counts_by_class,
            'detections': detections,
            'missing_components': missing,
            'golden_model_path': self.golden_model_path,
            'detector_path': self.detector_path,
            'base_tolerance': self.base_tolerance,
            'confidence_threshold': self.confidence_threshold
        }
        
        # Print summary
        print(f"\n{format_inspection_summary(report)}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = "../outputs"
        
        output_dir = Path(output_dir)
        
        # Save visualization
        if save_visualization:
            timestamp = generate_timestamp()
            image_filename = Path(image_path).stem
            output_image_path = output_dir / "images" / f"{image_filename}_{timestamp}_annotated.jpg"
            
            self.visualizer.visualize_inspection_result(
                image_path,
                detections,
                missing,
                status,
                compliance_score,
                str(output_image_path)
            )
            
            report['output_image_path'] = str(output_image_path)
        
        # Save report
        if save_report:
            timestamp = generate_timestamp()
            image_filename = Path(image_path).stem
            report_path = output_dir / "reports" / f"{image_filename}_{timestamp}_report.json"
            save_json_report(report, str(report_path))
            
            report['report_path'] = str(report_path)
        
        return report
    
    def batch_inspect(
        self,
        image_dir: str,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Inspect multiple images from a directory
        
        Args:
            image_dir: Directory containing images to inspect
            output_dir: Directory for outputs
            
        Returns:
            List of inspection reports
        """
        image_dir = Path(image_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(image_paths)
        
        print(f"\n{'='*60}")
        print(f"BATCH INSPECTION: {len(image_paths)} images")
        print(f"{'='*60}")
        
        # Inspect each image
        reports = []
        for image_path in image_paths:
            report = self.inspect(str(image_path), output_dir)
            reports.append(report)
        
        # Summary statistics
        total = len(reports)
        
        if total == 0:
            print(f"\n{'='*60}")
            print(f"BATCH INSPECTION SUMMARY")
            print(f"{'='*60}")
            print(f"No images found in: {image_dir}")
            print(f"Supported formats: {', '.join(image_extensions)}")
            print(f"{'='*60}")
            return reports
        
        passed = sum(1 for r in reports if r['status'] == 'PASS')
        failed = total - passed
        avg_compliance = np.mean([r['compliance_score'] for r in reports])
        
        print(f"\n{'='*60}")
        print(f"BATCH INSPECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total inspected: {total}")
        print(f"Passed: {passed} ({passed/total:.1%})")
        print(f"Failed: {failed} ({failed/total:.1%})")
        print(f"Average compliance: {avg_compliance:.1%}")
        print(f"{'='*60}")
        
        return reports


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect assembly for missing components')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--detector',
        type=str,
        default='../runs/models/detector/train/weights/best.pt',
        help='Path to YOLO detector model'
    )
    parser.add_argument(
        '--golden-model',
        type=str,
        default='../models/golden_model/golden_model.json',
        help='Path to golden model JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.05,
        help='Base spatial tolerance for matching'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Minimum confidence threshold for detections'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images in directory'
    )
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = AssemblyInspector(
        detector_path=args.detector,
        golden_model_path=args.golden_model,
        base_tolerance=args.tolerance,
        confidence_threshold=args.confidence
    )
    
    # Perform inspection
    if args.batch or os.path.isdir(args.image):
        inspector.batch_inspect(args.image, args.output)
    else:
        inspector.inspect(args.image, args.output)


if __name__ == "__main__":
    main()
