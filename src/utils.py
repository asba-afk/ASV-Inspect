"""
Utility functions for ASV-INSPECT
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import json
from datetime import datetime


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        
    Returns:
        Distance as float
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compute_iou(box1: Tuple[int, int, int, int], 
                box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: (x1, y1, x2, y2) - first box
        box2: (x1, y1, x2, y2) - second box
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def generate_timestamp() -> str:
    """Generate timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def convert_to_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def save_json_report(report: Dict, output_path: str):
    """
    Save inspection report as JSON
    
    Args:
        report: Report dictionary
        output_path: Path to save JSON file
    """
    ensure_dir(os.path.dirname(output_path))
    
    # Convert numpy types to Python native types
    serializable_report = convert_to_serializable(report)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    print(f"Report saved to: {output_path}")


def load_json_report(report_path: str) -> Dict:
    """Load inspection report from JSON file"""
    with open(report_path, 'r') as f:
        report = json.load(f)
    return report


def format_inspection_summary(report: Dict) -> str:
    """
    Format inspection report as human-readable text
    
    Args:
        report: Inspection report dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ASV-INSPECT - ASSEMBLY INSPECTION REPORT")
    lines.append("=" * 60)
    lines.append(f"Timestamp: {report.get('timestamp', 'N/A')}")
    lines.append(f"Image: {report.get('image_name', 'N/A')}")
    lines.append(f"Status: {report.get('status', 'N/A')}")
    lines.append(f"Compliance Score: {report.get('compliance_score', 0):.1%}")
    lines.append("")
    
    lines.append(f"Expected Components: {report.get('expected_count', 0)}")
    lines.append(f"Detected Components: {report.get('detected_count', 0)}")
    lines.append(f"Missing Components: {report.get('missing_count', 0)}")
    lines.append("")
    
    if report.get('missing_components'):
        lines.append("Missing Components Details:")
        
        # Use the counts from report
        expected_by_class = report.get('expected_counts_by_class', {})
        matched_by_class = report.get('matched_counts_by_class', {})
        
        # Group missing by type
        from collections import Counter
        missing_by_type = Counter([m['class_name'] for m in report['missing_components']])
        
        for class_name, missing_count in missing_by_type.items():
            expected = expected_by_class.get(class_name, 0)
            detected = matched_by_class.get(class_name, 0)
            lines.append(f"  - {missing_count} {class_name}(s): "
                        f"detected {detected}/{expected}")
    else:
        lines.append("No missing components detected ✓")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """
    Generate a consistent color for each class ID
    
    Args:
        class_id: Integer class ID
        
    Returns:
        (B, G, R) color tuple for OpenCV
    """
    # Predefined colors for better visibility
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
    ]
    
    return colors[class_id % len(colors)]


def non_max_suppression(
    boxes: List[Tuple],
    scores: List[float],
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    
    Args:
        boxes: List of (x1, y1, x2, y2) tuples
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by scores
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_indices = indices[1:]
        
        ious = [compute_iou(current_box, boxes[i]) for i in remaining_indices]
        
        # Keep boxes with IoU below threshold
        indices = remaining_indices[np.array(ious) < iou_threshold]
    
    return keep


def calculate_adaptive_tolerance(
    std_x: float,
    std_y: float,
    base_tolerance: float = 0.05,
    std_multiplier: float = 2.0
) -> float:
    """
    Calculate adaptive spatial tolerance based on positional variance
    
    Args:
        std_x: Standard deviation in x direction
        std_y: Standard deviation in y direction
        base_tolerance: Base tolerance value
        std_multiplier: Multiplier for standard deviation
        
    Returns:
        Adaptive tolerance value
    """
    # Use larger of the two standard deviations
    max_std = max(std_x, std_y)
    
    # Adaptive tolerance is base + multiple of std
    adaptive_tolerance = base_tolerance + (std_multiplier * max_std)
    
    return adaptive_tolerance


def validate_dataset_structure(dataset_path: str) -> Tuple[bool, List[str]]:
    """
    Validate that dataset has proper YOLO structure
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        (is_valid, error_messages)
    """
    dataset_path = Path(dataset_path)
    errors = []
    
    # Check if dataset directory exists
    if not dataset_path.exists():
        errors.append(f"Dataset directory not found: {dataset_path}")
        return False, errors
    
    # Check for required subdirectories
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        errors.append(f"Images directory not found: {images_dir}")
    
    if not labels_dir.exists():
        errors.append(f"Labels directory not found: {labels_dir}")
    
    # Check for obj.names file
    names_file = dataset_path / "obj.names"
    if not names_file.exists():
        errors.append(f"obj.names file not found: {names_file}")
    
    # Check if there are any images
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if len(image_files) == 0:
            errors.append("No image files found in images directory")
    
    # Check if there are any labels
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        if len(label_files) == 0:
            errors.append("No label files found in labels directory")
    
    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test distance calculation
    dist = euclidean_distance(0.5, 0.5, 0.6, 0.6)
    print(f"Distance: {dist:.4f}")
    
    # Test IoU calculation
    box1 = (10, 10, 50, 50)
    box2 = (30, 30, 70, 70)
    iou = compute_iou(box1, box2)
    print(f"IoU: {iou:.4f}")
    
    # Test adaptive tolerance
    tolerance = calculate_adaptive_tolerance(0.01, 0.015)
    print(f"Adaptive tolerance: {tolerance:.4f}")
    
    print("\nUtility functions working correctly!")
