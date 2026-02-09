"""
Visualization Module for ASV-INSPECT
Handles drawing bounding boxes, annotations, and inspection results
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from utils import get_color_for_class


class InspectionVisualizer:
    """Visualize inspection results on images"""
    
    def __init__(
        self,
        font_scale: float = 0.7,
        font_thickness: int = 3,
        box_thickness: int = 4
    ):
        """
        Initialize visualizer
        
        Args:
            font_scale: Scale for text
            font_thickness: Thickness for text
            box_thickness: Thickness for bounding boxes
        """
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.box_thickness = box_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_detection(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        class_name: str,
        confidence: float,
        color: Tuple[int, int, int],
        label_position: str = 'top'
    ) -> np.ndarray:
        """
        Draw a detected component bounding box
        
        Args:
            image: Image to draw on
            x1, y1, x2, y2: Bounding box coordinates
            class_name: Component class name
            confidence: Detection confidence
            color: (B, G, R) color tuple
            label_position: 'top' or 'bottom' for label placement
            
        Returns:
            Image with detection drawn
        """
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.box_thickness)
        
        # Prepare label
        label = f"{class_name} {confidence:.2f}"
        
        # Calculate label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness
        )
        
        # Determine label position
        if label_position == 'top':
            label_y = y1 - 10
            if label_y < label_height + 10:
                label_y = y1 + label_height + 10
        else:
            label_y = y2 + label_height + 10
            if label_y > image.shape[0]:
                label_y = y2 - 10
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, label_y - label_height - 5),
            (x1 + label_width + 10, label_y + 5),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1 + 5, label_y),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness
        )
        
        return image
    
    def draw_missing_indicator(
        self,
        image: np.ndarray,
        x_center: float,
        y_center: float,
        class_name: str,
        index: int,
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """
        Draw a small red circle with label for missing component
        
        Args:
            image: Image to draw on
            x_center, y_center: Normalized expected location
            class_name: Component class name
            index: Missing component number
            img_width, img_height: Image dimensions
            
        Returns:
            Image with missing indicator drawn
        """
        # Convert to pixel coordinates
        x_px = int(x_center * img_width)
        y_px = int(y_center * img_height)
        
        # Draw red circle
        color = (0, 0, 255)  # Red
        radius = 25
        cv2.circle(image, (x_px, y_px), radius, color, 3)
        
        # Draw X in the center
        offset = int(radius * 0.5)
        cv2.line(image, (x_px - offset, y_px - offset), (x_px + offset, y_px + offset), color, 3)
        cv2.line(image, (x_px + offset, y_px - offset), (x_px - offset, y_px + offset), color, 3)
        
        # Draw label with position info
        label = f"#{index} {class_name}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, self.font, 0.5, 2
        )
        
        # Position label to the right of circle
        label_x = x_px + radius + 5
        label_y = y_px + 5
        
        # Check if label goes off screen, if so put it on left
        if label_x + label_width > img_width - 10:
            label_x = x_px - radius - label_width - 5
        
        # Draw label background
        cv2.rectangle(
            image,
            (label_x - 2, label_y - label_height - 2),
            (label_x + label_width + 2, label_y + 2),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (label_x, label_y),
            self.font,
            0.5,
            (255, 255, 255),
            2
        )
        
        return image
    
    def draw_status_banner(
        self,
        image: np.ndarray,
        status: str,
        compliance_score: float,
        detected_count: int,
        expected_count: int
    ) -> np.ndarray:
        """
        Draw inspection status banner at the top of image
        
        Args:
            image: Image to draw on
            status: 'PASS' or 'FAIL'
            compliance_score: Score between 0 and 1
            detected_count: Number of components detected
            expected_count: Expected number of components
            
        Returns:
            Image with status banner
        """
        img_height, img_width = image.shape[:2]
        
        # Banner dimensions
        banner_height = 80
        
        # Banner color based on status
        if status == "PASS":
            banner_color = (0, 200, 0)  # Green
        else:
            banner_color = (0, 0, 200)  # Red
        
        # Draw semi-transparent banner
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (img_width, banner_height), banner_color, -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw status text
        status_text = f"STATUS: {status}"
        cv2.putText(
            image,
            status_text,
            (20, 35),
            self.font,
            1.0,
            (255, 255, 255),
            3
        )
        
        # Draw compliance score
        score_text = f"Detected: {detected_count}/{expected_count} ({compliance_score:.1%})"
        cv2.putText(
            image,
            score_text,
            (20, 65),
            self.font,
            0.7,
            (255, 255, 255),
            2
        )
        
        return image
    
    def visualize_inspection_result(
        self,
        image_path: str,
        detections: List[Dict],
        missing_components: List[Dict],
        status: str,
        compliance_score: float,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create complete visualization of inspection result
        
        Args:
            image_path: Path to input image
            detections: List of detected components
            missing_components: List of missing components
            status: 'PASS' or 'FAIL'
            compliance_score: Compliance score
            output_path: Optional path to save output image
            
        Returns:
            Annotated image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_height, img_width = image.shape[:2]
        
        # Draw detected components
        for det in detections:
            color = get_color_for_class(det['class_id'])
            self.draw_detection(
                image,
                det['x1'],
                det['y1'],
                det['x2'],
                det['y2'],
                det['class_name'],
                det.get('confidence', 1.0),
                color
            )
        
        # Draw missing component markers at their expected positions
        for i, missing in enumerate(missing_components, 1):
            if 'expected_x' in missing and 'expected_y' in missing:
                # Skip dummy positions in count-only mode
                if missing.get('count_only', False):
                    continue
                self.draw_missing_indicator(
                    image,
                    missing['expected_x'],
                    missing['expected_y'],
                    missing['class_name'],
                    i,
                    img_width,
                    img_height
                )
        
        # Draw status banner
        self.draw_status_banner(
            image,
            status,
            compliance_score,
            len(detections),
            len(detections) + len(missing_components)
        )
        
        # Save if output path provided
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
        
        return image
    
    def create_comparison_view(
        self,
        original_path: str,
        annotated_image: np.ndarray,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create side-by-side comparison of original and annotated images
        
        Args:
            original_path: Path to original image
            annotated_image: Annotated image array
            output_path: Optional path to save comparison
            
        Returns:
            Comparison image
        """
        # Load original
        original = cv2.imread(original_path)
        
        # Resize if dimensions don't match
        if original.shape != annotated_image.shape:
            annotated_image = cv2.resize(
                annotated_image,
                (original.shape[1], original.shape[0])
            )
        
        # Create side-by-side view
        comparison = np.hstack([original, annotated_image])
        
        # Add labels
        cv2.putText(
            comparison,
            "ORIGINAL",
            (20, 40),
            self.font,
            1.0,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            comparison,
            "INSPECTION RESULT",
            (original.shape[1] + 20, 40),
            self.font,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, comparison)
            print(f"Comparison image saved to: {output_path}")
        
        return comparison


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Use InspectionVisualizer class to visualize inspection results")
