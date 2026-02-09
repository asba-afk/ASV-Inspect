"""
Data Loader Module for ASV-INSPECT
Handles loading and parsing YOLO format dataset
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class YOLODataLoader:
    """Load and parse YOLO format annotations"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the data loader
        
        Args:
            dataset_path: Path to dataset directory containing images/ and labels/
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"
        self.class_names = self._load_class_names()
        
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from obj.names file"""
        names_file = self.dataset_path / "obj.names"
        class_names = {}
        
        if names_file.exists():
            with open(names_file, 'r') as f:
                for idx, line in enumerate(f):
                    class_names[idx] = line.strip()
        else:
            print(f"Warning: obj.names not found at {names_file}")
            
        return class_names
    
    def load_label_file(self, label_path: str) -> List[Dict]:
        """
        Load a single YOLO label file
        
        Args:
            label_path: Path to .txt label file
            
        Returns:
            List of detection dictionaries with keys:
                - class_id: int
                - class_name: str
                - x_center: float (normalized)
                - y_center: float (normalized)
                - width: float (normalized)
                - height: float (normalized)
        """
        detections = []
        
        if not os.path.exists(label_path):
            return detections
            
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                detection = {
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, f"class_{class_id}"),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                }
                detections.append(detection)
                
        return detections
    
    def load_all_labels(self) -> Dict[str, List[Dict]]:
        """
        Load all label files from the dataset
        
        Returns:
            Dictionary mapping image filename to list of detections
        """
        all_labels = {}
        
        if not self.labels_path.exists():
            print(f"Warning: Labels directory not found at {self.labels_path}")
            return all_labels
            
        for label_file in self.labels_path.glob("*.txt"):
            detections = self.load_label_file(str(label_file))
            # Use stem to get filename without extension
            image_name = label_file.stem
            all_labels[image_name] = detections
            
        return all_labels
    
    def denormalize_coordinates(
        self, 
        x_center: float, 
        y_center: float, 
        width: float, 
        height: float,
        img_width: int,
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Convert normalized YOLO coordinates to pixel coordinates
        
        Args:
            x_center, y_center, width, height: Normalized coordinates (0-1)
            img_width, img_height: Image dimensions in pixels
            
        Returns:
            (x1, y1, x2, y2) in pixel coordinates (top-left, bottom-right)
        """
        # Convert center coordinates to pixels
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Convert to corner coordinates
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        return x1, y1, x2, y2
    
    def normalize_coordinates(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert pixel coordinates to normalized YOLO format
        
        Args:
            x1, y1, x2, y2: Pixel coordinates (top-left, bottom-right)
            img_width, img_height: Image dimensions in pixels
            
        Returns:
            (x_center, y_center, width, height) normalized (0-1)
        """
        width_px = x2 - x1
        height_px = y2 - y1
        x_center_px = x1 + width_px / 2
        y_center_px = y1 + height_px / 2
        
        x_center = x_center_px / img_width
        y_center = y_center_px / img_height
        width = width_px / img_width
        height = height_px / img_height
        
        return x_center, y_center, width, height
    
    def get_image_paths(self) -> List[Path]:
        """Get all image paths from the images directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        if self.images_path.exists():
            for ext in image_extensions:
                image_paths.extend(self.images_path.glob(f"*{ext}"))
                image_paths.extend(self.images_path.glob(f"*{ext.upper()}"))
                
        return sorted(image_paths)
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        all_labels = self.load_all_labels()
        
        total_images = len(list(self.get_image_paths()))
        total_labels = len(all_labels)
        total_detections = sum(len(dets) for dets in all_labels.values())
        
        # Count detections per class
        class_counts = {}
        for detections in all_labels.values():
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_images': total_images,
            'total_labeled_images': total_labels,
            'total_detections': total_detections,
            'class_counts': class_counts,
            'class_names': self.class_names
        }


if __name__ == "__main__":
    # Test the data loader
    loader = YOLODataLoader("../dataset")
    stats = loader.get_dataset_stats()
    
    print("Dataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Labeled images: {stats['total_labeled_images']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"\nClass counts:")
    for class_name, count in stats['class_counts'].items():
        print(f"  {class_name}: {count}")
