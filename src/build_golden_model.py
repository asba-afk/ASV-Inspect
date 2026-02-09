"""
Golden Model Builder for ASV-INSPECT
Statistically learns expected component locations from multiple training images
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

from data_loader import YOLODataLoader
from utils import convert_to_serializable


class GoldenModelBuilder:
    """Build statistical golden model from training dataset"""
    
    def __init__(
        self, 
        dataset_path: str,
        eps: float = 0.05,
        min_samples: int = 2,
        min_occurrence_ratio: float = 0.5
    ):
        """
        Initialize the golden model builder
        
        Args:
            dataset_path: Path to dataset directory
            eps: Maximum distance between samples for DBSCAN clustering (normalized)
            min_samples: Minimum samples in a cluster to be considered valid
            min_occurrence_ratio: Minimum ratio of images where component must appear (0-1)
        """
        self.dataset_path = dataset_path
        self.data_loader = YOLODataLoader(dataset_path)
        self.eps = eps
        self.min_samples = min_samples
        self.min_occurrence_ratio = min_occurrence_ratio
        
    def collect_detections_by_class(self) -> Dict[int, List[Dict]]:
        """
        Collect all detections grouped by class
        
        Returns:
            Dictionary mapping class_id to list of detection dictionaries
        """
        all_labels = self.data_loader.load_all_labels()
        detections_by_class = defaultdict(list)
        
        for image_name, detections in all_labels.items():
            for det in detections:
                detections_by_class[det['class_id']].append({
                    'x_center': det['x_center'],
                    'y_center': det['y_center'],
                    'width': det['width'],
                    'height': det['height'],
                    'class_name': det['class_name'],
                    'image_name': image_name
                })
                
        return dict(detections_by_class)
    
    def cluster_positions(
        self, 
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Cluster component positions using DBSCAN
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of cluster centers with metadata
        """
        if len(detections) == 0:
            return []
        
        # Extract positions
        positions = np.array([[d['x_center'], d['y_center']] for d in detections])
        
        # Perform clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(positions)
        
        # Compute cluster centers
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            # Get all points in this cluster
            mask = labels == label
            cluster_positions = positions[mask]
            cluster_detections = [d for i, d in enumerate(detections) if mask[i]]
            
            # Compute statistics
            center = cluster_positions.mean(axis=0)
            std = cluster_positions.std(axis=0)
            
            # Average size
            avg_width = np.mean([d['width'] for d in cluster_detections])
            avg_height = np.mean([d['height'] for d in cluster_detections])
            
            cluster_info = {
                'x_center': float(center[0]),
                'y_center': float(center[1]),
                'std_x': float(std[0]),
                'std_y': float(std[1]),
                'avg_width': float(avg_width),
                'avg_height': float(avg_height),
                'count': int(mask.sum()),
                'class_name': cluster_detections[0]['class_name']
            }
            clusters.append(cluster_info)
            
        return clusters
    
    def build_model(self) -> Dict:
        """
        Build the complete golden model
        
        Returns:
            Dictionary containing expected component locations and metadata
        """
        print("Building golden model from dataset...")
        
        # Get dataset statistics
        stats = self.data_loader.get_dataset_stats()
        total_images = stats['total_labeled_images']
        
        if total_images == 0:
            raise ValueError("No labeled images found in dataset")
        
        print(f"Found {total_images} labeled images")
        
        # Collect detections by class
        detections_by_class = self.collect_detections_by_class()
        
        golden_model = {
            'metadata': {
                'total_training_images': total_images,
                'eps': self.eps,
                'min_samples': self.min_samples,
                'min_occurrence_ratio': self.min_occurrence_ratio,
                'class_names': self.data_loader.class_names
            },
            'expected_components': []
        }
        
        # Process each class
        for class_id, detections in detections_by_class.items():
            print(f"\nProcessing class {class_id}: {detections[0]['class_name']}")
            print(f"  Total detections: {len(detections)}")
            
            # Cluster positions
            clusters = self.cluster_positions(detections)
            print(f"  Found {len(clusters)} component locations")
            
            # Filter clusters by occurrence ratio
            min_occurrences = int(total_images * self.min_occurrence_ratio)
            
            for cluster in clusters:
                occurrence_ratio = cluster['count'] / total_images
                
                if cluster['count'] >= min_occurrences:
                    component = {
                        'class_id': class_id,
                        'class_name': cluster['class_name'],
                        'x_center': cluster['x_center'],
                        'y_center': cluster['y_center'],
                        'std_x': cluster['std_x'],
                        'std_y': cluster['std_y'],
                        'avg_width': cluster['avg_width'],
                        'avg_height': cluster['avg_height'],
                        'occurrence_count': cluster['count'],
                        'occurrence_ratio': float(occurrence_ratio)
                    }
                    golden_model['expected_components'].append(component)
                    print(f"    ✓ Location at ({cluster['x_center']:.3f}, {cluster['y_center']:.3f}) "
                          f"- occurs in {cluster['count']}/{total_images} images ({occurrence_ratio:.1%})")
                else:
                    print(f"    ✗ Location at ({cluster['x_center']:.3f}, {cluster['y_center']:.3f}) "
                          f"- only {cluster['count']}/{total_images} occurrences (below threshold)")
        
        print(f"\n{'='*60}")
        print(f"Golden model built successfully!")
        print(f"Expected components: {len(golden_model['expected_components'])}")
        print(f"{'='*60}")
        
        return golden_model
    
    def save_model(self, model: Dict, output_path: str):
        """Save golden model to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types
        serializable_model = convert_to_serializable(model)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_model, f, indent=2)
            
        print(f"\nGolden model saved to: {output_path}")
    
    def load_model(self, model_path: str) -> Dict:
        """Load golden model from JSON file"""
        with open(model_path, 'r') as f:
            model = json.load(f)
        return model


def main():
    """Main function to build and save golden model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build golden statistical model')
    parser.add_argument(
        '--dataset',
        type=str,
        default='../dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../models/golden_model/golden_model.json',
        help='Output path for golden model'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.05,
        help='DBSCAN eps parameter (spatial tolerance)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=2,
        help='Minimum samples for DBSCAN cluster'
    )
    parser.add_argument(
        '--min-occurrence',
        type=float,
        default=0.5,
        help='Minimum occurrence ratio (0-1) for components'
    )
    
    args = parser.parse_args()
    
    # Build model
    builder = GoldenModelBuilder(
        dataset_path=args.dataset,
        eps=args.eps,
        min_samples=args.min_samples,
        min_occurrence_ratio=args.min_occurrence
    )
    
    model = builder.build_model()
    builder.save_model(model, args.output)


if __name__ == "__main__":
    main()
