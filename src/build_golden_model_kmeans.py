"""
Build golden model using k-means clustering.
Better for known component counts than DBSCAN.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from utils import convert_to_serializable


def build_golden_model_kmeans(
    dataset_path: str,
    expected_counts: dict = None,
    output_path: str = '../models/golden_model/golden_model.json'
):
    """
    Build golden model using k-means clustering.
    
    Args:
        dataset_path: Path to dataset directory
        expected_counts: Dict mapping class_id to expected count (e.g., {0: 12, 1: 4, 2: 2})
        output_path: Where to save the model
    """
    # Default expected counts based on data analysis
    if expected_counts is None:
        expected_counts = {
            0: 12,  # bolts
            1: 4,   # bearings
            2: 2    # oil jets
        }
    
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / 'labels'
    
    print("Building golden model from dataset...")
    
    # Load all labels
    all_detections = defaultdict(list)
    label_files = list(labels_dir.glob('*.txt'))
    
    print(f"Found {len(label_files)} labeled images\n")
    
    # Collect all detections by class
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                all_detections[class_id].append({
                    'x': x_center,
                    'y': y_center,
                    'width': width,
                    'height': height
                })
    
    # Build golden model
    class_names = {0: 'bolt', 1: 'bearing', 2: 'oil jet'}
    expected_components = []
    
    for class_id in sorted(all_detections.keys()):
        detections = all_detections[class_id]
        class_name = class_names.get(class_id, f'class_{class_id}')
        n_expected = expected_counts.get(class_id, len(detections) // len(label_files))
        
        print(f"Processing class {class_id}: {class_name}")
        print(f"  Total detections: {len(detections)}")
        print(f"  Expected components per image: {n_expected}")
        
        # Convert to numpy array
        positions = np.array([[d['x'], d['y']] for d in detections])
        
        # Apply k-means clustering
        if len(positions) < n_expected:
            print(f"  ⚠ Warning: Only {len(positions)} detections, expected {n_expected}")
            n_clusters = len(positions)
        else:
            n_clusters = n_expected
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(positions)
        
        # Get cluster centers and statistics
        for i, center in enumerate(kmeans.cluster_centers_):
            # Find all points in this cluster
            cluster_mask = kmeans.labels_ == i
            cluster_points = positions[cluster_mask]
            cluster_detections = [d for d, mask in zip(detections, cluster_mask) if mask]
            
            # Calculate statistics
            x_center = float(center[0])
            y_center = float(center[1])
            std_x = float(np.std(cluster_points[:, 0]))
            std_y = float(np.std(cluster_points[:, 1]))
            avg_width = float(np.mean([d['width'] for d in cluster_detections]))
            avg_height = float(np.mean([d['height'] for d in cluster_detections]))
            
            component = {
                'class_id': int(class_id),
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'std_x': std_x,
                'std_y': std_y,
                'avg_width': avg_width,
                'avg_height': avg_height,
                'cluster_size': int(len(cluster_points)),
                'cluster_id': i
            }
            
            expected_components.append(component)
            print(f"    ✓ Component {i+1} at ({x_center:.3f}, {y_center:.3f}) "
                  f"- {len(cluster_points)} detections, std=({std_x:.3f}, {std_y:.3f})")
        
        print()
    
    # Create golden model
    golden_model = {
        'metadata': {
            'total_training_images': len(label_files),
            'clustering_method': 'kmeans',
            'expected_counts': expected_counts,
            'class_names': class_names
        },
        'expected_components': expected_components
    }
    
    # Convert numpy types to native Python types
    golden_model = convert_to_serializable(golden_model)
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(golden_model, f, indent=2)
    
    print("=" * 60)
    print(f"Golden model built successfully!")
    print(f"Expected components: {len(expected_components)}")
    print("=" * 60)
    print(f"\nGolden model saved to: {output_path}")
    
    # Print summary by class
    print("\nComponent breakdown:")
    for class_id in sorted(expected_counts.keys()):
        count = sum(1 for c in expected_components if c['class_id'] == class_id)
        print(f"  {class_names[class_id]}: {count}")
    
    return golden_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build golden model using k-means clustering')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--bolts', type=int, default=12,
                        help='Expected number of bolts per assembly (default: 12)')
    parser.add_argument('--bearings', type=int, default=4,
                        help='Expected number of bearings per assembly (default: 4)')
    parser.add_argument('--oil-jets', type=int, default=2,
                        help='Expected number of oil jets per assembly (default: 2)')
    parser.add_argument('--output', type=str, default='../models/golden_model/golden_model.json',
                        help='Output path for golden model')
    
    args = parser.parse_args()
    
    expected_counts = {
        0: args.bolts,
        1: args.bearings,
        2: args.oil_jets
    }
    
    build_golden_model_kmeans(
        dataset_path=args.dataset,
        expected_counts=expected_counts,
        output_path=args.output
    )
