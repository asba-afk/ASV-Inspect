"""
YOLO Detector Training Module for ASV-INSPECT
Handles training YOLOv8 model on custom dataset
"""

import os
from pathlib import Path
import yaml
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

from data_loader import YOLODataLoader
from utils import validate_dataset_structure


class YOLOTrainer:
    """Train YOLO detector on custom assembly dataset"""
    
    def __init__(
        self,
        dataset_path: str,
        model_size: str = 'n',
        device: str = 'cpu'
    ):
        """
        Initialize YOLO trainer
        
        Args:
            dataset_path: Path to dataset directory
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            device: Device to use ('cpu' or 'cuda')
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.device = device
        
        # Validate dataset
        is_valid, errors = validate_dataset_structure(str(self.dataset_path))
        if not is_valid:
            print("Dataset validation errors:")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("Invalid dataset structure")
        
        # Load data info
        self.data_loader = YOLODataLoader(str(self.dataset_path))
        
        print(f"YOLO Trainer initialized")
        print(f"Dataset: {self.dataset_path}")
        print(f"Model size: yolov8{model_size}")
        print(f"Device: {device}")
    
    def create_data_yaml(self, output_path: Optional[str] = None) -> str:
        """
        Create data.yaml configuration file for YOLO training
        
        Args:
            output_path: Optional custom path for data.yaml
            
        Returns:
            Path to created data.yaml file
        """
        if output_path is None:
            output_path = self.dataset_path / "data.yaml"
        
        # Get class names
        class_names_dict = self.data_loader.class_names
        nc = len(class_names_dict)
        names = [class_names_dict[i] for i in sorted(class_names_dict.keys())]
        
        # Create YAML content
        data_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images',
            'val': 'images',  # Use same images for validation if no separate val set
            'nc': nc,
            'names': names
        }
        
        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"\nCreated data.yaml at: {output_path}")
        print(f"  Classes: {nc}")
        print(f"  Names: {names}")
        
        return str(output_path)
    
    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 50,
        save_dir: Optional[str] = None,
        pretrained: bool = True,
        resume: bool = False,
        **kwargs
    ):
        """
        Train YOLO model
        
        Args:
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            patience: Early stopping patience
            save_dir: Directory to save training results
            pretrained: Use pretrained weights
            resume: Resume training from last checkpoint
            **kwargs: Additional YOLO training arguments
        """
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. Install with: pip install ultralytics")
        
        # Create data.yaml
        data_yaml_path = self.create_data_yaml()
        
        # Setup save directory
        if save_dir is None:
            save_dir = "../models/detector"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        if resume:
            # Resume from last checkpoint
            checkpoint_path = save_dir / 'train' / 'weights' / 'last.pt'
            if checkpoint_path.exists():
                print(f"\nResuming training from: {checkpoint_path}")
                model = YOLO(str(checkpoint_path))
            else:
                print(f"\nWarning: No checkpoint found at {checkpoint_path}")
                print("Starting new training instead...")
                resume = False
                model_name = f"yolov8{self.model_size}.pt" if pretrained else f"yolov8{self.model_size}.yaml"
                model = YOLO(model_name)
        else:
            model_name = f"yolov8{self.model_size}.pt" if pretrained else f"yolov8{self.model_size}.yaml"
            print(f"\nInitializing model: {model_name}")
            model = YOLO(model_name)
        
        # Get dataset stats
        stats = self.data_loader.get_dataset_stats()
        
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Total images: {stats['total_images']}")
        print(f"Labeled images: {stats['total_labeled_images']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Classes: {len(stats['class_names'])}")
        for class_name, count in stats['class_counts'].items():
            print(f"  {class_name}: {count}")
        print(f"\nModel: yolov8{self.model_size}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Train model
        if resume:
            results = model.train(
                resume=True,
                **kwargs
            )
        else:
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                device=self.device,
                project=str(save_dir),
                name='train',
                exist_ok=True,
                pretrained=pretrained,
                **kwargs
            )
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved to: {save_dir / 'train'}")
        print(f"Best weights: {save_dir / 'train' / 'weights' / 'best.pt'}")
        print(f"Last weights: {save_dir / 'train' / 'weights' / 'last.pt'}")
        print(f"{'='*60}\n")
        
        return results
    
    def validate(
        self,
        model_path: str,
        imgsz: int = 640,
        **kwargs
    ):
        """
        Validate trained model
        
        Args:
            model_path: Path to trained model weights
            imgsz: Input image size
            **kwargs: Additional validation arguments
        """
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available")
        
        # Create data.yaml if needed
        data_yaml_path = self.create_data_yaml()
        
        print(f"\nValidating model: {model_path}")
        model = YOLO(model_path)
        
        results = model.val(
            data=data_yaml_path,
            imgsz=imgsz,
            device=self.device,
            **kwargs
        )
        
        print(f"\nValidation completed")
        return results
    
    def export_model(
        self,
        model_path: str,
        format: str = 'onnx',
        **kwargs
    ):
        """
        Export trained model to different format
        
        Args:
            model_path: Path to trained model weights
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            **kwargs: Additional export arguments
        """
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available")
        
        print(f"\nExporting model to {format} format...")
        model = YOLO(model_path)
        
        export_path = model.export(format=format, **kwargs)
        
        print(f"Model exported to: {export_path}")
        return export_path


def main():
    """Main function for command-line training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO detector on assembly dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        default='../dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../models/detector',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--validate',
        type=str,
        help='Path to model weights for validation only'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOTrainer(
        dataset_path=args.dataset,
        model_size=args.model_size,
        device=args.device
    )
    
    # Validate or train
    if args.validate:
        trainer.validate(args.validate, imgsz=args.imgsz)
    else:
        trainer.train(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            save_dir=args.output,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
