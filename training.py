#!/usr/bin/env python3
"""
YOLOv8 Nano Training Script for Helmet Detection Dataset
Optimized for RTX 3060 GPU (12GB VRAM)
Author: AI Assistant
"""

import os
import torch
import gc
from ultralytics import YOLO
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HelmetDetectionTrainer:
    def __init__(self, 
                 data_yaml_path="data.yaml",
                 model_variant="yolov8n.pt",
                 project_name="helmet_detection",
                 experiment_name="yolov8n_training"):
        
        self.data_yaml_path = data_yaml_path
        self.model_variant = model_variant
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        # GPU Configuration for RTX 3060
        self.device = self._setup_gpu()
        
        # Training parameters optimized for RTX 3060 (12GB VRAM)
        self.training_params = {
            'epochs': 100,
            'imgsz': 640,          # Standard YOLO input size
            'batch': 32,           # Optimized for 12GB VRAM
            'workers': 8,          # Adjust based on CPU cores
            'patience': 50,        # Early stopping patience
            'save_period': 5,     # Save checkpoint every 10 epochs
            'device': self.device,
            'project': self.project_name,
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,          # Initial learning rate
            'lrf': 0.1,           # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,           # Box loss gain
            'cls': 0.5,           # Class loss gain
            'dfl': 1.5,           # DFL loss gain
            'pose': 12.0,         # Pose loss gain (if applicable)
            'kobj': 1.0,          # Keypoint objective loss gain
            'label_smoothing': 0.0,
            'nbs': 64,            # Nominal batch size
            'hsv_h': 0.015,       # HSV-Hue augmentation
            'hsv_s': 0.7,         # HSV-Saturation augmentation
            'hsv_v': 0.4,         # HSV-Value augmentation
            'degrees': 0.0,       # Rotation degrees
            'translate': 0.1,     # Translation fraction
            'scale': 0.5,         # Scale fraction
            'shear': 0.0,         # Shear degrees
            'perspective': 0.0,   # Perspective fraction
            'flipud': 0.0,        # Vertical flip probability
            'fliplr': 0.5,        # Horizontal flip probability
            'mosaic': 1.0,        # Mosaic probability
            'mixup': 0.0,         # Mixup probability
            'copy_paste': 0.0,    # Copy-paste probability
            'auto_augment': 'randaugment',
            'erasing': 0.4,       # Random erasing probability
            'crop_fraction': 1.0, # Crop fraction
        }
        
        # Checkpoint configuration
        self.checkpoint_dir = Path(f"checkpoints/{self.experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_gpu(self):
        """Setup GPU configuration for RTX 3060"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} GPU(s)")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Use first GPU (assuming RTX 3060)
            device = 0
            torch.cuda.set_device(device)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Enable mixed precision training for better memory efficiency
            torch.backends.cudnn.benchmark = True
            
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return 'cpu'
    
    def verify_dataset(self):
        """Verify dataset structure and paths"""
        logger.info("Verifying dataset structure...")
        
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"Dataset YAML not found: {self.data_yaml_path}")
        
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data_config:
                raise ValueError(f"Missing required field in data.yaml: {field}")
        
        # Verify paths exist
        base_path = Path(self.data_yaml_path).parent
        train_path = base_path / data_config['train']
        val_path = base_path / data_config['val']
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training images directory not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation images directory not found: {val_path}")
        
        # Count images
        train_images = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.jpeg'))) + len(list(train_path.glob('*.png')))
        val_images = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.jpeg'))) + len(list(val_path.glob('*.png')))
        
        logger.info(f"Dataset verified successfully!")
        logger.info(f"Classes: {data_config['names']}")
        logger.info(f"Number of classes: {data_config['nc']}")
        logger.info(f"Training images: {train_images}")
        logger.info(f"Validation images: {val_images}")
        
        return data_config
    
    def optimize_batch_size(self):
        """Automatically find optimal batch size for RTX 3060"""
        logger.info("Finding optimal batch size for RTX 3060...")
        
        # Start with a reasonable batch size and test
        test_batch_sizes = [64, 48, 32, 24, 16, 12, 8]
        optimal_batch_size = 8  # fallback
        
        for batch_size in test_batch_sizes:
            try:
                # Create a small test model
                model = YOLO(self.model_variant)
                
                # Try to allocate memory for this batch size
                dummy_input = torch.randn(batch_size, 3, 640, 640).cuda()
                
                with torch.no_grad():
                    _ = model.model(dummy_input)
                
                optimal_batch_size = batch_size
                logger.info(f"Optimal batch size found: {batch_size}")
                
                # Clean up
                del model, dummy_input
                torch.cuda.empty_cache()
                gc.collect()
                break
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info(f"Batch size {batch_size} too large, trying smaller...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        self.training_params['batch'] = optimal_batch_size
        return optimal_batch_size
    
    def setup_callbacks(self, model):
        """Setup custom callbacks for monitoring and checkpointing"""
        def on_train_epoch_end(trainer):
            """Callback executed at the end of each training epoch"""
            epoch = trainer.epoch
            
            # Save checkpoint every save_period epochs
            if epoch % self.training_params['save_period'] == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                trainer.save_model(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Epoch {epoch} - GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        def on_train_start(trainer):
            """Callback executed at the start of training"""
            logger.info("Training started!")
            logger.info(f"Model: {self.model_variant}")
            logger.info(f"Dataset: {self.data_yaml_path}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Batch size: {self.training_params['batch']}")
            logger.info(f"Image size: {self.training_params['imgsz']}")
            logger.info(f"Epochs: {self.training_params['epochs']}")
        
        def on_train_end(trainer):
            """Callback executed at the end of training"""
            logger.info("Training completed!")
            
            # Save final model
            final_model_path = self.checkpoint_dir / "final_model.pt"
            trainer.save_model(final_model_path)
            logger.info(f"Final model saved: {final_model_path}")
        
        # Add callbacks to model
        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        model.add_callback('on_train_start', on_train_start)
        model.add_callback('on_train_end', on_train_end)
    
    def train(self):
        """Main training function"""
        try:
            # Verify dataset
            data_config = self.verify_dataset()
            
            # Optimize batch size for RTX 3060
            optimal_batch_size = self.optimize_batch_size()
            
            # Initialize model
            logger.info(f"Loading YOLOv8 model: {self.model_variant}")
            model = YOLO(self.model_variant)
            
            # Setup callbacks
            self.setup_callbacks(model)
            
            # Log training configuration
            logger.info("Training Configuration:")
            for key, value in self.training_params.items():
                logger.info(f"  {key}: {value}")
            
            # Start training
            logger.info("Starting training...")
            results = model.train(
                data=self.data_yaml_path,
                **self.training_params
            )
            
            # Log results
            logger.info("Training completed successfully!")
            logger.info(f"Best model saved to: {model.trainer.best}")
            logger.info(f"Results saved to: {model.trainer.save_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e
        
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def resume_training(self, checkpoint_path):
        """Resume training from a checkpoint"""
        logger.info(f"Resuming training from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model from checkpoint
        model = YOLO(checkpoint_path)
        
        # Setup callbacks
        self.setup_callbacks(model)
        
        # Resume training
        results = model.train(
            data=self.data_yaml_path,
            resume=True,
            **self.training_params
        )
        
        return results

def main():
    """Main function to run the training"""
    # Initialize trainer
    trainer = HelmetDetectionTrainer()
    
    # Check if there's a checkpoint to resume from
    checkpoint_dir = trainer.checkpoint_dir
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    if checkpoints:
        # Find the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        
        print(f"Found checkpoint: {latest_checkpoint}")
        resume = input("Do you want to resume from this checkpoint? (y/n): ").lower().strip()
        
        if resume == 'y':
            trainer.resume_training(latest_checkpoint)
        else:
            trainer.train()
    else:
        trainer.train()

if __name__ == "__main__":
    main()