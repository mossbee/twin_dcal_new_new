#!/usr/bin/env python3
"""
Kaggle training script for DCAL Twin Face Verification.
Handles 12-hour timeout constraints, WandB tracking, and emergency checkpointing.
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import DCALEncoder, SiameseDCAL, VisionTransformer
from src.data import TwinDataset, get_train_transforms, get_val_transforms
from src.training import Trainer, create_trainer
from src.utils import Config, get_config, create_tracker, KaggleCheckpointManager, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DCAL Twin Face Verification Model on Kaggle')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/kaggle_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--auto-resume', action='store_true', default=True,
                       help='Automatically resume from latest checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='/kaggle/input/nd-twin',
                       help='Root directory for data')
    parser.add_argument('--image-size', type=int, default=None,
                       help='Image size for training')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name of the experiment')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name of the run')
    
    # Kaggle-specific
    parser.add_argument('--timeout-hours', type=float, default=11.5,
                       help='Hours before timeout (default 11.5 for safety)')
    parser.add_argument('--save-interval', type=int, default=1,
                       help='Checkpoint save interval (epochs)')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> 'Config':
    """Load configuration from file and override with command line arguments."""
    # Load base config if exists, otherwise create default
    if os.path.exists(config_path):
        config = get_config(config_path)
        config_dict = config.to_dict()
    else:
        config_dict = get_default_kaggle_config()
    
    # Override with command line arguments
    if args.epochs is not None:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config_dict['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_dict['training']['learning_rate'] = args.learning_rate
    if args.data_root is not None:
        config_dict['data']['data_dir'] = args.data_root
    if args.image_size is not None:
        config_dict['data']['image_size'] = args.image_size
    if args.experiment_name is not None:
        config_dict['system']['experiment_name'] = args.experiment_name
    if args.run_name is not None:
        config_dict['system']['run_name'] = args.run_name
    if args.timeout_hours is not None:
        config_dict['system']['timeout_hours'] = args.timeout_hours
    if args.save_interval is not None:
        config_dict['system']['save_interval'] = args.save_interval
    
    # Set environment
    config_dict['environment'] = 'kaggle'
    
    # Kaggle-specific optimizations
    config_dict['data']['num_workers'] = 2  # Kaggle has limited CPU
    config_dict['data']['pin_memory'] = True
    config_dict['training']['use_amp'] = True  # Use mixed precision
    config_dict['system']['max_checkpoints'] = 3  # Limit checkpoints
    
    # Debug mode
    if args.debug:
        config_dict['training']['epochs'] = 2
        config_dict['training']['batch_size'] = 4
        config_dict['data']['debug'] = True
    
    # Convert back to Config object
    return Config.from_dict(config_dict)


def get_default_kaggle_config() -> dict:
    """Get default configuration for Kaggle environment."""
    return {
        'environment': 'kaggle',
        'device': 'cuda',
        'model': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'patch_size': 16,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_sa_blocks': 12,
            'num_glca_blocks': 1,
            'num_pwca_blocks': 12,
            'local_ratio_fgvc': 0.1,
            'local_ratio_reid': 0.3,
            'similarity_function': 'cosine',
            'feature_dim': 768,
            'temperature': 0.07,
            'learnable_temperature': True,
            'pretrained': True
        },
        'data': {
            'image_size': 224,
            'train_dataset_info': '/kaggle/input/nd-twin/train_dataset_infor.json',
            'train_twin_pairs': '/kaggle/input/nd-twin/train_twin_pairs.json',
            'val_dataset_info': '/kaggle/input/nd-twin/test_dataset_infor.json',
            'val_twin_pairs': '/kaggle/input/nd-twin/test_twin_pairs.json',
            'data_root': '/kaggle/input/nd-twin',
            'negative_ratio': 1.0,
            'hard_negative_ratio': 0.3,
            'soft_negative_ratio': 0.2,
            'strong_augmentation': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_workers': 2,
            'pin_memory': True,
            'use_amp': True,
            'gradient_accumulation_steps': 4,
            'max_grad_norm': 1.0,
            'warmup_epochs': 5,
            'early_stop_patience': 10,
            'early_stop_metric': 'val_roc_auc'
        },
        'tracking': {
            'type': 'wandb',
            'experiment_name': 'dcal_twin_verification',
            'wandb_entity': 'hunchoquavodb-hanoi-university-of-science-and-technology',
            'wandb_project': 'dcal_twin_verification'
        },
        'checkpointing': {
            'checkpoint_dir': '/kaggle/working/checkpoints',
            'timeout_hours': 11.5,
            'save_interval': 1,
            'max_checkpoints': 3,
            'save_best': True,
            'best_metric': 'val_roc_auc',
            'best_mode': 'max'
        },
        'logging': {
            'log_level': 'INFO',
            'log_dir': '/kaggle/working/logs',
            'log_file': '/kaggle/working/logs/training.log'
        }
    }


def create_model(config: dict) -> SiameseDCAL:
    """Create the DCAL model."""
    model_config = config['model']
    data_config = config['data']
    dcal_config = config['dcal']
    
    # Create backbone
    backbone = VisionTransformer(
        img_size=data_config['image_size'],
        patch_size=data_config['patch_size'],
        in_channels=3,
        embed_dim=model_config['embed_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        mlp_ratio=4.0,  # Default value for ViT
        dropout=model_config['dropout'],
        pretrained=model_config.get('pretrained', True)
    )
    
    # Create DCAL encoder
    dcal_encoder = DCALEncoder(
        backbone_config=model_config['backbone'],  # Pass the backbone config name
        num_sa_blocks=dcal_config['num_sa_blocks'],
        num_glca_blocks=dcal_config['num_glca_blocks'],
        num_pwca_blocks=dcal_config['num_pwca_blocks'],
        local_ratio=dcal_config['local_ratio_fgvc'],  # Use fgvc ratio as default
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        mlp_ratio=4.0,
        dropout=model_config['dropout'],
        use_dynamic_loss=dcal_config['use_dynamic_loss']
    )
    
    # Create Siamese DCAL model with default values for missing parameters
    siamese_model = SiameseDCAL(
        dcal_encoder=dcal_encoder,
        similarity_function='cosine',  # Default value
        feature_dim=model_config['embed_dim'],
        dropout=model_config['dropout'],
        temperature=0.07,  # Default value
        learnable_temperature=True  # Default value
    )
    
    return siamese_model


def create_data_loaders(config: dict) -> tuple:
    """Create training and validation data loaders."""
    data_config = config['data']
    training_config = config['training']
    
    # Create transforms
    train_transform = get_train_transforms(
        image_size=data_config['image_size'],
        strong_augmentation=data_config.get('augmentation', True)
    )
    val_transform = get_val_transforms(image_size=data_config['image_size'])
    
    # Create datasets
    train_dataset = TwinDataset(
        dataset_info_path=data_config['train_info_file'],
        twin_pairs_path=data_config['train_pairs_file'],
        data_root=data_config['data_dir'],
        transform=train_transform,
        mode='train',
        negative_ratio=1.0,  # Default value
        hard_negative_ratio=data_config['hard_negative_ratio'],
        soft_negative_ratio=data_config['soft_negative_ratio']
    )
    
    val_dataset = TwinDataset(
        dataset_info_path=data_config['test_info_file'],
        twin_pairs_path=data_config['test_pairs_file'],
        data_root=data_config['data_dir'],
        transform=val_transform,
        mode='val',
        negative_ratio=1.0,  # Default value
        hard_negative_ratio=data_config['hard_negative_ratio'],
        soft_negative_ratio=data_config['soft_negative_ratio']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=False
    )
    
    return train_loader, val_loader


class KaggleTrainer:
    """Kaggle-specific trainer wrapper with timeout handling."""
    
    def __init__(self, trainer: Trainer, checkpoint_manager: KaggleCheckpointManager):
        self.trainer = trainer
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(__name__)
    
    def train_with_timeout(self, num_epochs: int):
        """Train with timeout monitoring."""
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Check timeout before each epoch
            if self.checkpoint_manager.is_timeout_approaching(buffer_minutes=20):
                self.logger.warning(f"Timeout approaching at epoch {epoch}, saving emergency checkpoint...")
                self.checkpoint_manager.emergency_save(
                    model=self.trainer.base_model,
                    optimizer=self.trainer.optimizer,
                    epoch=epoch,
                    batch_idx=0,
                    metrics=self.trainer.val_history[-1] if self.trainer.val_history else {},
                    scheduler=self.trainer.scheduler,
                    scaler=self.trainer.scaler if hasattr(self.trainer, 'scaler') else None
                )
                self.logger.info(f"Emergency checkpoint saved. Stopping training at epoch {epoch}")
                break
            
            # Train one epoch
            try:
                self.trainer.epoch = epoch
                train_metrics = self.trainer.train_epoch()
                val_metrics = self.trainer.validate_epoch()
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    model=self.trainer.base_model,
                    optimizer=self.trainer.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    scheduler=self.trainer.scheduler,
                    scaler=self.trainer.scaler if hasattr(self.trainer, 'scaler') else None,
                    config=self.trainer.config
                )
                
                # Log metrics
                self.trainer._log_epoch_metrics(train_metrics, val_metrics)
                
                # Check early stopping
                if self.trainer._should_early_stop(val_metrics):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Update history
                self.trainer.train_history.append(train_metrics)
                self.trainer.val_history.append(val_metrics)
                
                # Log remaining time
                remaining_time = self.checkpoint_manager.get_time_remaining()
                self.logger.info(f"Epoch {epoch} completed. Time remaining: {remaining_time:.2f} hours")
                
            except Exception as e:
                self.logger.error(f"Error in epoch {epoch}: {e}")
                # Save emergency checkpoint
                self.checkpoint_manager.emergency_save(
                    model=self.trainer.base_model,
                    optimizer=self.trainer.optimizer,
                    epoch=epoch,
                    batch_idx=0,
                    metrics=self.trainer.val_history[-1] if self.trainer.val_history else {},
                    scheduler=self.trainer.scheduler,
                    scaler=self.trainer.scaler if hasattr(self.trainer, 'scaler') else None
                )
                raise


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Setup logging
    logger = setup_logging(config.system.__dict__)
    logger.info("Starting DCAL Twin Face Verification training on Kaggle")
    logger.info(f"Configuration: {config}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available on Kaggle!")
        return
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config.to_dict())
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config.to_dict())
    logger.info(f"Train dataset: {len(train_loader.dataset)} samples")
    logger.info(f"Val dataset: {len(val_loader.dataset)} samples")
    
    # Create experiment tracker
    logger.info("Setting up WandB tracking...")
    tracker = create_tracker(config.system.__dict__)
    
    # Create Kaggle checkpoint manager
    logger.info("Setting up Kaggle checkpoint manager...")
    checkpoint_manager = KaggleCheckpointManager(
        checkpoint_dir=config.system.checkpoint_dir,
        timeout_hours=config.system.timeout_hours,
        max_checkpoints=config.system.max_checkpoints,
        save_best=True,
        best_metric='roc_auc',
        best_mode='max'
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.to_dict()
    )
    
    # Auto-resume from checkpoint
    start_epoch = 0
    if args.auto_resume:
        logger.info("Attempting to auto-resume from checkpoint...")
        
        # Try quick checkpoint first
        checkpoint_data = checkpoint_manager.load_quick_checkpoint(
            model=trainer.base_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if hasattr(trainer, 'scaler') else None
        )
        
        # If no quick checkpoint, try regular checkpoint
        if checkpoint_data is None:
            checkpoint_data = checkpoint_manager.auto_resume_training(
                model=trainer.base_model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler if hasattr(trainer, 'scaler') else None
            )
        
        if checkpoint_data:
            start_epoch = checkpoint_data['epoch'] + 1
            logger.info(f"Resumed from epoch {checkpoint_data['epoch']}")
            # Update trainer state
            trainer.epoch = start_epoch
            trainer.best_val_score = checkpoint_data.get('best_val_score', 0.0)
    
    # Start experiment tracking
    tracker.start_run(config.to_dict())
    
    # Create Kaggle trainer wrapper
    kaggle_trainer = KaggleTrainer(trainer, checkpoint_manager)
    
    try:
        # Train model with timeout monitoring
        logger.info("Starting training with timeout monitoring...")
        kaggle_trainer.train_with_timeout(config.training.epochs)
        
        # Save final model
        logger.info("Saving final model...")
        checkpoint_manager.save_model_only(trainer.base_model, "final_model.pth")
        
        # Log final model to WandB
        tracker.log_model(trainer.base_model, "final_model")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        checkpoint_manager.emergency_save(
            model=trainer.base_model,
            optimizer=trainer.optimizer,
            epoch=trainer.epoch,
            batch_idx=0,
            metrics=trainer.val_history[-1] if trainer.val_history else {},
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if hasattr(trainer, 'scaler') else None
        )
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        checkpoint_manager.emergency_save(
            model=trainer.base_model,
            optimizer=trainer.optimizer,
            epoch=trainer.epoch,
            batch_idx=0,
            metrics=trainer.val_history[-1] if trainer.val_history else {},
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if hasattr(trainer, 'scaler') else None
        )
        raise
        
    finally:
        # End tracking
        tracker.end_run()
        
        # Save final checkpoint info
        checkpoint_info = checkpoint_manager.get_checkpoint_info()
        logger.info(f"Final checkpoint info: {checkpoint_info}")
        
        # Log time usage
        remaining_time = checkpoint_manager.get_time_remaining()
        logger.info(f"Time remaining: {remaining_time:.2f} hours")


if __name__ == '__main__':
    main() 