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
from src.utils import Config, get_config, create_tracker, CheckpointManager, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DCAL Twin Face Verification Model')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/kaggle_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory for data')
    parser.add_argument('--image-size', type=int, default=None,
                       help='Image size for training')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name of the experiment')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name of the run')
    
    # Training parameters
    parser.add_argument('--save-interval', type=int, default=1,
                       help='Checkpoint save interval (epochs)')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help='Path to checkpoint to resume from (if not provided, will use latest)')
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> 'Config':
    """Load configuration from file and override with command line arguments."""
    # Load base config if exists, otherwise create default
    if os.path.exists(config_path):
        config = get_config(config_path)
        config_dict = config.to_dict()
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
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
    if args.save_interval is not None:
        config_dict['system']['save_interval'] = args.save_interval
    
    # Debug mode
    if args.debug:
        config_dict['training']['epochs'] = 5
        config_dict['training']['batch_size'] = 4
        config_dict['data']['debug'] = True
    elif config_dict['data'].get('debug', False):
        # If debug is set in config file, also reduce epochs
        config_dict['training']['epochs'] = 5
    
    # Convert back to Config object
    return Config.from_dict(config_dict)


def create_model(config: dict) -> SiameseDCAL:
    model_config = config['model']
    data_config = config['data']
    dcal_config = config['dcal']
    # Use 'num_layers' for VisionTransformer, fallback to 'depth' if needed
    num_layers = model_config.get('num_layers', model_config.get('depth'))
    # Prepare DCALEncoder arguments
    dcal_encoder = DCALEncoder(
        backbone_config=model_config.get('backbone', 'vit_base_patch16_224'),
        num_sa_blocks=dcal_config.get('num_sa_blocks', 12),
        num_glca_blocks=dcal_config.get('num_glca_blocks', 1),
        num_pwca_blocks=dcal_config.get('num_pwca_blocks', 12),
        local_ratio=dcal_config.get('local_ratio_fgvc', 0.1),
        embed_dim=model_config.get('embed_dim', 768),
        num_heads=model_config.get('num_heads', 12),
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        dropout=model_config.get('dropout', 0.1),
        pretrained=model_config.get('pretrained', True),
        pretrained_path=model_config.get('pretrained_path', None),
        num_layers=num_layers,
        use_dynamic_loss=dcal_config.get('use_dynamic_loss', True)
    )
    model = SiameseDCAL(
        dcal_encoder=dcal_encoder,
        similarity_function='cosine',
        feature_dim=model_config.get('embed_dim', 768),
        dropout=model_config.get('dropout', 0.1),
        temperature=model_config.get('temperature', 0.07),
        learnable_temperature=model_config.get('learnable_temperature', True)
    )
    return model


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
        soft_negative_ratio=data_config['soft_negative_ratio'],
        use_only_hard_negatives=data_config.get('use_only_hard_negatives', True)
    )
    
    val_dataset = TwinDataset(
        dataset_info_path=data_config['test_info_file'],
        twin_pairs_path=data_config['test_pairs_file'],
        data_root=data_config['data_dir'],
        transform=val_transform,
        mode='val',
        negative_ratio=1.0,  # Default value
        hard_negative_ratio=data_config['hard_negative_ratio'],
        soft_negative_ratio=data_config['soft_negative_ratio'],
        use_only_hard_negatives=data_config.get('use_only_hard_negatives', True)
    )
    
    # Debug mode: use smaller subset
    if data_config.get('debug', False):
        from torch.utils.data import Subset
        import numpy as np
        
        # Use only 1000 samples for training and 200 for validation
        train_indices = np.random.choice(len(train_dataset), min(1000, len(train_dataset)), replace=False)
        val_indices = np.random.choice(len(val_dataset), min(200, len(val_dataset)), replace=False)
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        print(f"[DEBUG MODE] Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
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


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Setup logging
    logger = setup_logging(config.system.__dict__)
    logger.info("Starting DCAL Twin Face Verification training")
    logger.info(f"Configuration: {config}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
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
    
    # Create checkpoint manager
    logger.info("Setting up checkpoint manager...")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.system.checkpoint_dir,
        max_checkpoints=config.system.max_checkpoints,
        save_best=True,
        best_metric='roc_auc',
        best_mode='max',
        save_interval=config.system.save_interval,
        auto_resume=False
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
    if args.resume:
        logger.info("Resuming from checkpoint...")
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = checkpoint_manager.get_latest_checkpoint()
        if checkpoint_path is not None:
            checkpoint_data = checkpoint_manager.load_checkpoint(
                model=trainer.base_model,
                optimizer=trainer.optimizer,
                checkpoint_path=checkpoint_path,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler if hasattr(trainer, 'scaler') else None
            )
            if checkpoint_data:
                start_epoch = checkpoint_data['epoch'] + 1
                logger.info(f"Resumed from epoch {checkpoint_data['epoch']}")
                trainer.epoch = start_epoch
                trainer.best_val_score = checkpoint_data.get('best_val_score', 0.0)
        else:
            logger.warning("No checkpoint found to resume from. Starting fresh.")
    
    # Start experiment tracking
    tracker.start_run(config.to_dict())
    
    try:
        # Train model with timeout monitoring
        logger.info("Starting training...")
        for epoch in range(start_epoch, config.training.epochs):
            trainer.epoch = epoch
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate_epoch()
            checkpoint_manager.save_checkpoint(
                model=trainer.base_model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                metrics=val_metrics,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler if hasattr(trainer, 'scaler') else None,
                config=trainer.config
            )
            trainer._log_epoch_metrics(train_metrics, val_metrics)
            if trainer._should_early_stop(val_metrics):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            trainer.train_history.append(train_metrics)
            trainer.val_history.append(val_metrics)
        logger.info("Saving final model...")
        checkpoint_manager.save_model_only(trainer.base_model, "final_model.pth")
        tracker.log_model(trainer.base_model, "final_model")
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
        
    finally:
        # End tracking
        tracker.end_run()
        
        # Save final checkpoint info
        checkpoint_info = checkpoint_manager.get_checkpoint_info()
        logger.info(f"Final checkpoint info: {checkpoint_info}")


if __name__ == '__main__':
    main() 