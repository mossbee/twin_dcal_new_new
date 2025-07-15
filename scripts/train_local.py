#!/usr/bin/env python3
"""
Local training script for DCAL Twin Face Verification.
Supports multi-GPU training, experiment tracking, and checkpointing.
"""

import os
import sys
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
from src.utils import Config, get_config, create_tracker, create_checkpoint_manager, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DCAL Twin Face Verification Model')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/local_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--auto-resume', action='store_true',
                       help='Automatically resume from latest checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory for data')
    parser.add_argument('--image-size', type=int, default=None,
                       help='Image size for training')
    
    # Experiment tracking
    parser.add_argument('--tracking', type=str, default=None,
                       choices=['mlflow', 'wandb', 'none'],
                       help='Experiment tracking backend')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name of the experiment')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name of the run')
    
    # Output directories
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory for checkpoints')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load configuration from file and override with command line arguments."""
    config = get_config(config_path)
    config_dict = config.to_dict()
    
    # Override with command line arguments
    if args.epochs is not None:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config_dict['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_dict['training']['learning_rate'] = args.learning_rate
    if args.device is not None:
        config_dict['device'] = args.device
    if args.data_root is not None:
        config_dict['data']['data_root'] = args.data_root
    if args.image_size is not None:
        config_dict['data']['image_size'] = args.image_size
    if args.tracking is not None:
        config_dict['tracking'] = config_dict.get('tracking', {})
        config_dict['tracking']['type'] = args.tracking
    if args.experiment_name is not None:
        config_dict['tracking'] = config_dict.get('tracking', {})
        config_dict['tracking']['experiment_name'] = args.experiment_name
    if args.run_name is not None:
        config_dict['tracking'] = config_dict.get('tracking', {})
        config_dict['tracking']['run_name'] = args.run_name
    if args.log_dir is not None:
        config_dict['logging'] = config_dict.get('logging', {})
        config_dict['logging']['log_dir'] = args.log_dir
    if args.checkpoint_dir is not None:
        config_dict['checkpointing'] = config_dict.get('checkpointing', {})
        config_dict['checkpointing']['checkpoint_dir'] = args.checkpoint_dir
    
    # Set environment
    config_dict['environment'] = 'local'
    
    # Debug mode
    if args.debug:
        config_dict['training']['epochs'] = 2
        config_dict['training']['batch_size'] = 4
        config_dict['data']['debug'] = True
    
    return config_dict


def create_model(config: dict) -> SiameseDCAL:
    """Create the DCAL model."""
    model_config = config['model']
    
    # Create backbone
    backbone = VisionTransformer(
        img_size=config['data']['image_size'],
        patch_size=model_config['patch_size'],
        in_channels=3,
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config['mlp_ratio'],
        dropout=model_config['dropout'],
        pretrained=model_config.get('pretrained', True)
    )
    
    # Create DCAL encoder
    dcal_encoder = DCALEncoder(
        backbone=backbone,
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_sa_blocks=model_config['num_sa_blocks'],
        num_glca_blocks=model_config['num_glca_blocks'],
        num_pwca_blocks=model_config['num_pwca_blocks'],
        local_ratio_fgvc=model_config['local_ratio_fgvc'],
        local_ratio_reid=model_config['local_ratio_reid'],
        dropout=model_config['dropout']
    )
    
    # Create Siamese DCAL model
    siamese_model = SiameseDCAL(
        dcal_encoder=dcal_encoder,
        similarity_function=model_config['similarity_function'],
        feature_dim=model_config['feature_dim'],
        dropout=model_config['dropout'],
        temperature=model_config['temperature'],
        learnable_temperature=model_config['learnable_temperature']
    )
    
    return siamese_model


def create_data_loaders(config: dict) -> tuple:
    """Create training and validation data loaders."""
    data_config = config['data']
    training_config = config['training']
    
    # Create transforms
    train_transform = get_train_transforms(
        image_size=data_config['image_size'],
        strong_augmentation=data_config.get('strong_augmentation', True)
    )
    val_transform = get_val_transforms(image_size=data_config['image_size'])
    
    # Create datasets
    train_dataset = TwinDataset(
        dataset_info_path=data_config['train_dataset_info'],
        twin_pairs_path=data_config['train_twin_pairs'],
        data_root=data_config.get('data_root', ''),
        transform=train_transform,
        mode='train',
        negative_ratio=data_config.get('negative_ratio', 1.0),
        hard_negative_ratio=data_config.get('hard_negative_ratio', 0.3),
        soft_negative_ratio=data_config.get('soft_negative_ratio', 0.2)
    )
    
    val_dataset = TwinDataset(
        dataset_info_path=data_config['val_dataset_info'],
        twin_pairs_path=data_config['val_twin_pairs'],
        data_root=data_config.get('data_root', ''),
        transform=val_transform,
        mode='val',
        negative_ratio=data_config.get('negative_ratio', 1.0),
        hard_negative_ratio=data_config.get('hard_negative_ratio', 0.3),
        soft_negative_ratio=data_config.get('soft_negative_ratio', 0.2)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Setup logging
    logger = setup_logging(config.get('logging', {}))
    logger.info("Starting DCAL Twin Face Verification training")
    logger.info(f"Configuration: {config}")
    
    # Setup device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f"Train dataset: {len(train_loader.dataset)} samples")
    logger.info(f"Val dataset: {len(val_loader.dataset)} samples")
    
    # Create experiment tracker
    logger.info("Setting up experiment tracking...")
    tracker = create_tracker(config.get('tracking', {}))
    
    # Create checkpoint manager
    logger.info("Setting up checkpoint manager...")
    checkpoint_manager = create_checkpoint_manager(config.get('checkpointing', {}))
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Auto-resume or manual resume
    start_epoch = 0
    if args.auto_resume:
        logger.info("Attempting to auto-resume from latest checkpoint...")
        checkpoint_data = checkpoint_manager.auto_resume_training(
            model=trainer.base_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if hasattr(trainer, 'scaler') else None
        )
        if checkpoint_data:
            start_epoch = checkpoint_data['epoch'] + 1
            logger.info(f"Resumed from epoch {checkpoint_data['epoch']}")
    elif args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_data = checkpoint_manager.load_checkpoint(
            model=trainer.base_model,
            optimizer=trainer.optimizer,
            checkpoint_path=args.resume,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if hasattr(trainer, 'scaler') else None
        )
        start_epoch = checkpoint_data['epoch'] + 1
        logger.info(f"Resumed from epoch {checkpoint_data['epoch']}")
    
    # Start experiment tracking
    tracker.start_run(config)
    
    try:
        # Train model
        logger.info("Starting training...")
        trainer.train(
            num_epochs=config['training']['epochs'],
            save_interval=config['checkpointing'].get('save_interval', 1)
        )
        
        # Save final model
        logger.info("Saving final model...")
        checkpoint_manager.save_model_only(trainer.base_model, "final_model.pth")
        
        # Log final model
        tracker.log_model(trainer.base_model, "final_model")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save emergency checkpoint
        checkpoint_manager.save_checkpoint(
            model=trainer.base_model,
            optimizer=trainer.optimizer,
            epoch=trainer.epoch,
            metrics=trainer.val_history[-1] if trainer.val_history else {},
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if hasattr(trainer, 'scaler') else None,
            config=config,
            force_save=True
        )
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
        
    finally:
        # End tracking
        tracker.end_run()
        
        # Save tracking history if no tracking
        if hasattr(tracker.tracker, 'save_history'):
            history_path = os.path.join(
                config.get('logging', {}).get('log_dir', './logs'),
                'training_history.json'
            )
            tracker.save_history(history_path)


if __name__ == '__main__':
    main() 