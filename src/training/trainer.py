"""
Comprehensive trainer for twin face verification using DCAL.
Supports multi-GPU training, mixed precision, gradient accumulation, and validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

import os
import time
import logging
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from tqdm import tqdm
import json

from .loss import DCALLoss, VerificationLoss, UncertaintyLoss
from .metrics import MetricsTracker, calculate_verification_metrics
from ..models import SiameseDCAL


class Trainer:
    """
    Comprehensive trainer for twin face verification.
    
    Supports multi-GPU training, mixed precision, gradient accumulation,
    learning rate scheduling, and validation.
    """
    
    def __init__(
        self,
        model: SiameseDCAL,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        config: Dict[str, Any] = None,
        logger: Optional[logging.Logger] = None,
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SiameseDCAL model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
            logger: Logger instance
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Ensure directories exist before setting up logger
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        
        # Setup model and training components
        self._setup_model()
        self._setup_loss_function(loss_fn)
        self._setup_optimizer(optimizer)
        self._setup_scheduler(scheduler)
        self._setup_mixed_precision()
        self._setup_tensorboard()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.best_threshold = 0.5
        self.metrics_tracker = MetricsTracker()
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        self.logger.info(f"Trainer initialized with device: {device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_model(self):
        """Setup model for training."""
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            if self.config.get('use_ddp', False):
                # Distributed training
                self.model = DistributedDataParallel(self.model)
            else:
                # DataParallel
                self.model = DataParallel(self.model)
        
        # Get the actual model (unwrap if wrapped in DataParallel)
        self.base_model = self.model.module if hasattr(self.model, 'module') else self.model
    
    def _setup_loss_function(self, loss_fn: Optional[nn.Module]):
        """Setup loss function."""
        if loss_fn is None:
            # Default loss configuration
            verification_loss = VerificationLoss(
                use_contrastive=True,
                use_triplet=False,
                use_focal=False,
                contrastive_margin=self.config.get('contrastive_margin', 1.0)
            )
            self.loss_fn = DCALLoss(
                verification_loss=verification_loss,
                attention_reg_weight=self.config.get('attention_reg_weight', 0.01),
                diversity_weight=self.config.get('diversity_weight', 0.1)
            )
        else:
            self.loss_fn = loss_fn
        
        self.loss_fn = self.loss_fn.to(self.device)
    
    def _setup_optimizer(self, optimizer: Optional[optim.Optimizer]):
        """Setup optimizer."""
        if optimizer is None:
            # Default optimizer configuration
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = optimizer
    
    def _setup_scheduler(self, scheduler: Optional[optim.lr_scheduler._LRScheduler]):
        """Setup learning rate scheduler."""
        if scheduler is None:
            # Default scheduler configuration
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('max_epochs', 100),
                eta_min=self.config.get('min_learning_rate', 1e-6)
            )
        else:
            self.scheduler = scheduler
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        self.use_amp = self.config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        if self.config.get('use_tensorboard', True) and HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        else:
            self.writer = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics_tracker.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            img1, img2, labels = batch
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.model(img1, img2, return_attention=True)
                
                # Compute loss
                loss, loss_components = self.loss_fn(
                    output['features1'],
                    output['features2'], 
                    labels,
                    attention_maps=output.get('attention1', {}),
                    logits=output['logits']
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics
            similarity_scores = output['similarity']
            self.metrics_tracker.update(labels, similarity_scores)
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log metrics
                if self.writer and self.global_step % self.config.get('log_interval', 100) == 0:
                    self._log_training_metrics(loss.item() * gradient_accumulation_steps, loss_components)
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Compute epoch metrics
        epoch_metrics = self.metrics_tracker.compute_metrics()
        epoch_metrics['loss'] = total_loss / num_batches
        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.metrics_tracker.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                img1, img2, labels = batch
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    output = self.model(img1, img2, return_attention=False)
                    
                    # Compute loss
                    loss, _ = self.loss_fn(
                        output['features1'],
                        output['features2'],
                        labels,
                        logits=output['logits']
                    )
                
                total_loss += loss.item()
                
                # Update metrics
                similarity_scores = output['similarity']
                self.metrics_tracker.update(labels, similarity_scores)
        
        # Compute validation metrics
        val_metrics = self.metrics_tracker.compute_metrics()
        val_metrics['loss'] = total_loss / num_batches
        
        # Update optimal threshold
        if hasattr(self.base_model, 'update_optimal_threshold'):
            all_labels = torch.cat([torch.tensor(self.metrics_tracker.all_labels)])
            all_scores = torch.cat([torch.tensor(self.metrics_tracker.all_scores)])
            self.base_model.update_optimal_threshold(all_labels, all_scores)
        
        return val_metrics
    
    def train(self, num_epochs: int, save_interval: int = 10):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_interval: Interval for saving checkpoints
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            self.val_history.append(val_metrics)
            
            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % save_interval == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping check
            if self._should_early_stop(val_metrics):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final metrics
        self._save_training_history()
        
        if self.writer:
            self.writer.close()
        
        self.logger.info("Training completed")
    
    def _log_training_metrics(self, loss: float, loss_components: Dict[str, torch.Tensor]):
        """Log training metrics to TensorBoard."""
        if self.writer:
            self.writer.add_scalar('Train/Loss', loss, self.global_step)
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Log loss components
            for name, value in loss_components.items():
                self.writer.add_scalar(f'Train/Loss_{name}', value.item(), self.global_step)
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch metrics."""
        # Console logging
        self.logger.info(f"Epoch {self.epoch}:")
        self.logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, ROC-AUC: {train_metrics.get('roc_auc', 0):.4f}")
        if val_metrics:
            self.logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
        
        # TensorBoard logging
        if self.writer:
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(f'Epoch/Train_{metric_name}', value, self.epoch)
            
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f'Epoch/Val_{metric_name}', value, self.epoch)
    
    def _should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        if not val_metrics:
            return False
        
        early_stop_patience = self.config.get('early_stop_patience', 10)
        early_stop_metric = self.config.get('early_stop_metric', 'roc_auc')
        
        if early_stop_metric not in val_metrics:
            return False
        
        current_score = val_metrics[early_stop_metric]
        
        # Check if current score is better than best
        if current_score > self.best_val_score:
            self.best_val_score = current_score
            self.best_threshold = val_metrics.get('threshold', 0.5)
            self.patience_counter = 0
            return False
        else:
            self.patience_counter = getattr(self, 'patience_counter', 0) + 1
            return self.patience_counter >= early_stop_patience
    
    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'best_threshold': self.best_threshold,
            'config': self.config,
            'val_metrics': val_metrics
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if val_metrics and val_metrics.get('roc_auc', 0) >= self.best_val_score:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"Saved best checkpoint with ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
    
    def _save_training_history(self):
        """Save training history to file."""
        history = {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        self.best_threshold = checkpoint.get('best_threshold', 0.5)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def resume_training(self, checkpoint_path: str, num_epochs: int):
        """Resume training from checkpoint."""
        self.load_checkpoint(checkpoint_path)
        remaining_epochs = num_epochs - self.epoch
        
        if remaining_epochs > 0:
            self.logger.info(f"Resuming training for {remaining_epochs} epochs")
            self.train(remaining_epochs)
        else:
            self.logger.info("Training already completed")


def create_trainer(
    model: SiameseDCAL,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Dict[str, Any] = None
) -> Trainer:
    """
    Create a trainer with default configuration.
    
    Args:
        model: SiameseDCAL model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
    
    Returns:
        Configured trainer
    """
    config = config or {}
    
    # Setup device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup directories
    log_dir = config.get('log_dir', './logs')
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    return trainer 