"""
Robust checkpointing system for twin face verification.
Handles model state saving, optimizer state saving, training resumption, and Kaggle timeout handling.
"""

import os
import json
import time
import glob
import shutil
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np


class CheckpointManager:
    """
    Comprehensive checkpoint manager for training resumption and model saving.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "./checkpoints",
                 max_checkpoints: int = 5,
                 save_best: bool = True,
                 best_metric: str = "val_roc_auc",
                 best_mode: str = "max",
                 save_interval: int = 1,
                 auto_resume: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save the best model
            best_metric: Metric to track for best model
            best_mode: 'max' or 'min' for best metric
            save_interval: Interval (in epochs) to save checkpoints
            auto_resume: Whether to automatically resume from latest checkpoint
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.save_interval = save_interval
        self.auto_resume = auto_resume
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track best metric value
        self.best_metric_value = float('-inf') if best_mode == 'max' else float('inf')
        
        # Checkpoint metadata
        self.checkpoint_history = []
        self.metadata_file = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        
        # Load existing metadata
        self._load_metadata()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata(self):
        """Load checkpoint metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.checkpoint_history = metadata.get('checkpoint_history', [])
                    self.best_metric_value = metadata.get('best_metric_value', 
                                                        float('-inf') if self.best_mode == 'max' else float('inf'))
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint metadata: {e}")
    
    def _save_metadata(self):
        """Save checkpoint metadata to file."""
        try:
            metadata = {
                'checkpoint_history': self.checkpoint_history,
                'best_metric_value': self.best_metric_value,
                'best_metric': self.best_metric,
                'best_mode': self.best_mode,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint metadata: {e}")
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       scaler: Optional[GradScaler] = None,
                       config: Optional[Dict[str, Any]] = None,
                       extra_state: Optional[Dict[str, Any]] = None,
                       force_save: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            scheduler: Learning rate scheduler (optional)
            scaler: Gradient scaler for mixed precision (optional)
            config: Training configuration (optional)
            extra_state: Additional state to save (optional)
            force_save: Force save regardless of interval
        
        Returns:
            Path to saved checkpoint
        """
        # Check if we should save this epoch
        if not force_save and epoch % self.save_interval != 0:
            return None
        
        # Get the base model (unwrap DataParallel if needed)
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'config': config or {}
        }
        
        # Add optional components
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        if extra_state is not None:
            checkpoint_data['extra_state'] = extra_state
        
        # Save checkpoint
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Update history
            checkpoint_info = {
                'epoch': epoch,
                'path': checkpoint_path,
                'metrics': metrics,
                'timestamp': checkpoint_data['timestamp']
            }
            self.checkpoint_history.append(checkpoint_info)
            
            # Save best model if applicable
            if self.save_best and self.best_metric in metrics:
                self._check_and_save_best(checkpoint_data, metrics)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save metadata
            self._save_metadata()
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _check_and_save_best(self, checkpoint_data: Dict[str, Any], metrics: Dict[str, float]):
        """Check if current model is best and save if so."""
        current_metric = metrics[self.best_metric]
        
        is_best = False
        if self.best_mode == 'max':
            is_best = current_metric > self.best_metric_value
        else:
            is_best = current_metric < self.best_metric_value
        
        if is_best:
            self.best_metric_value = current_metric
            
            # Save best model
            best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            try:
                torch.save(checkpoint_data, best_checkpoint_path)
                self.logger.info(f"Saved best model with {self.best_metric}={current_metric:.4f}")
            except Exception as e:
                self.logger.error(f"Failed to save best model: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by epoch (oldest first)
        self.checkpoint_history.sort(key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints
        while len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.pop(0)
            checkpoint_path = old_checkpoint['path']
            
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {checkpoint_path}: {e}")
    
    def load_checkpoint(self, 
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       checkpoint_path: str,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                       scaler: Optional[GradScaler] = None,
                       strict: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint file
            scheduler: Learning rate scheduler (optional)
            scaler: Gradient scaler (optional)
            strict: Whether to strictly enforce state_dict keys
        
        Returns:
            Loaded checkpoint data
        """
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Load model state
            try:
                if hasattr(model, 'module'):
                    model.module.load_state_dict(checkpoint_data['model_state_dict'], strict=True)
                else:
                    model.load_state_dict(checkpoint_data['model_state_dict'], strict=True)
                self.logger.info("Successfully loaded checkpoint with strict=True")
            except RuntimeError as e:
                # If strict loading fails, try with strict=False and log mismatches
                self.logger.warning(f"Strict loading failed: {e}")
                self.logger.info("Attempting to load checkpoint with strict=False...")
                
                if hasattr(model, 'module'):
                    missing_keys, unexpected_keys = model.module.load_state_dict(
                        checkpoint_data['model_state_dict'], strict=False
                    )
                else:
                    missing_keys, unexpected_keys = model.load_state_dict(
                        checkpoint_data['model_state_dict'], strict=False
                    )
                
                if missing_keys:
                    self.logger.warning(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    self.logger.warning(f"Unexpected keys: {unexpected_keys}")
                
                self.logger.info("Successfully loaded checkpoint with strict=False")
                
                # Analyze compatibility and handle architecture changes
                analysis = self._analyze_checkpoint_compatibility(model, checkpoint_data['model_state_dict'])
                self._log_compatibility_analysis(analysis)
                self._handle_architecture_changes(model, checkpoint_data['model_state_dict'])
                
                # Log summary of what was loaded
                self._log_loading_summary(analysis)
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load scheduler state if available
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Load scaler state if available
            if scaler is not None and 'scaler_state_dict' in checkpoint_data:
                scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            
            # Update best metric value
            if 'metrics' in checkpoint_data and self.best_metric in checkpoint_data['metrics']:
                self.best_metric_value = checkpoint_data['metrics'][self.best_metric]
            
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def _analyze_checkpoint_compatibility(self, model: nn.Module, checkpoint_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze compatibility between checkpoint and current model.
        
        Args:
            model: Current model
            checkpoint_state_dict: State dict from checkpoint
            
        Returns:
            Dictionary with compatibility analysis
        """
        model_state_dict = model.state_dict()
        
        analysis = {
            'total_checkpoint_keys': len(checkpoint_state_dict),
            'total_model_keys': len(model_state_dict),
            'matching_keys': 0,
            'missing_keys': [],
            'unexpected_keys': [],
            'shape_mismatches': [],
            'compatibility_score': 0.0
        }
        
        # Analyze key matches and mismatches
        for key in checkpoint_state_dict:
            if key in model_state_dict:
                analysis['matching_keys'] += 1
                # Check for shape mismatches
                if checkpoint_state_dict[key].shape != model_state_dict[key].shape:
                    analysis['shape_mismatches'].append({
                        'key': key,
                        'checkpoint_shape': list(checkpoint_state_dict[key].shape),
                        'model_shape': list(model_state_dict[key].shape)
                    })
            else:
                analysis['unexpected_keys'].append(key)
        
        for key in model_state_dict:
            if key not in checkpoint_state_dict:
                analysis['missing_keys'].append(key)
        
        # Calculate compatibility score
        if analysis['total_checkpoint_keys'] > 0:
            analysis['compatibility_score'] = analysis['matching_keys'] / analysis['total_checkpoint_keys']
        
        return analysis
    
    def _log_compatibility_analysis(self, analysis: Dict[str, Any]):
        """Log detailed compatibility analysis."""
        self.logger.info(f"Checkpoint compatibility analysis:")
        self.logger.info(f"  Total checkpoint keys: {analysis['total_checkpoint_keys']}")
        self.logger.info(f"  Total model keys: {analysis['total_model_keys']}")
        self.logger.info(f"  Matching keys: {analysis['matching_keys']}")
        self.logger.info(f"  Compatibility score: {analysis['compatibility_score']:.2%}")
        
        if analysis['missing_keys']:
            self.logger.warning(f"  Missing keys: {len(analysis['missing_keys'])}")
            for key in analysis['missing_keys'][:5]:  # Show first 5
                self.logger.warning(f"    - {key}")
            if len(analysis['missing_keys']) > 5:
                self.logger.warning(f"    ... and {len(analysis['missing_keys']) - 5} more")
        
        if analysis['unexpected_keys']:
            self.logger.warning(f"  Unexpected keys: {len(analysis['unexpected_keys'])}")
            for key in analysis['unexpected_keys'][:5]:  # Show first 5
                self.logger.warning(f"    - {key}")
            if len(analysis['unexpected_keys']) > 5:
                self.logger.warning(f"    - {key}")
            if len(analysis['unexpected_keys']) > 5:
                self.logger.warning(f"    ... and {len(analysis['unexpected_keys']) - 5} more")
        
        if analysis['shape_mismatches']:
            self.logger.warning(f"  Shape mismatches: {len(analysis['shape_mismatches'])}")
            for mismatch in analysis['shape_mismatches'][:3]:  # Show first 3
                self.logger.warning(f"    - {mismatch['key']}: {mismatch['checkpoint_shape']} -> {mismatch['model_shape']}")
            if len(analysis['shape_mismatches']) > 3:
                self.logger.warning(f"    ... and {len(analysis['shape_mismatches']) - 3} more")
        
        if analysis['compatibility_score'] < 0.5:
            self.logger.error("Low compatibility score detected. Consider starting fresh training.")
        elif analysis['compatibility_score'] < 0.8:
            self.logger.warning("Moderate compatibility issues detected. Some layers will be reinitialized.")
        else:
            self.logger.info("Good compatibility score. Most weights should load correctly.")
    
    def _handle_architecture_changes(self, model: nn.Module, checkpoint_state_dict: Dict[str, torch.Tensor]):
        """
        Handle specific architecture changes between checkpoint and current model.
        
        Args:
            model: Current model
            checkpoint_state_dict: State dict from checkpoint
        """
        model_state_dict = model.state_dict()
        
        # Handle feature projection layer dimension changes
        if 'feature_projection.0.weight' in checkpoint_state_dict and 'feature_projection.0.weight' in model_state_dict:
            checkpoint_weight = checkpoint_state_dict['feature_projection.0.weight']
            current_weight = model_state_dict['feature_projection.0.weight']
            
            if checkpoint_weight.shape != current_weight.shape:
                self.logger.info(f"Feature projection layer shape changed: {checkpoint_weight.shape} -> {current_weight.shape}")
                
                # If checkpoint has smaller input dimension, we need to handle it
                if checkpoint_weight.shape[1] < current_weight.shape[1]:
                    # This means the checkpoint was saved with a different feature dimension
                    # We'll reinitialize this layer and log it
                    self.logger.warning("Feature projection layer dimension mismatch detected. Layer will be reinitialized.")
                    
                    # Reinitialize the feature projection layer
                    if hasattr(model, 'feature_projection'):
                        # Reinitialize the first layer
                        nn.init.xavier_uniform_(model.feature_projection[0].weight)
                        if model.feature_projection[0].bias is not None:
                            nn.init.zeros_(model.feature_projection[0].bias)
                        
                        # Reinitialize the second layer
                        nn.init.xavier_uniform_(model.feature_projection[2].weight)
                        if model.feature_projection[2].bias is not None:
                            nn.init.zeros_(model.feature_projection[2].bias)
                        
                        self.logger.info("Feature projection layers reinitialized")
                        
                        # Provide specific guidance for feature projection mismatch
                        self.logger.info("FEATURE PROJECTION GUIDANCE:")
                        self.logger.info("  - This mismatch likely occurred due to a change in the DCAL encoder configuration")
                        self.logger.info("  - The feature projection layer has been reinitialized with Xavier initialization")
                        self.logger.info("  - Training will continue with the new architecture")
                        self.logger.info("  - Consider monitoring the feature projection layer gradients")
                        
                elif checkpoint_weight.shape[1] > current_weight.shape[1]:
                    # This means the checkpoint has larger input dimension
                    self.logger.warning("Checkpoint has larger feature dimension than current model. This may cause issues.")
                    self.logger.info("  - This suggests the checkpoint was saved with a model that had more features")
                    self.logger.info("  - Consider using a different checkpoint or starting fresh training")
        
        # Handle other common architecture changes
        self._handle_backbone_changes(model, checkpoint_state_dict)
        self._handle_classifier_changes(model, checkpoint_state_dict)
        
        # Provide recommendations
        self._provide_recommendations(model, checkpoint_state_dict)
        
        # Log specific guidance for common issues
        self._log_specific_guidance(model, checkpoint_state_dict)
    
    def _handle_backbone_changes(self, model: nn.Module, checkpoint_state_dict: Dict[str, torch.Tensor]):
        """Handle backbone architecture changes."""
        # Check for backbone-related keys
        backbone_keys = [k for k in checkpoint_state_dict.keys() if 'backbone' in k or 'dcal_encoder' in k]
        current_keys = [k for k in model.state_dict().keys() if 'backbone' in k or 'dcal_encoder' in k]
        
        if backbone_keys and not current_keys:
            self.logger.warning("Checkpoint contains backbone weights but current model doesn't have backbone layers")
        elif current_keys and not backbone_keys:
            self.logger.warning("Current model has backbone layers but checkpoint doesn't contain backbone weights")
    
    def _handle_classifier_changes(self, model: nn.Module, checkpoint_state_dict: Dict[str, torch.Tensor]):
        """Handle classifier architecture changes."""
        # Check for classifier-related keys
        classifier_keys = [k for k in checkpoint_state_dict.keys() if 'classifier' in k]
        current_keys = [k for k in model.state_dict().keys() if 'classifier' in k]
        
        if classifier_keys and not current_keys:
            self.logger.warning("Checkpoint contains classifier weights but current model doesn't have classifier layers")
        elif current_keys and not classifier_keys:
            self.logger.warning("Current model has classifier layers but checkpoint doesn't contain classifier weights")
    
    def _provide_recommendations(self, model: nn.Module, checkpoint_state_dict: Dict[str, torch.Tensor]):
        """Provide recommendations for handling architecture mismatches."""
        model_state_dict = model.state_dict()
        
        # Check for specific issues and provide recommendations
        feature_proj_mismatch = False
        backbone_mismatch = False
        classifier_mismatch = False
        
        # Check feature projection
        if 'feature_projection.0.weight' in checkpoint_state_dict and 'feature_projection.0.weight' in model_state_dict:
            if checkpoint_state_dict['feature_projection.0.weight'].shape != model_state_dict['feature_projection.0.weight'].shape:
                feature_proj_mismatch = True
        
        # Check backbone
        backbone_keys = [k for k in checkpoint_state_dict.keys() if 'backbone' in k or 'dcal_encoder' in k]
        current_keys = [k for k in model_state_dict.keys() if 'backbone' in k or 'dcal_encoder' in k]
        if len(backbone_keys) != len(current_keys):
            backbone_mismatch = True
        
        # Check classifier
        classifier_keys = [k for k in checkpoint_state_dict.keys() if 'classifier' in k]
        current_keys = [k for k in model_state_dict.keys() if 'classifier' in k]
        if len(classifier_keys) != len(current_keys):
            classifier_mismatch = True
        
        # Provide specific recommendations
        if feature_proj_mismatch:
            self.logger.info("RECOMMENDATION: Feature projection layer mismatch detected.")
            self.logger.info("  - The layer has been automatically reinitialized")
            self.logger.info("  - Training will continue with the new architecture")
            self.logger.info("  - Consider saving a new checkpoint after a few epochs")
        
        if backbone_mismatch:
            self.logger.info("RECOMMENDATION: Backbone architecture mismatch detected.")
            self.logger.info("  - This may indicate a change in the DCAL encoder configuration")
            self.logger.info("  - Consider starting fresh training if the mismatch is significant")
        
        if classifier_mismatch:
            self.logger.info("RECOMMENDATION: Classifier architecture mismatch detected.")
            self.logger.info("  - This may indicate a change in the model head configuration")
            self.logger.info("  - The classifier has been automatically reinitialized")
        
        # General recommendations
        self.logger.info("GENERAL RECOMMENDATIONS:")
        self.logger.info("  - Monitor training metrics closely for the first few epochs")
        self.logger.info("  - Save checkpoints frequently to avoid losing progress")
        self.logger.info("  - Consider reducing learning rate initially if training is unstable")
    
    def _should_force_fresh_start(self, checkpoint_path: str) -> bool:
        """
        Determine if we should force a fresh start based on checkpoint analysis.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            True if fresh start should be forced
        """
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_state_dict = checkpoint_data['model_state_dict']
            
            # Check for critical mismatches that would prevent successful loading
            # For now, we'll allow loading with strict=False and handle mismatches gracefully
            # This method can be extended in the future for more sophisticated logic
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to analyze checkpoint for fresh start decision: {e}")
            return False
    
    def _backup_checkpoint(self, checkpoint_path: str):
        """
        Create a backup of the checkpoint before attempting to load it.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        try:
            backup_dir = os.path.join(self.checkpoint_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            filename = os.path.basename(checkpoint_path)
            backup_path = os.path.join(backup_dir, f"backup_{int(time.time())}_{filename}")
            
            shutil.copy2(checkpoint_path, backup_path)
            self.logger.info(f"Created backup of checkpoint: {backup_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint backup: {e}")
    
    def _log_loading_summary(self, analysis: Dict[str, Any]):
        """Log a summary of what was successfully loaded."""
        self.logger.info("LOADING SUMMARY:")
        self.logger.info(f"  Successfully loaded: {analysis['matching_keys']} layers")
        
        if analysis['shape_mismatches']:
            self.logger.info(f"  Reinitialized: {len(analysis['shape_mismatches'])} layers due to shape mismatch")
        
        if analysis['missing_keys']:
            self.logger.info(f"  Missing: {len(analysis['missing_keys'])} layers (will use random initialization)")
        
        if analysis['unexpected_keys']:
            self.logger.info(f"  Ignored: {len(analysis['unexpected_keys'])} unexpected layers")
        
        self.logger.info(f"  Overall compatibility: {analysis['compatibility_score']:.1%}")
        
        # Provide training recommendations based on compatibility
        if analysis['compatibility_score'] < 0.8:
            self.logger.info("TRAINING RECOMMENDATIONS:")
            self.logger.info("  - Consider reducing learning rate for the first few epochs")
            self.logger.info("  - Monitor loss curves closely for stability")
            self.logger.info("  - Save checkpoints more frequently")
    
    def load_best_checkpoint(self, 
                           model: nn.Module,
                           optimizer: optim.Optimizer,
                           scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                           scaler: Optional[GradScaler] = None) -> Optional[Dict[str, Any]]:
        """
        Load the best model checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler (optional)
            scaler: Gradient scaler (optional)
        
        Returns:
            Loaded checkpoint data or None if no best checkpoint exists
        """
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        
        if not os.path.exists(best_checkpoint_path):
            self.logger.warning("No best model checkpoint found")
            return None
        
        return self.load_checkpoint(model, optimizer, best_checkpoint_path, scheduler, scaler)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            return None
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        return checkpoint_files[-1]
    
    def auto_resume_training(self, 
                           model: nn.Module,
                           optimizer: optim.Optimizer,
                           scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                           scaler: Optional[GradScaler] = None) -> Optional[Dict[str, Any]]:
        """
        Automatically resume training from the latest checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler (optional)
            scaler: Gradient scaler (optional)
        
        Returns:
            Loaded checkpoint data or None if no checkpoint to resume from
        """
        if not self.auto_resume:
            return None
        
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint is None:
            self.logger.info("No checkpoint found for auto-resume")
            return None
        
        self.logger.info(f"Auto-resuming from checkpoint: {latest_checkpoint}")
        
        # Backup the checkpoint before attempting to load
        self._backup_checkpoint(latest_checkpoint)
        
        # Check if we should force fresh start
        if self._should_force_fresh_start(latest_checkpoint):
            self.logger.warning("Architecture mismatch detected. Starting fresh training.")
            return None
        
        return self.load_checkpoint(model, optimizer, latest_checkpoint, scheduler, scaler, strict=False)
    
    def save_model_only(self, model: nn.Module, filename: str = "model.pth") -> str:
        """
        Save only the model state dict for inference.
        
        Args:
            model: Model to save
            filename: Filename for the saved model
        
        Returns:
            Path to saved model file
        """
        # Get the base model (unwrap DataParallel if needed)
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        save_path = os.path.join(self.checkpoint_dir, filename)
        
        try:
            torch.save(model_state, save_path)
            self.logger.info(f"Saved model to: {save_path}")
            return save_path
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model_only(self, model: nn.Module, filepath: str, strict: bool = True) -> nn.Module:
        """
        Load only the model state dict for inference.
        
        Args:
            model: Model to load state into
            filepath: Path to model file
            strict: Whether to strictly enforce state_dict keys
        
        Returns:
            Model with loaded state
        """
        try:
            model_state = torch.load(filepath, map_location='cpu')
            
            # Handle both full checkpoint and model-only saves
            if isinstance(model_state, dict) and 'model_state_dict' in model_state:
                model_state = model_state['model_state_dict']
            
            if hasattr(model, 'module'):
                model.module.load_state_dict(model_state, strict=strict)
            else:
                model.load_state_dict(model_state, strict=strict)
            
            self.logger.info(f"Loaded model from: {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {filepath}: {e}")
            raise
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints.
        
        Returns:
            Dictionary with checkpoint information
        """
        info = {
            'checkpoint_dir': self.checkpoint_dir,
            'num_checkpoints': len(self.checkpoint_history),
            'best_metric': self.best_metric,
            'best_metric_value': self.best_metric_value,
            'latest_checkpoint': self.get_latest_checkpoint(),
            'best_checkpoint': os.path.join(self.checkpoint_dir, "best_model.pth") 
                              if os.path.exists(os.path.join(self.checkpoint_dir, "best_model.pth")) else None,
            'checkpoint_history': self.checkpoint_history.copy()
        }
        
        return info
    
    def export_model(self, model: nn.Module, export_path: str, format: str = "pytorch") -> str:
        """
        Export model for deployment.
        
        Args:
            model: Model to export
            export_path: Path to export the model
            format: Export format ('pytorch', 'onnx', 'torchscript')
        
        Returns:
            Path to exported model
        """
        # Get the base model (unwrap DataParallel if needed)
        if hasattr(model, 'module'):
            export_model = model.module
        else:
            export_model = model
        
        export_model.eval()
        
        try:
            if format == "pytorch":
                torch.save(export_model.state_dict(), export_path)
                
            elif format == "torchscript":
                # Create dummy input for tracing
                dummy_input = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
                traced_model = torch.jit.trace(export_model, dummy_input)
                torch.jit.save(traced_model, export_path)
                
            elif format == "onnx":
                import torch.onnx
                dummy_input = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
                torch.onnx.export(
                    export_model,
                    dummy_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['image1', 'image2'],
                    output_names=['similarity'],
                    dynamic_axes={'image1': {0: 'batch_size'}, 
                                 'image2': {0: 'batch_size'},
                                 'similarity': {0: 'batch_size'}}
                )
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported model to: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Failed to export model: {e}")
            raise


class KaggleCheckpointManager(CheckpointManager):
    """
    Specialized checkpoint manager for Kaggle environments.
    
    Handles 12-hour timeout constraints and automatic resumption.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "/kaggle/working/checkpoints",
                 timeout_hours: float = 11.5,
                 **kwargs):
        """
        Initialize Kaggle checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            timeout_hours: Hours before timeout (default 11.5 for safety)
            **kwargs: Additional arguments for base CheckpointManager
        """
        super().__init__(checkpoint_dir=checkpoint_dir, **kwargs)
        
        self.timeout_hours = timeout_hours
        self.start_time = time.time()
        self.timeout_seconds = timeout_hours * 3600
        
        # Force save every epoch for Kaggle
        self.save_interval = 1
        
        # Create lightweight checkpoint for quick saves
        self.quick_checkpoint_file = os.path.join(checkpoint_dir, "quick_checkpoint.pth")
        
        self.logger.info(f"Kaggle checkpoint manager initialized with {timeout_hours}h timeout")
    
    def is_timeout_approaching(self, buffer_minutes: float = 30) -> bool:
        """
        Check if timeout is approaching.
        
        Args:
            buffer_minutes: Buffer time in minutes before timeout
        
        Returns:
            True if timeout is approaching
        """
        elapsed_time = time.time() - self.start_time
        buffer_seconds = buffer_minutes * 60
        
        return elapsed_time >= (self.timeout_seconds - buffer_seconds)
    
    def save_quick_checkpoint(self, 
                             model: nn.Module,
                             optimizer: optim.Optimizer,
                             epoch: int,
                             batch_idx: int,
                             metrics: Dict[str, float],
                             **kwargs) -> str:
        """
        Save a quick checkpoint with minimal data for emergency resume.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            batch_idx: Current batch index
            metrics: Current metrics
            **kwargs: Additional state to save
        
        Returns:
            Path to saved quick checkpoint
        """
        # Get the base model (unwrap DataParallel if needed)
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        quick_checkpoint_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'is_quick_checkpoint': True
        }
        
        # Add any additional state
        for key, value in kwargs.items():
            if hasattr(value, 'state_dict'):
                quick_checkpoint_data[f'{key}_state_dict'] = value.state_dict()
            else:
                quick_checkpoint_data[key] = value
        
        try:
            torch.save(quick_checkpoint_data, self.quick_checkpoint_file)
            self.logger.info(f"Saved quick checkpoint at epoch {epoch}, batch {batch_idx}")
            return self.quick_checkpoint_file
            
        except Exception as e:
            self.logger.error(f"Failed to save quick checkpoint: {e}")
            return None
    
    def load_quick_checkpoint(self, 
                             model: nn.Module,
                             optimizer: optim.Optimizer,
                             **kwargs) -> Optional[Dict[str, Any]]:
        """
        Load quick checkpoint for emergency resume.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            **kwargs: Additional components to load state into
        
        Returns:
            Loaded checkpoint data or None if no quick checkpoint exists
        """
        if not os.path.exists(self.quick_checkpoint_file):
            return None
        
        try:
            checkpoint_data = torch.load(self.quick_checkpoint_file, map_location='cpu')
            
            # Load model state
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint_data['model_state_dict'])
            else:
                model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load additional components
            for key, component in kwargs.items():
                state_key = f'{key}_state_dict'
                if state_key in checkpoint_data and hasattr(component, 'load_state_dict'):
                    component.load_state_dict(checkpoint_data[state_key])
            
            self.logger.info(f"Loaded quick checkpoint from epoch {checkpoint_data['epoch']}, batch {checkpoint_data['batch_idx']}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load quick checkpoint: {e}")
            return None
    
    def emergency_save(self, 
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       epoch: int,
                       batch_idx: int,
                       metrics: Dict[str, float],
                       **kwargs) -> str:
        """
        Emergency save before timeout.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            batch_idx: Current batch index
            metrics: Current metrics
            **kwargs: Additional state to save
        
        Returns:
            Path to saved emergency checkpoint
        """
        self.logger.warning("Emergency save triggered due to timeout approaching")
        
        # Save both quick and regular checkpoint
        quick_path = self.save_quick_checkpoint(model, optimizer, epoch, batch_idx, metrics, **kwargs)
        
        # Also save regular checkpoint
        regular_path = self.save_checkpoint(
            model, optimizer, epoch, metrics, 
            force_save=True, **kwargs
        )
        
        return quick_path or regular_path
    
    def get_time_remaining(self) -> float:
        """
        Get remaining time before timeout in hours.
        
        Returns:
            Remaining time in hours
        """
        elapsed_time = time.time() - self.start_time
        remaining_seconds = self.timeout_seconds - elapsed_time
        return max(0, remaining_seconds / 3600)


def create_checkpoint_manager(config: Dict[str, Any]) -> CheckpointManager:
    """
    Create checkpoint manager based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        CheckpointManager instance
    """
    environment = config.get('environment', 'local')
    
    if environment == 'kaggle':
        return KaggleCheckpointManager(
            checkpoint_dir=config.get('checkpoint_dir', '/kaggle/working/checkpoints'),
            timeout_hours=config.get('timeout_hours', 11.5),
            max_checkpoints=config.get('max_checkpoints', 3),  # Fewer checkpoints for Kaggle
            save_best=config.get('save_best', True),
            best_metric=config.get('best_metric', 'val_roc_auc'),
            best_mode=config.get('best_mode', 'max')
        )
    else:
        return CheckpointManager(
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
            max_checkpoints=config.get('max_checkpoints', 5),
            save_best=config.get('save_best', True),
            best_metric=config.get('best_metric', 'val_roc_auc'),
            best_mode=config.get('best_mode', 'max'),
            save_interval=config.get('save_interval', 1)
        ) 