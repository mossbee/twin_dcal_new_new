"""
Experiment tracking system for twin face verification.
Supports MLFlow, WandB, and no-tracking modes.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import numpy as np
import torch

# Optional imports for different tracking backends
try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


class BaseTracker(ABC):
    """Base class for experiment tracking."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.is_active = False
    
    @abstractmethod
    def start_run(self, config: Dict[str, Any]):
        """Start a new run."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        pass
    
    @abstractmethod
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts."""
        pass
    
    @abstractmethod
    def end_run(self):
        """End the current run."""
        pass


class MLFlowTracker(BaseTracker):
    """MLFlow experiment tracker."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None, 
                 tracking_uri: Optional[str] = None):
        super().__init__(experiment_name, run_name)
        
        if not HAS_MLFLOW:
            raise ImportError("MLFlow not installed. Install with: pip install mlflow")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id
        except Exception as e:
            logging.warning(f"Failed to set MLFlow experiment: {e}")
            self.experiment_id = None
    
    def start_run(self, config: Dict[str, Any]):
        """Start MLFlow run."""
        if self.experiment_id:
            mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id)
        else:
            mlflow.start_run(run_name=self.run_name)
        
        self.is_active = True
        
        # Log configuration
        self.log_params(config)
        
        logging.info(f"MLFlow run started: {self.run_name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLFlow."""
        if not self.is_active:
            return
        
        try:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    mlflow.log_metric(name, value, step=step)
        except Exception as e:
            logging.warning(f"Failed to log metrics to MLFlow: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLFlow."""
        if not self.is_active:
            return
        
        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)
            
            for name, value in flat_params.items():
                # Convert value to string if needed
                if not isinstance(value, (str, int, float, bool)):
                    value = str(value)
                mlflow.log_param(name, value)
        except Exception as e:
            logging.warning(f"Failed to log parameters to MLFlow: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts to MLFlow."""
        if not self.is_active:
            return
        
        try:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    mlflow.log_artifact(path, artifact_path=name)
        except Exception as e:
            logging.warning(f"Failed to log artifacts to MLFlow: {e}")
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Log PyTorch model to MLFlow."""
        if not self.is_active:
            return
        
        try:
            mlflow.pytorch.log_model(model, model_name)
        except Exception as e:
            logging.warning(f"Failed to log model to MLFlow: {e}")
    
    def end_run(self):
        """End MLFlow run."""
        if self.is_active:
            mlflow.end_run()
            self.is_active = False
            logging.info(f"MLFlow run ended: {self.run_name}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class WandBTracker(BaseTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None,
                 entity: Optional[str] = None, project: Optional[str] = None):
        super().__init__(experiment_name, run_name)
        
        if not HAS_WANDB:
            raise ImportError("WandB not installed. Install with: pip install wandb")
        
        self.entity = entity
        self.project = project or experiment_name
        self.run = None
    
    def start_run(self, config: Dict[str, Any]):
        """Start WandB run."""
        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=config,
                reinit=True
            )
            self.is_active = True
            logging.info(f"WandB run started: {self.run_name}")
        except Exception as e:
            logging.warning(f"Failed to start WandB run: {e}")
            self.is_active = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to WandB."""
        if not self.is_active or not self.run:
            return
        
        try:
            # Filter out invalid values
            filtered_metrics = {}
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    filtered_metrics[name] = value
            
            if filtered_metrics:
                if step is not None:
                    self.run.log(filtered_metrics, step=step)
                else:
                    self.run.log(filtered_metrics)
        except Exception as e:
            logging.warning(f"Failed to log metrics to WandB: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to WandB."""
        if not self.is_active or not self.run:
            return
        
        try:
            # WandB config is set during init, but we can update it
            self.run.config.update(params)
        except Exception as e:
            logging.warning(f"Failed to log parameters to WandB: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts to WandB."""
        if not self.is_active or not self.run:
            return
        
        try:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    self.run.log_artifact(path, name=name)
        except Exception as e:
            logging.warning(f"Failed to log artifacts to WandB: {e}")
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Log PyTorch model to WandB."""
        if not self.is_active or not self.run:
            return
        
        try:
            # Save model to temporary file and log as artifact
            temp_path = f"/tmp/{model_name}.pth"
            torch.save(model.state_dict(), temp_path)
            self.run.log_artifact(temp_path, name=model_name)
            os.remove(temp_path)
        except Exception as e:
            logging.warning(f"Failed to log model to WandB: {e}")
    
    def end_run(self):
        """End WandB run."""
        if self.is_active and self.run:
            try:
                self.run.finish()
                self.is_active = False
                logging.info(f"WandB run ended: {self.run_name}")
            except Exception as e:
                logging.warning(f"Failed to end WandB run: {e}")


class NoTracker(BaseTracker):
    """No-op tracker for when tracking is disabled."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        super().__init__(experiment_name, run_name)
        self.metrics_history = []
        self.params = {}
        self.artifacts = {}
    
    def start_run(self, config: Dict[str, Any]):
        """Start no-op run."""
        self.is_active = True
        self.params = config.copy()
        logging.info(f"No-tracking run started: {self.run_name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to memory."""
        if not self.is_active:
            return
        
        metric_entry = {
            'step': step,
            'timestamp': time.time(),
            'metrics': metrics.copy()
        }
        self.metrics_history.append(metric_entry)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to memory."""
        if not self.is_active:
            return
        
        self.params.update(params)
    
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts to memory."""
        if not self.is_active:
            return
        
        self.artifacts.update(artifacts)
    
    def end_run(self):
        """End no-op run."""
        if self.is_active:
            self.is_active = False
            logging.info(f"No-tracking run ended: {self.run_name}")
    
    def save_history(self, save_path: str):
        """Save tracking history to file."""
        history = {
            'run_name': self.run_name,
            'params': self.params,
            'metrics_history': self.metrics_history,
            'artifacts': self.artifacts
        }
        
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)


class ExperimentTracker:
    """
    Unified experiment tracker that supports multiple backends.
    """
    
    def __init__(self, 
                 tracker_type: str = "mlflow",
                 experiment_name: str = "twin_face_verification",
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment tracker.
        
        Args:
            tracker_type: Type of tracker ('mlflow', 'wandb', 'none')
            experiment_name: Name of the experiment
            run_name: Name of the run (optional)
            config: Configuration dict for the tracker
        """
        self.tracker_type = tracker_type
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.config = config or {}
        
        # Initialize tracker based on type
        if tracker_type == "mlflow":
            self.tracker = MLFlowTracker(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=self.config.get('mlflow_tracking_uri')
            )
        elif tracker_type == "wandb":
            self.tracker = WandBTracker(
                experiment_name=experiment_name,
                run_name=run_name,
                entity=self.config.get('wandb_entity'),
                project=self.config.get('wandb_project')
            )
        elif tracker_type == "none":
            self.tracker = NoTracker(
                experiment_name=experiment_name,
                run_name=run_name
            )
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
        
        self.is_active = False
    
    def start_run(self, config: Dict[str, Any]):
        """Start tracking run."""
        self.tracker.start_run(config)
        self.is_active = True
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.is_active:
            self.tracker.log_metrics(metrics, step)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.is_active:
            self.tracker.log_params(params)
    
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts."""
        if self.is_active:
            self.tracker.log_artifacts(artifacts)
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Log model."""
        if self.is_active and hasattr(self.tracker, 'log_model'):
            self.tracker.log_model(model, model_name)
    
    def end_run(self):
        """End tracking run."""
        if self.is_active:
            self.tracker.end_run()
            self.is_active = False
    
    def save_history(self, save_path: str):
        """Save tracking history (for NoTracker)."""
        if isinstance(self.tracker, NoTracker):
            self.tracker.save_history(save_path)


def create_tracker(config: Dict[str, Any]) -> ExperimentTracker:
    """
    Create experiment tracker from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        ExperimentTracker instance
    """
    tracker_type = config.get('tracking_type', 'mlflow')
    experiment_name = config.get('experiment_name', 'twin_face_verification')
    run_name = config.get('run_name')
    
    # Environment-specific configuration
    if config.get('environment') == 'kaggle':
        # Kaggle uses WandB by default
        if tracker_type == 'wandb' and 'WANDB_API_KEY' in os.environ:
            config['wandb_entity'] = config.get('wandb_entity', 'hunchoquavodb-hanoi-university-of-science-and-technology')
        elif tracker_type == 'mlflow':
            # MLFlow not available on Kaggle, switch to no tracking
            tracker_type = 'none'
    
    tracker = ExperimentTracker(
        tracker_type=tracker_type,
        experiment_name=experiment_name,
        run_name=run_name,
        config=config
    )
    
    return tracker


class MetricsLogger:
    """
    Helper class for logging metrics with automatic formatting.
    """
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.step = 0
    
    def log_training_metrics(self, metrics: Dict[str, float], epoch: int, batch_idx: int = None):
        """Log training metrics with proper prefixes."""
        prefixed_metrics = {}
        
        for name, value in metrics.items():
            if batch_idx is not None:
                prefixed_metrics[f'train/batch_{name}'] = value
            else:
                prefixed_metrics[f'train/{name}'] = value
        
        self.tracker.log_metrics(prefixed_metrics, step=self.step)
        self.step += 1
    
    def log_validation_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log validation metrics with proper prefixes."""
        prefixed_metrics = {}
        
        for name, value in metrics.items():
            prefixed_metrics[f'val/{name}'] = value
        
        self.tracker.log_metrics(prefixed_metrics, step=epoch)
    
    def log_test_metrics(self, metrics: Dict[str, float]):
        """Log test metrics with proper prefixes."""
        prefixed_metrics = {}
        
        for name, value in metrics.items():
            prefixed_metrics[f'test/{name}'] = value
        
        self.tracker.log_metrics(prefixed_metrics)
    
    def log_epoch_summary(self, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], epoch: int):
        """Log epoch summary with both training and validation metrics."""
        summary_metrics = {}
        
        # Training metrics
        for name, value in train_metrics.items():
            summary_metrics[f'epoch/train_{name}'] = value
        
        # Validation metrics
        for name, value in val_metrics.items():
            summary_metrics[f'epoch/val_{name}'] = value
        
        self.tracker.log_metrics(summary_metrics, step=epoch)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Logger instance
    """
    log_level = config.get('log_level', 'INFO')
    log_format = config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logger
    logger = logging.getLogger('twin_face_verification')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if config.get('log_file'):
        file_handler = logging.FileHandler(config['log_file'])
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger 