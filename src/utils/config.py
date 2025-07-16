"""
Configuration management system for DCAL Twin Faces Verification.

This module provides a flexible configuration system that supports:
- YAML-based configuration files
- Environment-specific configurations
- Hyperparameter validation
- Dynamic configuration updates
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "vit_base_patch16_224"
    num_classes: int = 2
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    
    # Pre-trained model settings
    pretrained: bool = True
    pretrained_path: Optional[str] = None


@dataclass
class DCALConfig:
    """DCAL-specific configuration."""
    num_sa_blocks: int = 12
    num_glca_blocks: int = 1
    num_pwca_blocks: int = 12
    local_ratio_fgvc: float = 0.1
    local_ratio_reid: float = 0.3
    
    # Attention mechanism settings
    attention_dropout: float = 0.1
    use_attention_rollout: bool = True
    
    # Dynamic loss weights
    use_dynamic_loss: bool = True
    loss_weight_sa: float = 1.0
    loss_weight_glca: float = 1.0
    loss_weight_pwca: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 5e-4
    epochs: int = 100
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # cosine, step, plateau
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Optimization settings
    optimizer: str = "adam"  # adam, sgd, adamw
    momentum: float = 0.9
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Mixed precision training
    use_amp: bool = True
    
    # Validation settings
    val_interval: int = 1
    val_metric: str = "roc_auc"  # roc_auc, eer, accuracy


@dataclass
class DataConfig:
    """Data configuration."""
    image_size: int = 224
    patch_size: int = 16
    train_split: float = 0.8
    val_split: float = 0.2
    augmentation: bool = True
    
    # Dataset paths
    data_dir: str = "./data"
    train_info_file: str = "train_dataset_infor.json"
    test_info_file: str = "test_dataset_infor.json"
    train_pairs_file: str = "train_twin_pairs.json"
    test_pairs_file: str = "test_twin_pairs.json"
    
    # Pair sampling configuration
    positive_ratio: float = 0.5
    hard_negative_ratio: float = 0.3
    soft_negative_ratio: float = 0.2
    use_only_hard_negatives: bool = True  # Use only hard negatives (twins) for training
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation settings
    horizontal_flip: bool = True
    color_jitter: bool = True
    random_rotation: float = 10.0
    normalize_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Debug mode
    debug: bool = False


@dataclass
class SystemConfig:
    """System and environment configuration."""
    device: str = "cuda"
    num_gpus: int = 1
    distributed: bool = False
    
    # Tracking and logging
    tracking: str = "none"  # mlflow, wandb, none
    experiment_name: str = "dcal_twin_verification"
    run_name: Optional[str] = None
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1
    max_checkpoints: int = 5
    resume_from: Optional[str] = None
    
    # Kaggle-specific settings
    timeout_hours: Optional[int] = None
    auto_resume: bool = True
    
    # Random seed
    seed: int = 42


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dcal: DCALConfig = field(default_factory=DCALConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate image and patch sizes
        if self.data.image_size % self.data.patch_size != 0:
            raise ValueError(f"Image size {self.data.image_size} must be divisible by patch size {self.data.patch_size}")
        
        # Validate batch size
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate learning rate
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate data splits
        if not (0 < self.data.train_split < 1):
            raise ValueError("Train split must be between 0 and 1")
        
        if not (0 < self.data.val_split < 1):
            raise ValueError("Validation split must be between 0 and 1")
        
        # Validate pair sampling ratios
        total_ratio = (self.data.positive_ratio + 
                      self.data.hard_negative_ratio + 
                      self.data.soft_negative_ratio)
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError("Pair sampling ratios must sum to 1.0")
        
        # Validate DCAL configuration
        if self.dcal.num_sa_blocks <= 0:
            raise ValueError("Number of SA blocks must be positive")
        
        if self.dcal.num_glca_blocks <= 0:
            raise ValueError("Number of GLCA blocks must be positive")
        
        if self.dcal.num_pwca_blocks <= 0:
            raise ValueError("Number of PWCA blocks must be positive")
        
        # Validate model configuration
        if self.model.embed_dim % self.model.num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        
        # Validate system configuration
        if self.system.num_gpus <= 0:
            raise ValueError("Number of GPUs must be positive")
        
        if self.system.tracking not in ["mlflow", "wandb", "none"]:
            raise ValueError("Tracking must be one of: mlflow, wandb, none")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "dcal": self.dcal.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "system": self.system.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            dcal=DCALConfig(**config_dict.get("dcal", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            system=SystemConfig(**config_dict.get("system", {}))
        )
    
    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for section, values in updates.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        raise ValueError(f"Unknown configuration key: {section}.{key}")
            else:
                raise ValueError(f"Unknown configuration section: {section}")
        
        # Re-validate after update
        self._validate()


def get_config(config_path: Optional[str] = None, 
               env_overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Get configuration with optional file loading and environment overrides.
    
    Args:
        config_path: Path to configuration file
        env_overrides: Dictionary of environment-specific overrides
        
    Returns:
        Config: Configuration object
    """
    # Start with default configuration
    if config_path and os.path.exists(config_path):
        config = Config.load(config_path)
    else:
        config = Config()
    
    # Apply environment overrides
    if env_overrides:
        config.update(env_overrides)
    
    # Apply environment variables
    _apply_env_variables(config)
    
    return config


def _apply_env_variables(config: Config):
    """Apply environment variables to configuration."""
    # System environment variables
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        config.system.num_gpus = len([x for x in gpu_ids if x.strip()])
    
    # Tracking environment variables
    if "MLFLOW_TRACKING_URI" in os.environ:
        config.system.tracking = "mlflow"
    
    if "WANDB_API_KEY" in os.environ:
        config.system.tracking = "wandb"
    
    # Data directory
    if "DCAL_DATA_DIR" in os.environ:
        config.data.data_dir = os.environ["DCAL_DATA_DIR"]
    
    # Checkpoint directory
    if "DCAL_CHECKPOINT_DIR" in os.environ:
        config.system.checkpoint_dir = os.environ["DCAL_CHECKPOINT_DIR"]


def create_default_configs():
    """Create default configuration files for different environments."""
    # Base configuration
    base_config = Config()
    base_config.save("configs/base_config.yaml")
    
    # Local server configuration
    local_config = Config()
    local_config.system.num_gpus = 2
    local_config.system.tracking = "mlflow"
    local_config.system.checkpoint_dir = "./checkpoints"
    local_config.data.data_dir = "./data"
    local_config.training.batch_size = 16
    local_config.save("configs/local_config.yaml")
    
    # Kaggle configuration
    kaggle_config = Config()
    kaggle_config.system.num_gpus = 1
    kaggle_config.system.tracking = "wandb"
    kaggle_config.system.checkpoint_dir = "/kaggle/working/checkpoints"
    kaggle_config.data.data_dir = "/kaggle/input/nd-twin"
    kaggle_config.system.timeout_hours = 11
    kaggle_config.system.auto_resume = True
    kaggle_config.training.batch_size = 8
    kaggle_config.save("configs/kaggle_config.yaml")
    
    print("Default configuration files created successfully!")


if __name__ == "__main__":
    # Create default configuration files
    create_default_configs() 