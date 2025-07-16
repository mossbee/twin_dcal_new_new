"""
Core DCAL model implementation for Twin Faces Verification.

This module implements the main DCAL architecture that coordinates:
- Self-Attention (SA) branch
- Global-Local Cross-Attention (GLCA) branch 
- Pair-Wise Cross-Attention (PWCA) branch

Features:
- Multi-task learning coordination
- Dynamic loss weight learning (from FairMOT)
- Inference mode (SA + GLCA only)
- Flexible backbone integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from .backbone import VisionTransformer, create_vit_model
from .attention_blocks import DCALTransformerBlock, AttentionRollout


class DynamicLossWeights(nn.Module):
    """
    Dynamic loss weight learning based on FairMOT uncertainty loss.
    
    This implements the uncertainty-based loss weighting:
    L = 0.5 * (exp(-w1) * L1 + exp(-w2) * L2 + w1 + w2)
    
    where w1, w2 are learnable parameters.
    """
    
    def __init__(self, num_branches: int = 3, init_weights: Optional[List[float]] = None):
        super().__init__()
        
        self.num_branches = num_branches
        
        # Initialize learnable parameters
        if init_weights is None:
            # Default initialization from FairMOT
            init_weights = [-1.85, -1.05, -1.0][:num_branches]
        
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.tensor(w, dtype=torch.float32)) 
            for w in init_weights
        ])
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            losses: List of loss values for each branch
            
        Returns:
            Weighted combined loss
        """
        assert len(losses) == self.num_branches, f"Expected {self.num_branches} losses, got {len(losses)}"
        
        # Compute uncertainty-weighted loss
        weighted_losses = []
        regularization = 0.0
        
        for i, loss in enumerate(losses):
            if loss is not None:
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss
                weighted_losses.append(weighted_loss)
                regularization += self.log_vars[i]
        
        # Combine losses
        total_loss = sum(weighted_losses) + regularization
        return 0.5 * total_loss


class DCALEncoder(nn.Module):
    """
    DCAL encoder that implements the core architecture.
    
    This combines the Vision Transformer backbone with DCAL attention blocks.
    """
    
    def __init__(self,
                 backbone_config: str = "vit_base_patch16_224",
                 num_sa_blocks: int = 12,
                 num_glca_blocks: int = 1,
                 num_pwca_blocks: int = 12,
                 local_ratio: float = 0.1,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.0,
                 use_dynamic_loss: bool = True,
                 **backbone_kwargs):
        super().__init__()
        
        self.num_sa_blocks = num_sa_blocks
        self.num_glca_blocks = num_glca_blocks
        self.num_pwca_blocks = num_pwca_blocks
        self.local_ratio = local_ratio
        self.embed_dim = embed_dim
        self.use_dynamic_loss = use_dynamic_loss
        
        # Vision Transformer backbone
        # Filter out DCAL-specific keys from kwargs to avoid passing them to create_vit_model
        dcal_keys = {'backbone', 'local_ratio_fgvc', 'local_ratio_reid', 'num_sa_blocks', 
                     'num_glca_blocks', 'num_pwca_blocks', 'use_dynamic_loss'}
        filtered_kwargs = {k: v for k, v in backbone_kwargs.items() if k not in dcal_keys}
        self.backbone = create_vit_model(backbone_config, **filtered_kwargs)
        
        # DCAL transformer blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_sa_blocks)]
        
        self.dcal_blocks = nn.ModuleList([
            DCALTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path_rates[i],
                local_ratio=local_ratio,
                use_glca=(i < num_glca_blocks),  # Only first M blocks use GLCA
                use_pwca=(i < num_pwca_blocks)   # Only first T blocks use PWCA
            ) for i in range(num_sa_blocks)
        ])
        
        # Dynamic loss weights
        if use_dynamic_loss:
            self.loss_weights = DynamicLossWeights(num_branches=3)
        
        # Classification heads for each branch
        self.sa_classifier = nn.Linear(embed_dim, embed_dim)
        self.glca_classifier = nn.Linear(embed_dim, embed_dim) 
        self.pwca_classifier = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, 
                x: torch.Tensor,
                x_pair: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DCAL encoder.
        
        Args:
            x: Input image tensor (B, C, H, W)
            x_pair: Paired image for PWCA (B, C, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing outputs from different branches
        """
        # Vision Transformer backbone
        x = self.backbone(x, return_attention=return_attention)
        
        if return_attention:
            x, backbone_attention = x
        
        # Process paired image if provided
        if x_pair is not None:
            x_pair = self.backbone(x_pair)
            if return_attention:
                x_pair = x_pair[0]  # Only take features, ignore attention
        
        # Initialize results
        results = {
            'sa_features': [],
            'glca_features': [],
            'pwca_features': [],
            'sa_attention': [] if return_attention else None,
            'glca_attention': [] if return_attention else None,
            'pwca_attention': [] if return_attention else None
        }
        
        # Track attention history for rollout
        attention_history = []
        
        # Process through DCAL blocks
        for i, block in enumerate(self.dcal_blocks):
            # Use backbone attention for first block if available
            if i == 0 and return_attention and 'backbone_attention' in locals():
                current_attention_history = [backbone_attention[-1]]  # Use last backbone layer
            else:
                current_attention_history = attention_history if i > 0 else None
            
            block_outputs = block(
                x, 
                x_pair=x_pair,
                attention_history=current_attention_history,
                return_attention=return_attention
            )
            
            # Update main feature
            if 'sa_output' in block_outputs:
                x = block_outputs['sa_output']
                results['sa_features'].append(x)
            
            # Store branch outputs
            if 'glca_output' in block_outputs:
                results['glca_features'].append(block_outputs['glca_output'])
            
            if 'pwca_output' in block_outputs:
                results['pwca_features'].append(block_outputs['pwca_output'])
            
            # Store attention weights
            if return_attention:
                if 'sa_attention' in block_outputs:
                    results['sa_attention'].append(block_outputs['sa_attention'])
                    attention_history.append(block_outputs['sa_attention'])
                
                if 'glca_attention' in block_outputs:
                    results['glca_attention'].append(block_outputs['glca_attention'])
                
                if 'pwca_attention' in block_outputs:
                    results['pwca_attention'].append(block_outputs['pwca_attention'])
        
        # Get final features from each branch
        results['sa_final'] = results['sa_features'][-1] if results['sa_features'] else x
        results['glca_final'] = results['glca_features'][-1] if results['glca_features'] else None
        results['pwca_final'] = results['pwca_features'][-1] if results['pwca_features'] else None
        
        # Apply classification heads
        results['sa_logits'] = self.sa_classifier(results['sa_final'][:, 0])  # CLS token
        
        if results['glca_final'] is not None:
            results['glca_logits'] = self.glca_classifier(results['glca_final'][:, 0])
        
        if results['pwca_final'] is not None:
            results['pwca_logits'] = self.pwca_classifier(results['pwca_final'][:, 0])
        
        return results
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features for inference (SA + GLCA only).
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Combined features tensor
        """
        outputs = self.forward(x, x_pair=None, return_attention=False)
        
        # Combine SA and GLCA features
        sa_features = outputs['sa_logits']
        
        if outputs.get('glca_logits') is not None:
            glca_features = outputs['glca_logits']
            # Concatenate features
            combined_features = torch.cat([sa_features, glca_features], dim=1)
        else:
            # If no GLCA, duplicate SA features to maintain consistent dimensions
            combined_features = torch.cat([sa_features, sa_features], dim=1)
        
        return combined_features
    
    def compute_loss(self, 
                     outputs: Dict[str, torch.Tensor],
                     targets: torch.Tensor,
                     criterion: nn.Module) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            criterion: Loss criterion
            
        Returns:
            Total loss and loss components
        """
        losses = {}
        loss_values = []
        
        # SA loss
        if 'sa_logits' in outputs:
            sa_loss = criterion(outputs['sa_logits'], targets)
            losses['sa_loss'] = sa_loss
            loss_values.append(sa_loss)
        else:
            loss_values.append(None)
        
        # GLCA loss
        if 'glca_logits' in outputs:
            glca_loss = criterion(outputs['glca_logits'], targets)
            losses['glca_loss'] = glca_loss
            loss_values.append(glca_loss)
        else:
            loss_values.append(None)
        
        # PWCA loss (same target as SA - distractor learning)
        if 'pwca_logits' in outputs:
            pwca_loss = criterion(outputs['pwca_logits'], targets)
            losses['pwca_loss'] = pwca_loss
            loss_values.append(pwca_loss)
        else:
            loss_values.append(None)
        
        # Combine losses
        if self.use_dynamic_loss:
            total_loss = self.loss_weights(loss_values)
        else:
            # Fixed weights
            total_loss = 0.0
            if losses.get('sa_loss') is not None:
                total_loss += losses['sa_loss']
            if losses.get('glca_loss') is not None:
                total_loss += losses['glca_loss']
            if losses.get('pwca_loss') is not None:
                total_loss += 0.1 * losses['pwca_loss']  # Smaller weight for PWCA
        
        losses['total_loss'] = total_loss
        return total_loss, losses


class DCALModel(nn.Module):
    """
    Main DCAL model for twin faces verification.
    
    This wraps the DCAL encoder and provides task-specific functionality.
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 backbone_config: str = "vit_base_patch16_224",
                 num_sa_blocks: int = 12,
                 num_glca_blocks: int = 1,
                 num_pwca_blocks: int = 12,
                 local_ratio: float = 0.1,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.0,
                 use_dynamic_loss: bool = True,
                 **backbone_kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # DCAL encoder
        self.encoder = DCALEncoder(
            backbone_config=backbone_config,
            num_sa_blocks=num_sa_blocks,
            num_glca_blocks=num_glca_blocks,
            num_pwca_blocks=num_pwca_blocks,
            local_ratio=local_ratio,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            use_dynamic_loss=use_dynamic_loss,
            **backbone_kwargs
        )
        
        # Task-specific heads
        self.classifier = nn.Linear(embed_dim * 2, num_classes)  # SA + GLCA features
        self.feature_dim = embed_dim * 2
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                x: torch.Tensor,
                x_pair: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, C, H, W)
            x_pair: Paired image for PWCA (B, C, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            Model outputs
        """
        # Encode features
        encoder_outputs = self.encoder(x, x_pair, return_attention)
        
        # Get inference features (SA + GLCA)
        features = self.encoder.get_features(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Combine outputs
        outputs = {
            'logits': logits,
            'features': features,
            **encoder_outputs
        }
        
        return outputs
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features for inference."""
        return self.encoder.get_features(x)
    
    def compute_loss(self, 
                     outputs: Dict[str, torch.Tensor],
                     targets: torch.Tensor,
                     criterion: nn.Module) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss."""
        return self.encoder.compute_loss(outputs, targets, criterion)


def create_dcal_model(config: Any) -> DCALModel:
    """
    Create DCAL model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        DCAL model instance
    """
    model = DCALModel(
        num_classes=config.model.num_classes,
        backbone_config=config.model.backbone,
        num_sa_blocks=config.dcal.num_sa_blocks,
        num_glca_blocks=config.dcal.num_glca_blocks,
        num_pwca_blocks=config.dcal.num_pwca_blocks,
        local_ratio=config.dcal.local_ratio_fgvc,  # Use appropriate ratio
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        mlp_ratio=4.0,
        dropout=config.model.dropout,
        drop_path_rate=0.1,
        use_dynamic_loss=config.dcal.use_dynamic_loss,
        img_size=config.data.image_size,
        patch_size=config.data.patch_size,
        pretrained=config.model.pretrained,
        pretrained_path=config.model.pretrained_path
    )
    
    return model


if __name__ == "__main__":
    # Test the DCAL model
    from ..utils.config import Config
    
    # Create test configuration
    config = Config()
    
    # Create model
    model = create_dcal_model(config)
    
    # Test data
    batch_size = 2
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    print("Testing forward pass...")
    model.train()
    outputs = model(x1, x_pair=x2, return_attention=True)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Available outputs: {list(outputs.keys())}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    criterion = nn.CrossEntropyLoss()
    total_loss, loss_components = model.compute_loss(outputs, targets, criterion)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {list(loss_components.keys())}")
    
    # Test inference mode
    print("\nTesting inference mode...")
    model.eval()
    with torch.no_grad():
        inference_outputs = model(x1)
        features = model.get_features(x1)
    
    print(f"Inference logits shape: {inference_outputs['logits'].shape}")
    print(f"Inference features shape: {features.shape}")
    
    # Test attention visualization
    print("\nTesting attention visualization...")
    model.eval()
    with torch.no_grad():
        attention_outputs = model(x1, return_attention=True)
        
        if attention_outputs['sa_attention']:
            print(f"SA attention layers: {len(attention_outputs['sa_attention'])}")
            print(f"SA attention shape: {attention_outputs['sa_attention'][0].shape}")
        
        if attention_outputs['glca_attention']:
            print(f"GLCA attention layers: {len(attention_outputs['glca_attention'])}")
    
    print("\nDCAL model test completed successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}") 