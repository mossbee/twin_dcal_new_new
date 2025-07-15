"""
Vision Transformer backbone implementation for DCAL Twin Faces Verification.

This module provides a flexible ViT backbone with:
- Patch embedding for different image sizes
- Positional encoding (learnable/fixed)
- Support for pre-trained models
- Multi-scale support (224x224, 448x448)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
from functools import partial


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts images to patch tokens.
    
    Args:
        img_size: Input image size (height, width)
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding using convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Validate input size
        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}x{W} doesn't match expected size {self.img_size}x{self.img_size}"
        
        # Extract patches and embed
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for patch embeddings.
    
    Args:
        num_patches: Number of patches
        embed_dim: Embedding dimension
        learnable: Whether to use learnable positional embeddings
        dropout: Dropout rate
    """
    
    def __init__(self, 
                 num_patches: int,
                 embed_dim: int,
                 learnable: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.learnable = learnable
        
        if learnable:
            # Learnable positional embeddings
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        else:
            # Fixed sinusoidal positional embeddings
            self.register_buffer('pos_embed', self._get_sinusoidal_encoding(num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_sinusoidal_encoding(self, seq_len: int, embed_dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encodings."""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        
        pos_embed = torch.zeros(seq_len, embed_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embed.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Input embeddings (B, seq_len, embed_dim)
            
        Returns:
            Embeddings with positional encoding
        """
        x = x + self.pos_embed
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, seq_len, embed_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (B, seq_len, embed_dim)
            Optionally attention weights (B, num_heads, seq_len, seq_len)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x


class FeedForward(nn.Module):
    """
    Feed-forward network for transformer blocks.
    
    Args:
        embed_dim: Input embedding dimension
        mlp_ratio: Hidden dimension ratio
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(self, 
                 embed_dim: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        drop_path: Drop path rate
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout)
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Forward pass."""
        # Self-attention with residual connection
        attn_output = self.attn(self.norm1(x), return_attention=return_attention)
        
        if return_attention:
            attn_output, attn_weights = attn_output
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(attn_output)
        
        # Feed-forward with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if return_attention:
            return x, attn_weights
        return x


class DropPath(nn.Module):
    """Drop path (stochastic depth) regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformer(nn.Module):
    """
    Vision Transformer backbone.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        drop_path_rate: Drop path rate
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.0,
                 pretrained: bool = True,
                 pretrained_path: Optional[str] = None):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional encoding
        self.pos_embed = PositionalEncoding(
            num_patches=self.patch_embed.num_patches,
            embed_dim=embed_dim,
            learnable=True,
            dropout=dropout
        )
        
        # Transformer blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path_rates[i]
            ) for i in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained(self, pretrained_path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Filter out incompatible keys
            model_dict = self.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
            
            print(f"Loaded pretrained weights from {pretrained_path}")
            print(f"Loaded {len(filtered_dict)}/{len(state_dict)} parameters")
            
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            Output features (B, seq_len, embed_dim)
            Optionally attention weights from all layers
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional encoding
        x = self.pos_embed(x)
        
        # Transformer blocks
        attention_weights = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        if return_attention:
            return x, attention_weights
        return x
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get features from different layers.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dictionary of features from different layers
        """
        features = {}
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        features['patch_embed'] = x
        
        # Add class token and positional encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_embed(x)
        features['pos_embed'] = x
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            features[f'block_{i}'] = x
        
        # Final layer norm
        x = self.norm(x)
        features['final'] = x
        
        return features


def create_vit_model(config_name: str = "vit_base_patch16_224", **kwargs) -> VisionTransformer:
    """
    Create a Vision Transformer model with predefined configurations.
    
    Args:
        config_name: Model configuration name
        **kwargs: Additional arguments to override defaults
        
    Returns:
        VisionTransformer model
    """
    configs = {
        "vit_tiny_patch16_224": {
            "img_size": 224, "patch_size": 16, "embed_dim": 192,
            "num_layers": 12, "num_heads": 3, "mlp_ratio": 4.0
        },
        "vit_small_patch16_224": {
            "img_size": 224, "patch_size": 16, "embed_dim": 384,
            "num_layers": 12, "num_heads": 6, "mlp_ratio": 4.0
        },
        "vit_base_patch16_224": {
            "img_size": 224, "patch_size": 16, "embed_dim": 768,
            "num_layers": 12, "num_heads": 12, "mlp_ratio": 4.0
        },
        "vit_large_patch16_224": {
            "img_size": 224, "patch_size": 16, "embed_dim": 1024,
            "num_layers": 24, "num_heads": 16, "mlp_ratio": 4.0
        },
        "vit_base_patch16_448": {
            "img_size": 448, "patch_size": 16, "embed_dim": 768,
            "num_layers": 12, "num_heads": 12, "mlp_ratio": 4.0
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = configs[config_name]
    config.update(kwargs)
    
    return VisionTransformer(**config)


if __name__ == "__main__":
    # Test the model
    model = create_vit_model("vit_base_patch16_224")
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    features = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    
    # Test with attention
    features, attention = model(x, return_attention=True)
    print(f"Attention weights shape: {len(attention)} layers")
    print(f"Each attention shape: {attention[0].shape}")
    
    # Test feature extraction
    feature_dict = model.get_features(x)
    print(f"Available features: {list(feature_dict.keys())}")
    
    print("Vision Transformer backbone test completed successfully!") 