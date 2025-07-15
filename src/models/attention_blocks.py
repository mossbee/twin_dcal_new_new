"""
Attention blocks implementation for DCAL Twin Faces Verification.

This module implements the core attention mechanisms:
- Multi-Head Self-Attention (MSA) - Standard transformer attention
- Global-Local Cross-Attention (GLCA) - For local discriminative features
- Pair-Wise Cross-Attention (PWCA) - For regularization during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math


class AttentionRollout:
    """
    Attention rollout implementation for computing accumulated attention scores.
    
    This is used for GLCA to select high-response local queries.
    """
    
    @staticmethod
    def compute_rollout(attention_weights: List[torch.Tensor], 
                       add_residual: bool = True) -> torch.Tensor:
        """
        Compute attention rollout across layers.
        
        Args:
            attention_weights: List of attention weights from each layer
                Each tensor: (B, num_heads, seq_len, seq_len)
            add_residual: Whether to add residual connections
            
        Returns:
            Accumulated attention weights: (B, seq_len, seq_len)
        """
        B, num_heads, seq_len, _ = attention_weights[0].shape
        
        # Average attention heads
        rolled_attention = []
        for attn in attention_weights:
            # Average over heads
            attn_avg = attn.mean(dim=1)  # (B, seq_len, seq_len)
            
            if add_residual:
                # Add residual connection: 0.5 * attention + 0.5 * identity
                identity = torch.eye(seq_len, device=attn.device, dtype=attn.dtype).unsqueeze(0)
                attn_avg = 0.5 * attn_avg + 0.5 * identity
                
                # Re-normalize
                attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            
            rolled_attention.append(attn_avg)
        
        # Compute cumulative attention rollout
        result = rolled_attention[0]
        for i in range(1, len(rolled_attention)):
            result = torch.matmul(rolled_attention[i], result)
        
        return result
    
    @staticmethod
    def select_top_queries(attention_rollout: torch.Tensor, 
                          queries: torch.Tensor, 
                          ratio: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-R% queries based on attention rollout.
        
        Args:
            attention_rollout: Accumulated attention weights (B, seq_len, seq_len)
            queries: Query tensor (B, seq_len, embed_dim)
            ratio: Ratio of queries to select
            
        Returns:
            Selected queries: (B, num_selected, embed_dim)
            Selection indices: (B, num_selected)
        """
        B, seq_len, embed_dim = queries.shape
        
        # Use CLS token attention (first row) to select important patches
        cls_attention = attention_rollout[:, 0, 1:]  # (B, seq_len-1) - exclude CLS token
        
        # Select top-R% patches
        num_selected = max(1, int(ratio * (seq_len - 1)))
        _, top_indices = torch.topk(cls_attention, num_selected, dim=-1)
        
        # Add 1 to indices to account for CLS token
        top_indices = top_indices + 1
        
        # Select queries
        batch_indices = torch.arange(B, device=queries.device).unsqueeze(1)
        selected_queries = queries[batch_indices, top_indices]  # (B, num_selected, embed_dim)
        
        return selected_queries, top_indices


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MSA) block.
    
    This is the standard transformer self-attention mechanism.
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
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
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


class GlobalLocalCrossAttention(nn.Module):
    """
    Global-Local Cross-Attention (GLCA) block.
    
    This mechanism emphasizes the interaction between global images and 
    local high-response regions by:
    1. Using attention rollout to find high-response regions
    2. Selecting top-R% queries based on accumulated attention
    3. Computing cross-attention between local queries and global key-values
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 local_ratio: float = 0.1,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.local_ratio = local_ratio
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Separate projections for local queries and global key-values
        self.q_local = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_global = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                attention_rollout: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, seq_len, embed_dim)
            attention_rollout: Accumulated attention weights (B, seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (B, seq_len, embed_dim)
            Optionally attention weights
        """
        B, N, C = x.shape
        
        # Select top-R% local queries
        local_queries, local_indices = AttentionRollout.select_top_queries(
            attention_rollout, x, self.local_ratio
        )
        num_local = local_queries.shape[1]
        
        # Generate local queries and global key-values
        q_local = self.q_local(local_queries).reshape(B, num_local, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_global = self.kv_global(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_global, v_global = kv_global.unbind(0)  # Each: (B, num_heads, seq_len, head_dim)
        
        # Compute cross-attention: local queries x global key-values
        attn = (q_local @ k_global.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to global values
        local_output = (attn @ v_global).transpose(1, 2).reshape(B, num_local, C)
        local_output = self.proj(local_output)
        local_output = self.proj_dropout(local_output)
        
        # Create full output tensor
        output = torch.zeros_like(x)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
        output[batch_indices, local_indices] = local_output
        
        # Copy non-selected positions from input (residual)
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask[batch_indices, local_indices] = False
        output[mask] = x[mask]
        
        if return_attention:
            return output, attn
        return output


class PairWiseCrossAttention(nn.Module):
    """
    Pair-Wise Cross-Attention (PWCA) block.
    
    This mechanism provides regularization during training by:
    1. Taking query from target image and key-values from both images
    2. Concatenating key-values from image pair
    3. Computing contaminated attention scores
    
    Only used during training - removed during inference.
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
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections (shared with SA - same weights)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass for pair-wise cross-attention.
        
        Args:
            x1: Target image features (B, seq_len, embed_dim)
            x2: Distractor image features (B, seq_len, embed_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (B, seq_len, embed_dim)
            Optionally attention weights
        """
        B, N, C = x1.shape
        
        # Generate Q, K, V for both images
        qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q1, k1, v1 = qkv1.unbind(0)  # Each: (B, num_heads, seq_len, head_dim)
        q2, k2, v2 = qkv2.unbind(0)  # Each: (B, num_heads, seq_len, head_dim)
        
        # Concatenate key-values from both images
        k_combined = torch.cat([k1, k2], dim=2)  # (B, num_heads, 2*seq_len, head_dim)
        v_combined = torch.cat([v1, v2], dim=2)  # (B, num_heads, 2*seq_len, head_dim)
        
        # Compute attention: query from target image x combined key-values
        attn = (q1 @ k_combined.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to combined values
        x = (attn @ v_combined).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x


class DCALTransformerBlock(nn.Module):
    """
    DCAL Transformer block that combines SA, GLCA, and PWCA.
    
    This block coordinates the three attention mechanisms:
    - SA: Standard self-attention
    - GLCA: Global-local cross-attention (separate weights)
    - PWCA: Pair-wise cross-attention (shares weights with SA)
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path: float = 0.0,
                 local_ratio: float = 0.1,
                 use_glca: bool = True,
                 use_pwca: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_glca = use_glca
        self.use_pwca = use_pwca
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Self-attention branch
        self.sa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Global-local cross-attention branch (separate weights)
        if use_glca:
            self.glca = GlobalLocalCrossAttention(embed_dim, num_heads, local_ratio, dropout)
        
        # Pair-wise cross-attention branch (shares weights with SA)
        if use_pwca:
            self.pwca = PairWiseCrossAttention(embed_dim, num_heads, dropout)
            # Share weights with SA
            self.pwca.qkv = self.sa.qkv
            self.pwca.proj = self.sa.proj
        
        # Feed-forward network
        self.mlp = self._build_mlp(embed_dim, mlp_ratio, dropout)
        
        # Drop path
        self.drop_path = self._build_drop_path(drop_path)
        
    def _build_mlp(self, embed_dim: int, mlp_ratio: float, dropout: float) -> nn.Module:
        """Build MLP block."""
        hidden_dim = int(embed_dim * mlp_ratio)
        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def _build_drop_path(self, drop_path: float) -> nn.Module:
        """Build drop path block."""
        if drop_path > 0.0:
            return DropPath(drop_path)
        return nn.Identity()
        
    def forward(self, 
                x: torch.Tensor, 
                x_pair: Optional[torch.Tensor] = None,
                attention_history: Optional[List[torch.Tensor]] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, seq_len, embed_dim)
            x_pair: Paired image for PWCA (B, seq_len, embed_dim)
            attention_history: Previous attention weights for rollout
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing outputs from different branches
        """
        results = {}
        
        # Self-attention branch
        x_norm = self.norm1(x)
        sa_output = self.sa(x_norm, return_attention=return_attention)
        
        if return_attention:
            sa_output, sa_attention = sa_output
            results['sa_attention'] = sa_attention
        
        sa_output = x + self.drop_path(sa_output)
        results['sa_output'] = sa_output + self.drop_path(self.mlp(self.norm2(sa_output)))
        
        # Global-local cross-attention branch
        if self.use_glca:
            if attention_history is not None:
                # Compute attention rollout
                attention_rollout = AttentionRollout.compute_rollout(attention_history)
            else:
                # Create identity attention rollout as fallback
                B, N, _ = x.shape
                attention_rollout = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)
            
            glca_output = self.glca(x_norm, attention_rollout, return_attention=return_attention)
            
            if return_attention:
                glca_output, glca_attention = glca_output
                results['glca_attention'] = glca_attention
            
            glca_output = x + self.drop_path(glca_output)
            results['glca_output'] = glca_output + self.drop_path(self.mlp(self.norm2(glca_output)))
        
        # Pair-wise cross-attention branch (training only)
        if self.use_pwca and x_pair is not None and self.training:
            pwca_output = self.pwca(x_norm, x_pair, return_attention=return_attention)
            
            if return_attention:
                pwca_output, pwca_attention = pwca_output
                results['pwca_attention'] = pwca_attention
            
            pwca_output = x + self.drop_path(pwca_output)
            results['pwca_output'] = pwca_output + self.drop_path(self.mlp(self.norm2(pwca_output)))
        
        return results


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


if __name__ == "__main__":
    # Test the attention mechanisms
    batch_size = 2
    seq_len = 197  # 196 patches + 1 CLS token for 224x224 image
    embed_dim = 768
    num_heads = 12
    
    # Create test tensors
    x1 = torch.randn(batch_size, seq_len, embed_dim)
    x2 = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test Multi-Head Self-Attention
    print("Testing Multi-Head Self-Attention...")
    msa = MultiHeadSelfAttention(embed_dim, num_heads)
    sa_output, sa_attention = msa(x1, return_attention=True)
    print(f"SA output shape: {sa_output.shape}")
    print(f"SA attention shape: {sa_attention.shape}")
    
    # Test Global-Local Cross-Attention
    print("\nTesting Global-Local Cross-Attention...")
    glca = GlobalLocalCrossAttention(embed_dim, num_heads, local_ratio=0.1)
    
    # Create fake attention rollout
    attention_history = [sa_attention]
    attention_rollout = AttentionRollout.compute_rollout(attention_history)
    
    glca_output = glca(x1, attention_rollout)
    print(f"GLCA output shape: {glca_output.shape}")
    
    # Test Pair-Wise Cross-Attention
    print("\nTesting Pair-Wise Cross-Attention...")
    pwca = PairWiseCrossAttention(embed_dim, num_heads)
    pwca_output, pwca_attention = pwca(x1, x2, return_attention=True)
    print(f"PWCA output shape: {pwca_output.shape}")
    print(f"PWCA attention shape: {pwca_attention.shape}")
    
    # Test DCAL Transformer Block
    print("\nTesting DCAL Transformer Block...")
    dcal_block = DCALTransformerBlock(embed_dim, num_heads, use_glca=True, use_pwca=True)
    dcal_block.train()  # Enable training mode for PWCA
    
    results = dcal_block(x1, x_pair=x2, attention_history=attention_history, return_attention=True)
    print(f"DCAL block results: {list(results.keys())}")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\nAll attention mechanisms tested successfully!") 