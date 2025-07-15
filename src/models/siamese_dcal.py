"""
Siamese DCAL model for twin face verification.
Wraps the DCAL core to handle pair-wise verification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy import interpolate

from .dcal_core import DCALEncoder


class SiameseDCAL(nn.Module):
    """
    Siamese DCAL model for twin face verification.
    
    Uses shared DCAL encoder to process image pairs and compute similarity scores.
    """
    
    def __init__(
        self,
        dcal_encoder: DCALEncoder,
        similarity_function: str = 'cosine',
        feature_dim: int = 768,
        dropout: float = 0.1,
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        """
        Initialize Siamese DCAL model.
        
        Args:
            dcal_encoder: Pre-initialized DCAL encoder
            similarity_function: Similarity function ('cosine', 'euclidean', 'learned')
            feature_dim: Dimension of feature vectors
            dropout: Dropout rate for similarity computation
            temperature: Temperature for scaling similarity scores
            learnable_temperature: Whether temperature is learnable
        """
        super(SiameseDCAL, self).__init__()
        
        self.dcal_encoder = dcal_encoder
        self.similarity_function = similarity_function
        self.feature_dim = feature_dim
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(dcal_encoder.embed_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Temperature parameter for scaling
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Learned similarity function
        if similarity_function == 'learned':
            self.similarity_net = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1)
            )
        
        # Classification head (optional)
        self.classifier = nn.Linear(1, 2)  # Binary classification
        
        # Optimal threshold (learned during training)
        self.register_buffer('optimal_threshold', torch.tensor(0.5))
        
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for twin verification.
        
        Args:
            img1: First image tensor [batch_size, 3, H, W]
            img2: Second image tensor [batch_size, 3, H, W]
            return_features: Whether to return feature vectors
            return_attention: Whether to return attention maps
        
        Returns:
            Dictionary containing similarity scores and optional features/attention
        """
        batch_size = img1.size(0)
        
        # Process images through DCAL encoder
        output1 = self.dcal_encoder(img1, return_attention=return_attention)
        output2 = self.dcal_encoder(img2, return_attention=return_attention)
        
        # Extract features from SA+GLCA (inference mode)
        features1 = output1['features']  # [batch_size, embed_dim]
        features2 = output2['features']  # [batch_size, embed_dim]
        
        # Project features
        features1 = self.feature_projection(features1)
        features2 = self.feature_projection(features2)
        
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # Compute similarity
        similarity = self._compute_similarity(features1, features2)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Classification logits
        logits = self.classifier(similarity.unsqueeze(1))
        
        # Prepare output
        output = {
            'similarity': similarity,
            'logits': logits,
            'features1': features1,
            'features2': features2
        }
        
        if return_features:
            output['raw_features1'] = output1['features']
            output['raw_features2'] = output2['features']
        
        if return_attention:
            output['attention1'] = output1.get('attention_maps', {})
            output['attention2'] = output2.get('attention_maps', {})
        
        return output
    
    def _compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between feature vectors.
        
        Args:
            features1: First feature vectors [batch_size, feature_dim]
            features2: Second feature vectors [batch_size, feature_dim]
        
        Returns:
            Similarity scores [batch_size]
        """
        if self.similarity_function == 'cosine':
            similarity = F.cosine_similarity(features1, features2, dim=1)
        elif self.similarity_function == 'euclidean':
            # Convert euclidean distance to similarity
            distance = F.pairwise_distance(features1, features2)
            similarity = 1 / (1 + distance)
        elif self.similarity_function == 'learned':
            # Concatenate features and use learned similarity
            combined = torch.cat([features1, features2], dim=1)
            similarity = self.similarity_net(combined).squeeze(1)
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")
        
        return similarity
    
    def predict(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict whether two images are the same person.
        
        Args:
            img1: First image tensor
            img2: Second image tensor  
            threshold: Classification threshold (uses optimal if None)
        
        Returns:
            Dictionary with predictions and scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(img1, img2)
            
            similarity = output['similarity']
            
            if threshold is None:
                threshold = self.optimal_threshold.item()
            
            predictions = (similarity >= threshold).long()
            
            return {
                'predictions': predictions,
                'similarity': similarity,
                'threshold': threshold,
                'logits': output['logits']
            }
    
    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single image.
        
        Args:
            img: Image tensor [batch_size, 3, H, W]
        
        Returns:
            Normalized feature vectors [batch_size, feature_dim]
        """
        self.eval()
        with torch.no_grad():
            output = self.dcal_encoder(img, return_attention=False)
            features = self.feature_projection(output['features'])
            features = F.normalize(features, p=2, dim=1)
            return features
    
    def compute_similarity_matrix(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between two sets of features.
        
        Args:
            features1: First set of features [N, feature_dim]
            features2: Second set of features [M, feature_dim]
        
        Returns:
            Similarity matrix [N, M]
        """
        if self.similarity_function == 'cosine':
            similarity_matrix = torch.mm(features1, features2.t())
        elif self.similarity_function == 'euclidean':
            # Compute pairwise euclidean distances
            distances = torch.cdist(features1, features2, p=2)
            similarity_matrix = 1 / (1 + distances)
        else:
            # For learned similarity, compute pairwise
            N, M = features1.size(0), features2.size(0)
            similarity_matrix = torch.zeros(N, M, device=features1.device)
            
            for i in range(N):
                for j in range(M):
                    feat1 = features1[i:i+1]
                    feat2 = features2[j:j+1]
                    similarity_matrix[i, j] = self._compute_similarity(feat1, feat2)
        
        return similarity_matrix
    
    def update_optimal_threshold(self, labels: torch.Tensor, similarities: torch.Tensor):
        """
        Update optimal threshold based on validation data.
        
        Args:
            labels: Ground truth labels [N]
            similarities: Similarity scores [N]
        """
        # Convert to numpy for sklearn
        labels_np = labels.detach().cpu().numpy()
        similarities_np = similarities.detach().cpu().numpy()
        
        # Calculate EER threshold
        fpr, tpr, thresholds = roc_curve(labels_np, similarities_np)
        fnr = 1 - tpr
        eer_threshold = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, fnr)(x), 0., 1.)
        threshold = interpolate.interp1d(fpr, thresholds)(eer_threshold)
        
        self.optimal_threshold.data = torch.tensor(threshold, dtype=self.optimal_threshold.dtype)
    
    def get_attention_maps(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
        
        Returns:
            Dictionary of attention maps
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(img1, img2, return_attention=True)
            
            attention_maps = {
                'attention1': output['attention1'],
                'attention2': output['attention2']
            }
            
            return attention_maps


class TwinVerificationHead(nn.Module):
    """
    Verification head for twin face verification.
    Can be used with different backbone architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize verification head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(TwinVerificationHead, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Second layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through verification head.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Output features [batch_size, output_dim]
        """
        return self.layers(x)


class AdaptiveSimilarityLearner(nn.Module):
    """
    Adaptive similarity learner that adjusts similarity computation based on input.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        """
        Initialize adaptive similarity learner.
        
        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
        """
        super(AdaptiveSimilarityLearner, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Multi-head attention for adaptive similarity
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        # Similarity computation
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive similarity between features.
        
        Args:
            features1: First feature vectors [batch_size, feature_dim]
            features2: Second feature vectors [batch_size, feature_dim]
        
        Returns:
            Similarity scores [batch_size]
        """
        batch_size = features1.size(0)
        
        # Apply attention to features
        q1 = self.query_projection(features1)
        k1 = self.key_projection(features1)
        v1 = self.value_projection(features1)
        
        q2 = self.query_projection(features2)
        k2 = self.key_projection(features2)
        v2 = self.value_projection(features2)
        
        # Reshape for multi-head attention
        q1 = q1.view(batch_size, self.num_heads, self.head_dim)
        k1 = k1.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)
        
        q2 = q2.view(batch_size, self.num_heads, self.head_dim)
        k2 = k2.view(batch_size, self.num_heads, self.head_dim)
        v2 = v2.view(batch_size, self.num_heads, self.head_dim)
        
        # Cross-attention between features
        attention_scores = torch.einsum('bhi,bhj->bhij', q1, k2)
        attention_weights = F.softmax(attention_scores / (self.head_dim ** 0.5), dim=-1)
        
        # Apply attention to values
        attended_v2 = torch.einsum('bhij,bhj->bhi', attention_weights, v2)
        attended_features1 = attended_v2.contiguous().view(batch_size, self.feature_dim)
        
        # Reverse attention
        attention_scores_rev = torch.einsum('bhi,bhj->bhij', q2, k1)
        attention_weights_rev = F.softmax(attention_scores_rev / (self.head_dim ** 0.5), dim=-1)
        attended_v1 = torch.einsum('bhij,bhj->bhi', attention_weights_rev, v1)
        attended_features2 = attended_v1.contiguous().view(batch_size, self.feature_dim)
        
        # Combine attended features
        combined_features = torch.cat([attended_features1, attended_features2], dim=1)
        
        # Compute similarity
        similarity = self.similarity_net(combined_features).squeeze(1)
        
        return similarity 