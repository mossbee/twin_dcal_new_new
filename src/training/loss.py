"""
Loss functions for twin face verification using DCAL.
Implements uncertainty loss, verification losses, and multi-task coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class UncertaintyLoss(nn.Module):
    """
    Uncertainty loss for multi-task learning from FairMOT.
    
    Learns uncertainty weights for different tasks automatically,
    balancing multiple objectives during training.
    """
    
    def __init__(self, num_tasks: int = 2, init_weight: float = 1.0):
        """
        Initialize uncertainty loss.
        
        Args:
            num_tasks: Number of tasks to balance
            init_weight: Initial weight for tasks
        """
        super(UncertaintyLoss, self).__init__()
        self.num_tasks = num_tasks
        
        # Learnable log variance parameters
        self.log_vars = nn.Parameter(torch.ones(num_tasks) * np.log(init_weight))
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted loss.
        
        Args:
            losses: Dictionary of task losses
        
        Returns:
            Tuple of (total_loss, weights_dict)
        """
        loss_list = list(losses.values())
        total_loss = 0
        weights = {}
        
        for i, (task_name, loss) in enumerate(losses.items()):
            # Calculate uncertainty weight: 1 / (2 * sigma^2)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            
            total_loss += weighted_loss
            weights[f'{task_name}_weight'] = precision.item()
            weights[f'{task_name}_log_var'] = self.log_vars[i].item()
        
        return total_loss, weights


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for twin face verification.
    
    Minimizes distance for positive pairs and maximizes distance for negative pairs.
    """
    
    def __init__(self, margin: float = 1.0, distance_fn: str = 'cosine'):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
            distance_fn: Distance function ('cosine', 'euclidean')
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            labels: Binary labels (1 for same person, 0 for different)
        
        Returns:
            Contrastive loss
        """
        # Calculate distance
        if self.distance_fn == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarity = F.cosine_similarity(features1, features2, dim=1)
            distance = 1 - similarity
        else:
            # Euclidean distance
            distance = F.pairwise_distance(features1, features2, keepdim=True).squeeze()
        
        # Positive pairs: minimize distance
        positive_loss = labels * torch.pow(distance, 2)
        
        # Negative pairs: maximize distance up to margin
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for twin face verification.
    
    Ensures anchor-positive distance is smaller than anchor-negative distance by a margin.
    """
    
    def __init__(self, margin: float = 0.2, distance_fn: str = 'cosine'):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin between positive and negative distances
            distance_fn: Distance function ('cosine', 'euclidean')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor features
            positive: Positive features
            negative: Negative features
        
        Returns:
            Triplet loss
        """
        # Calculate distances
        if self.distance_fn == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
            neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=1)
        else:
            pos_dist = F.pairwise_distance(anchor, positive, keepdim=True).squeeze()
            neg_dist = F.pairwise_distance(anchor, negative, keepdim=True).squeeze()
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Focuses learning on hard examples by down-weighting easy examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
        
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class VerificationLoss(nn.Module):
    """
    Combined verification loss for twin face verification.
    
    Combines multiple loss functions for robust training.
    """
    
    def __init__(
        self,
        use_contrastive: bool = True,
        use_triplet: bool = False,
        use_focal: bool = False,
        contrastive_margin: float = 1.0,
        triplet_margin: float = 0.2,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        loss_weights: Dict[str, float] = None
    ):
        """
        Initialize verification loss.
        
        Args:
            use_contrastive: Whether to use contrastive loss
            use_triplet: Whether to use triplet loss
            use_focal: Whether to use focal loss
            contrastive_margin: Margin for contrastive loss
            triplet_margin: Margin for triplet loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            loss_weights: Weights for different loss components
        """
        super(VerificationLoss, self).__init__()
        
        self.use_contrastive = use_contrastive
        self.use_triplet = use_triplet
        self.use_focal = use_focal
        
        # Initialize loss functions
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        
        if use_triplet:
            self.triplet_loss = TripletLoss(margin=triplet_margin)
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Loss weights
        self.loss_weights = loss_weights or {
            'contrastive': 1.0,
            'triplet': 0.5,
            'focal': 0.3
        }
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute verification loss.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            labels: Binary labels
            logits: Classification logits (optional)
        
        Returns:
            Tuple of (total_loss, loss_components)
        """
        losses = {}
        total_loss = 0
        
        # Contrastive loss
        if self.use_contrastive:
            contrastive_loss = self.contrastive_loss(features1, features2, labels)
            losses['contrastive'] = contrastive_loss
            total_loss += self.loss_weights['contrastive'] * contrastive_loss
        
        # Triplet loss (requires special handling for triplet formation)
        if self.use_triplet:
            # Create triplets from pairs
            triplet_loss = self._compute_triplet_loss(features1, features2, labels)
            losses['triplet'] = triplet_loss
            total_loss += self.loss_weights['triplet'] * triplet_loss
        
        # Focal loss (requires logits)
        if self.use_focal and logits is not None:
            focal_loss = self.focal_loss(logits, labels)
            losses['focal'] = focal_loss
            total_loss += self.loss_weights['focal'] * focal_loss
        
        return total_loss, losses
    
    def _compute_triplet_loss(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss from pairs.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            labels: Binary labels
        
        Returns:
            Triplet loss
        """
        # Find positive and negative pairs
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        if not positive_mask.any() or not negative_mask.any():
            return torch.tensor(0.0, device=features1.device)
        
        # Use positive pairs as anchor-positive, negative pairs as anchor-negative
        positive_indices = torch.where(positive_mask)[0]
        negative_indices = torch.where(negative_mask)[0]
        
        # Sample triplets
        num_triplets = min(len(positive_indices), len(negative_indices))
        if num_triplets == 0:
            return torch.tensor(0.0, device=features1.device)
        
        # Random sampling for triplets
        pos_idx = positive_indices[:num_triplets]
        neg_idx = negative_indices[:num_triplets]
        
        # Create triplets: anchor=features1[pos], positive=features2[pos], negative=features1[neg]
        anchor = features1[pos_idx]
        positive = features2[pos_idx]
        negative = features1[neg_idx]
        
        return self.triplet_loss(anchor, positive, negative)


class DCALLoss(nn.Module):
    """
    DCAL-specific loss that combines verification loss with attention regularization.
    """
    
    def __init__(
        self,
        verification_loss: VerificationLoss,
        attention_reg_weight: float = 0.01,
        diversity_weight: float = 0.1
    ):
        """
        Initialize DCAL loss.
        
        Args:
            verification_loss: Base verification loss
            attention_reg_weight: Weight for attention regularization
            diversity_weight: Weight for attention diversity loss
        """
        super(DCALLoss, self).__init__()
        self.verification_loss = verification_loss
        self.attention_reg_weight = attention_reg_weight
        self.diversity_weight = diversity_weight
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: torch.Tensor,
        attention_maps: Optional[Dict[str, torch.Tensor]] = None,
        logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DCAL loss.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            labels: Binary labels
            attention_maps: Dictionary of attention maps
            logits: Classification logits
        
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Base verification loss
        verification_loss, loss_components = self.verification_loss(
            features1, features2, labels, logits
        )
        
        total_loss = verification_loss
        
        # Attention regularization
        if attention_maps is not None:
            # Attention sparsity regularization
            attention_reg = self._compute_attention_regularization(attention_maps)
            loss_components['attention_reg'] = attention_reg
            total_loss += self.attention_reg_weight * attention_reg
            
            # Attention diversity loss
            diversity_loss = self._compute_diversity_loss(attention_maps)
            loss_components['diversity'] = diversity_loss
            total_loss += self.diversity_weight * diversity_loss
        
        return total_loss, loss_components
    
    def _compute_attention_regularization(self, attention_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute attention regularization loss.
        
        Args:
            attention_maps: Dictionary of attention maps
        
        Returns:
            Attention regularization loss
        """
        reg_loss = 0
        count = 0

        if len(attention_maps) == 0:
            print("[DEBUG] attention_maps is empty in _compute_attention_regularization!")
        
        for key, attention_map_list in attention_maps.items():
            if isinstance(attention_map_list, list):
                for attention_map in attention_map_list:
                    reg_loss += torch.mean(torch.abs(attention_map))
                    count += 1
            elif isinstance(attention_map_list, torch.Tensor):
                reg_loss += torch.mean(torch.abs(attention_map_list))
                count += 1
        
        if count > 0:
            return reg_loss / count
        else:
            # Defensive: avoid IndexError if attention_maps is empty
            device = torch.device("cpu")
            if len(attention_maps) > 0:
                # Try to get device from first tensor in first list
                first = next(iter(attention_maps.values()))
                if isinstance(first, list) and len(first) > 0:
                    device = first[0].device
                elif isinstance(first, torch.Tensor):
                    device = first.device
            print("[DEBUG] Returning zero attention regularization loss on device:", device)
            return torch.tensor(0.0, device=device)
    
    def _compute_diversity_loss(self, attention_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute attention diversity loss to encourage different attention patterns.
        
        Args:
            attention_maps: Dictionary of attention maps
        
        Returns:
            Diversity loss
        """
        if len(attention_maps) < 2:
            print("[DEBUG] attention_maps is empty or has only one entry in _compute_diversity_loss!")
            device = torch.device("cpu")
            if len(attention_maps) > 0:
                device = list(attention_maps.values())[0].device
            print("[DEBUG] Returning zero diversity loss on device:", device)
            return torch.tensor(0.0, device=device)
        
        diversity_loss = 0
        count = 0
        
        attention_list = list(attention_maps.values())
        for i in range(len(attention_list)):
            for j in range(i + 1, len(attention_list)):
                # Encourage different attention patterns
                similarity = F.cosine_similarity(
                    attention_list[i].flatten(1),
                    attention_list[j].flatten(1),
                    dim=1
                )
                diversity_loss += torch.mean(similarity)
                count += 1
        
        if count > 0:
            return diversity_loss / count
        else:
            device = attention_list[0].device
            print("[DEBUG] Returning zero diversity loss (no pairs) on device:", device)
            return torch.tensor(0.0, device=device) 