"""
Pair sampler for balanced training with positive and negative pairs.
Implements intelligent sampling strategies for twin face verification.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Sampler
import torch


class TwinPairSampler(Sampler):
    """
    Sampler for balanced positive/negative pair sampling.
    
    Ensures balanced batches with specified ratios of positive and negative pairs,
    including hard negatives (different twins) and soft negatives (random pairs).
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        positive_ratio: float = 0.5,
        hard_negative_ratio: float = 0.3,
        soft_negative_ratio: float = 0.2,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize the pair sampler.
        
        Args:
            dataset: TwinDataset instance
            batch_size: Batch size for training
            positive_ratio: Ratio of positive pairs in batch
            hard_negative_ratio: Ratio of hard negative pairs in batch
            soft_negative_ratio: Ratio of soft negative pairs in batch
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.soft_negative_ratio = soft_negative_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Calculate batch composition
        self.positive_per_batch = int(batch_size * positive_ratio)
        self.hard_negative_per_batch = int(batch_size * hard_negative_ratio)
        self.soft_negative_per_batch = int(batch_size * soft_negative_ratio)
        self.remaining_per_batch = batch_size - (
            self.positive_per_batch + self.hard_negative_per_batch + self.soft_negative_per_batch
        )
        
        # Organize indices by type
        self._organize_indices()
        
        # Calculate number of batches
        self.num_batches = len(self.positive_indices) // self.positive_per_batch
        if not self.drop_last and len(self.positive_indices) % self.positive_per_batch > 0:
            self.num_batches += 1
    
    def _organize_indices(self):
        """Organize dataset indices by pair type."""
        self.positive_indices = []
        self.hard_negative_indices = []
        self.soft_negative_indices = []
        
        for idx, (_, _, label) in enumerate(self.dataset.pairs):
            if label == 1:
                self.positive_indices.append(idx)
            else:
                # Determine if it's hard or soft negative based on twin relationship
                img1_path, img2_path, _ = self.dataset.pairs[idx]
                
                # Extract person IDs from paths
                person1 = self._extract_person_id(img1_path)
                person2 = self._extract_person_id(img2_path)
                
                # Check if both persons are in twin pairs
                if (person1 in self.dataset.person_to_twin and 
                    person2 in self.dataset.person_to_twin and
                    person1 != person2):
                    self.hard_negative_indices.append(idx)
                else:
                    self.soft_negative_indices.append(idx)
    
    def _extract_person_id(self, image_path: str) -> str:
        """Extract person ID from image path."""
        # Handle different path formats
        if "/kaggle/input/nd-twin/ND_TWIN_Dataset_224/" in image_path:
            relative_path = image_path.split("/kaggle/input/nd-twin/ND_TWIN_Dataset_224/")[1]
            return relative_path.split("/")[0]
        else:
            # Assume format: person_id/image_name
            return image_path.split("/")[-2] if "/" in image_path else image_path.split("\\")[-2]
    
    def __iter__(self):
        """Generate batches with balanced sampling."""
        if self.shuffle:
            random.shuffle(self.positive_indices)
            random.shuffle(self.hard_negative_indices)
            random.shuffle(self.soft_negative_indices)
        
        # Create index iterators
        positive_iter = iter(self.positive_indices)
        hard_negative_iter = iter(self.hard_negative_indices)
        soft_negative_iter = iter(self.soft_negative_indices)
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add positive pairs
            for _ in range(self.positive_per_batch):
                try:
                    batch_indices.append(next(positive_iter))
                except StopIteration:
                    if self.drop_last:
                        return
                    # Reset iterator if not dropping last batch
                    positive_iter = iter(self.positive_indices)
                    batch_indices.append(next(positive_iter))
            
            # Add hard negative pairs
            for _ in range(self.hard_negative_per_batch):
                try:
                    batch_indices.append(next(hard_negative_iter))
                except StopIteration:
                    # Use soft negatives if hard negatives are exhausted
                    try:
                        batch_indices.append(next(soft_negative_iter))
                    except StopIteration:
                        soft_negative_iter = iter(self.soft_negative_indices)
                        batch_indices.append(next(soft_negative_iter))
            
            # Add soft negative pairs
            for _ in range(self.soft_negative_per_batch):
                try:
                    batch_indices.append(next(soft_negative_iter))
                except StopIteration:
                    # Use hard negatives if soft negatives are exhausted
                    try:
                        batch_indices.append(next(hard_negative_iter))
                    except StopIteration:
                        hard_negative_iter = iter(self.hard_negative_indices)
                        batch_indices.append(next(hard_negative_iter))
            
            # Fill remaining slots with random negatives
            remaining_negatives = self.hard_negative_indices + self.soft_negative_indices
            for _ in range(self.remaining_per_batch):
                if remaining_negatives:
                    batch_indices.append(random.choice(remaining_negatives))
            
            # Shuffle batch indices
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield from batch_indices
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_batches * self.batch_size


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler that ensures equal representation of positive and negative pairs.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize balanced batch sampler.
        
        Args:
            dataset: TwinDataset instance
            batch_size: Batch size (must be even)
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop the last incomplete batch
        """
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced sampling")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Split batch size equally
        self.pairs_per_type = batch_size // 2
        
        # Organize indices
        self.positive_indices = []
        self.negative_indices = []
        
        for idx, (_, _, label) in enumerate(self.dataset.pairs):
            if label == 1:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
        
        # Calculate number of batches
        self.num_batches = min(
            len(self.positive_indices) // self.pairs_per_type,
            len(self.negative_indices) // self.pairs_per_type
        )
        
        if not self.drop_last:
            self.num_batches = max(
                len(self.positive_indices) // self.pairs_per_type,
                len(self.negative_indices) // self.pairs_per_type
            )
    
    def __iter__(self):
        """Generate balanced batches."""
        if self.shuffle:
            random.shuffle(self.positive_indices)
            random.shuffle(self.negative_indices)
        
        positive_iter = iter(self.positive_indices)
        negative_iter = iter(self.negative_indices)
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add positive pairs
            for _ in range(self.pairs_per_type):
                try:
                    batch_indices.append(next(positive_iter))
                except StopIteration:
                    if self.drop_last:
                        return
                    positive_iter = iter(self.positive_indices)
                    batch_indices.append(next(positive_iter))
            
            # Add negative pairs
            for _ in range(self.pairs_per_type):
                try:
                    batch_indices.append(next(negative_iter))
                except StopIteration:
                    if self.drop_last:
                        return
                    negative_iter = iter(self.negative_indices)
                    batch_indices.append(next(negative_iter))
            
            # Shuffle batch indices
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield from batch_indices
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_batches * self.batch_size


class AdaptivePairSampler(Sampler):
    """
    Adaptive sampler that adjusts sampling ratios based on training progress.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        initial_positive_ratio: float = 0.5,
        min_positive_ratio: float = 0.3,
        max_positive_ratio: float = 0.7,
        adaptation_factor: float = 0.1,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize adaptive sampler.
        
        Args:
            dataset: TwinDataset instance
            batch_size: Batch size
            initial_positive_ratio: Initial ratio of positive pairs
            min_positive_ratio: Minimum positive ratio
            max_positive_ratio: Maximum positive ratio
            adaptation_factor: How much to adjust ratios
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = initial_positive_ratio
        self.min_positive_ratio = min_positive_ratio
        self.max_positive_ratio = max_positive_ratio
        self.adaptation_factor = adaptation_factor
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Organize indices
        self.positive_indices = []
        self.negative_indices = []
        
        for idx, (_, _, label) in enumerate(self.dataset.pairs):
            if label == 1:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
    
    def adapt_ratios(self, positive_accuracy: float, negative_accuracy: float):
        """
        Adapt sampling ratios based on training performance.
        
        Args:
            positive_accuracy: Accuracy on positive pairs
            negative_accuracy: Accuracy on negative pairs
        """
        if positive_accuracy < negative_accuracy:
            # Increase positive ratio if positive accuracy is lower
            self.positive_ratio = min(
                self.max_positive_ratio,
                self.positive_ratio + self.adaptation_factor
            )
        else:
            # Decrease positive ratio if negative accuracy is lower
            self.positive_ratio = max(
                self.min_positive_ratio,
                self.positive_ratio - self.adaptation_factor
            )
    
    def __iter__(self):
        """Generate adaptive batches."""
        if self.shuffle:
            random.shuffle(self.positive_indices)
            random.shuffle(self.negative_indices)
        
        positive_per_batch = int(self.batch_size * self.positive_ratio)
        negative_per_batch = self.batch_size - positive_per_batch
        
        positive_iter = iter(self.positive_indices)
        negative_iter = iter(self.negative_indices)
        
        num_batches = max(
            len(self.positive_indices) // positive_per_batch,
            len(self.negative_indices) // negative_per_batch
        )
        
        for _ in range(num_batches):
            batch_indices = []
            
            # Add positive pairs
            for _ in range(positive_per_batch):
                try:
                    batch_indices.append(next(positive_iter))
                except StopIteration:
                    if self.drop_last:
                        return
                    positive_iter = iter(self.positive_indices)
                    batch_indices.append(next(positive_iter))
            
            # Add negative pairs
            for _ in range(negative_per_batch):
                try:
                    batch_indices.append(next(negative_iter))
                except StopIteration:
                    if self.drop_last:
                        return
                    negative_iter = iter(self.negative_indices)
                    batch_indices.append(next(negative_iter))
            
            # Shuffle batch indices
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield from batch_indices
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.positive_indices) + len(self.negative_indices) 