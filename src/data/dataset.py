"""
Twin faces dataset loader for verification task.
Handles loading images, creating positive/negative pairs, and data preprocessing.
"""

import json
import os
import random
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class TwinDataset(Dataset):
    """
    Dataset class for twin faces verification.
    
    Loads images from the ND TWIN dataset and creates positive/negative pairs
    for training the verification model.
    """
    
    def __init__(
        self,
        dataset_info_path: str,
        twin_pairs_path: str,
        data_root: str = "",
        transform=None,
        mode: str = "train",
        negative_ratio: float = 1.0,
        hard_negative_ratio: float = 0.3,
        soft_negative_ratio: float = 0.2,
        use_only_hard_negatives: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset_info_path: Path to dataset info JSON file
            twin_pairs_path: Path to twin pairs JSON file
            data_root: Root directory for image paths
            transform: Data transform to apply
            mode: Dataset mode ('train', 'val', 'test')
            negative_ratio: Ratio of negative to positive pairs
            hard_negative_ratio: Ratio of hard negatives (different twins)
            soft_negative_ratio: Ratio of soft negatives (random pairs)
            use_only_hard_negatives: If True, use only hard negatives (twins) for training
        """
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.soft_negative_ratio = soft_negative_ratio
        self.use_only_hard_negatives = use_only_hard_negatives
        
        # Load dataset info
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        # Load twin pairs
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Create person ID to twin pair mapping
        self.person_to_twin = {}
        for twin_pair in self.twin_pairs:
            self.person_to_twin[twin_pair[0]] = twin_pair[1]
            self.person_to_twin[twin_pair[1]] = twin_pair[0]
        
        # Get all person IDs and create twin sets
        self.all_person_ids = list(self.dataset_info.keys())
        self.twin_sets = self._create_twin_sets()
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        print(f"Dataset initialized with {len(self.pairs)} pairs")
        print(f"Positive pairs: {sum(1 for _, _, label in self.pairs if label == 1)}")
        print(f"Negative pairs: {sum(1 for _, _, label in self.pairs if label == 0)}")
    
    def _create_twin_sets(self) -> List[List[str]]:
        """Create sets of twin pairs for hard negative mining."""
        twin_sets = []
        processed_pairs = set()
        
        for person_id in self.all_person_ids:
            if person_id in self.person_to_twin:
                twin_id = self.person_to_twin[person_id]
                pair = tuple(sorted([person_id, twin_id]))
                
                if pair not in processed_pairs:
                    twin_sets.append([person_id, twin_id])
                    processed_pairs.add(pair)
        
        return twin_sets
    
    def _generate_pairs(self) -> List[Tuple[str, str, int]]:
        """Generate positive and negative pairs for training."""
        pairs = []
        
        # Generate positive pairs (same person)
        positive_pairs = []
        for person_id, image_paths in self.dataset_info.items():
            if len(image_paths) > 1:
                # Create all possible pairs within the same person
                for i in range(len(image_paths)):
                    for j in range(i + 1, len(image_paths)):
                        positive_pairs.append((image_paths[i], image_paths[j], 1))
        
        pairs.extend(positive_pairs)
        num_positive = len(positive_pairs)
        
        # Generate negative pairs
        num_negative = int(num_positive * self.negative_ratio)
        
        if self.use_only_hard_negatives:
            # ONLY HARD NEGATIVES: Different people from twin pairs (most challenging)
            hard_negatives = []
            
            # Generate all possible hard negative combinations
            all_hard_negatives = []
            for twin_set1 in self.twin_sets:
                for twin_set2 in self.twin_sets:
                    if twin_set1 != twin_set2:
                        # Create pairs between different twin sets
                        for person1 in twin_set1:
                            for person2 in twin_set2:
                                if person1 in self.dataset_info and person2 in self.dataset_info:
                                    for img1 in self.dataset_info[person1]:
                                        for img2 in self.dataset_info[person2]:
                                            all_hard_negatives.append((img1, img2, 0))
            
            # Sample from all possible hard negatives
            if len(all_hard_negatives) >= num_negative:
                hard_negatives = random.sample(all_hard_negatives, num_negative)
            else:
                # If not enough hard negatives, use all of them and add duplicates
                hard_negatives = all_hard_negatives[:]
                while len(hard_negatives) < num_negative:
                    hard_negatives.extend(random.sample(all_hard_negatives, 
                                                      min(len(all_hard_negatives), 
                                                          num_negative - len(hard_negatives))))
            
            pairs.extend(hard_negatives)
            
            print(f"Using ONLY hard negatives: {len(hard_negatives)} pairs")
            print(f"Total possible hard negatives: {len(all_hard_negatives)}")
            print(f"Hard negative sampling ratio: {len(hard_negatives) / len(all_hard_negatives):.2%}")
        
        else:
            # MIXED NEGATIVES: Original implementation with hard + soft negatives
            num_hard_negative = int(num_negative * self.hard_negative_ratio)
            num_soft_negative = num_negative - num_hard_negative
            
            # Hard negatives: Different people from twin pairs
            hard_negatives = []
            for _ in range(num_hard_negative):
                # Select two different twin pairs
                twin_set1 = random.choice(self.twin_sets)
                twin_set2 = random.choice(self.twin_sets)
                
                # Ensure different twin pairs
                while twin_set1 == twin_set2:
                    twin_set2 = random.choice(self.twin_sets)
                
                # Select random person from each twin pair
                person1 = random.choice(twin_set1)
                person2 = random.choice(twin_set2)
                
                # Select random images
                img1 = random.choice(self.dataset_info[person1])
                img2 = random.choice(self.dataset_info[person2])
                
                hard_negatives.append((img1, img2, 0))
            
            # Soft negatives: Random different people
            soft_negatives = []
            for _ in range(num_soft_negative):
                person1 = random.choice(self.all_person_ids)
                person2 = random.choice(self.all_person_ids)
                
                # Ensure different people
                while person1 == person2:
                    person2 = random.choice(self.all_person_ids)
                
                img1 = random.choice(self.dataset_info[person1])
                img2 = random.choice(self.dataset_info[person2])
                
                soft_negatives.append((img1, img2, 0))
            
            pairs.extend(hard_negatives)
            pairs.extend(soft_negatives)
            
            print(f"Using mixed negatives: {len(hard_negatives)} hard + {len(soft_negatives)} soft")
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        # Handle different path formats
        if self.data_root and not image_path.startswith(self.data_root):
            # Extract relative path from absolute path
            if "/kaggle/input/nd-twin/ND_TWIN_Dataset_224/" in image_path:
                relative_path = image_path.split("/kaggle/input/nd-twin/ND_TWIN_Dataset_224/")[1]
                image_path = os.path.join(self.data_root, relative_path)
            else:
                image_path = os.path.join(self.data_root, image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def __len__(self) -> int:
        """Return the number of pairs in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a pair of images and their label.
        
        Args:
            idx: Index of the pair
        
        Returns:
            Tuple of (image1, image2, label)
        """
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label
    
    def get_person_images(self, person_id: str) -> List[str]:
        """Get all images for a specific person."""
        return self.dataset_info.get(person_id, [])
    
    def get_twin_pair(self, person_id: str) -> Optional[str]:
        """Get the twin pair for a given person."""
        return self.person_to_twin.get(person_id)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_persons': len(self.all_person_ids),
            'total_twin_pairs': len(self.twin_sets),
            'total_images': sum(len(images) for images in self.dataset_info.values()),
            'total_pairs': len(self.pairs),
            'positive_pairs': sum(1 for _, _, label in self.pairs if label == 1),
            'negative_pairs': sum(1 for _, _, label in self.pairs if label == 0),
            'images_per_person': {
                'min': min(len(images) for images in self.dataset_info.values()),
                'max': max(len(images) for images in self.dataset_info.values()),
                'avg': np.mean([len(images) for images in self.dataset_info.values()])
            }
        }
        return stats


class TwinVerificationDataset(Dataset):
    """
    Dataset for twin verification evaluation.
    
    Provides all possible pairs for evaluation without balancing.
    """
    
    def __init__(
        self,
        dataset_info_path: str,
        twin_pairs_path: str,
        data_root: str = "",
        transform=None,
        same_person_only: bool = False,
        hard_pairs_only: bool = False
    ):
        """
        Initialize evaluation dataset.
        
        Args:
            dataset_info_path: Path to dataset info JSON file
            twin_pairs_path: Path to twin pairs JSON file
            data_root: Root directory for image paths
            transform: Data transform to apply
            same_person_only: If True, only generate same-person pairs
            hard_pairs_only: If True, only generate same-person or twin-person pairs
        """
        self.data_root = data_root
        self.transform = transform
        self.same_person_only = same_person_only
        self.hard_pairs_only = hard_pairs_only
        
        # Load dataset info
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        # Load twin pairs
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        # Build twin lookup for fast check
        self.twin_lookup = set()
        for a, b in self.twin_pairs:
            self.twin_lookup.add((a, b))
            self.twin_lookup.add((b, a))
        
        # Generate all evaluation pairs
        self.pairs = self._generate_evaluation_pairs()
        
        print(f"Evaluation dataset initialized with {len(self.pairs)} pairs")
    
    def _generate_evaluation_pairs(self) -> List[Tuple[str, str, int, str, str]]:
        """Generate all evaluation pairs with metadata."""
        pairs = []
        
        if self.same_person_only:
            # Only same-person pairs
            for person_id, image_paths in self.dataset_info.items():
                if len(image_paths) > 1:
                    for i in range(len(image_paths)):
                        for j in range(i + 1, len(image_paths)):
                            pairs.append((image_paths[i], image_paths[j], 1, person_id, person_id))
        elif self.hard_pairs_only:
            # Only same-person or twin-person pairs
            # Same-person pairs
            for person_id, image_paths in self.dataset_info.items():
                if len(image_paths) > 1:
                    for i in range(len(image_paths)):
                        for j in range(i + 1, len(image_paths)):
                            pairs.append((image_paths[i], image_paths[j], 1, person_id, person_id))
            # Twin-person pairs
            for a, b in self.twin_pairs:
                if a in self.dataset_info and b in self.dataset_info:
                    for img1 in self.dataset_info[a]:
                        for img2 in self.dataset_info[b]:
                            pairs.append((img1, img2, 0, a, b))
                        
                    for img1 in self.dataset_info[b]:
                        for img2 in self.dataset_info[a]:
                            pairs.append((img1, img2, 0, b, a))
        else:
            # All possible pairs
            all_images = []
            for person_id, image_paths in self.dataset_info.items():
                for img_path in image_paths:
                    all_images.append((img_path, person_id))
            # Generate all pairs
            for i in range(len(all_images)):
                for j in range(i + 1, len(all_images)):
                    img1_path, person1 = all_images[i]
                    img2_path, person2 = all_images[j]
                    label = 1 if person1 == person2 else 0
                    pairs.append((img1_path, img2_path, label, person1, person2))
        
        return pairs
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        # Handle different path formats
        if self.data_root and not image_path.startswith(self.data_root):
            # Extract relative path from absolute path
            if "/kaggle/input/nd-twin/ND_TWIN_Dataset_224/" in image_path:
                relative_path = image_path.split("/kaggle/input/nd-twin/ND_TWIN_Dataset_224/")[1]
                image_path = os.path.join(self.data_root, relative_path)
            else:
                image_path = os.path.join(self.data_root, image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def __len__(self) -> int:
        """Return the number of pairs in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """
        Get a pair of images with metadata.
        
        Args:
            idx: Index of the pair
        
        Returns:
            Tuple of (image1, image2, label, person1, person2)
        """
        img1_path, img2_path, label, person1, person2 = self.pairs[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label, person1, person2 