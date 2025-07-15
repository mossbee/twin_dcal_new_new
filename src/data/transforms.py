"""
Data augmentation transforms for twin face verification.
Face-specific augmentations that preserve facial structure.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import random
import numpy as np


class FaceAwareColorJitter(object):
    """Color jitter that preserves facial structure."""
    
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        # Apply mild color jitter to preserve facial features
        transform = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )
        return transform(img)


class FaceAwareRotation(object):
    """Small rotation that preserves facial structure."""
    
    def __init__(self, degrees=5):
        self.degrees = degrees
    
    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR)


class FaceAwareAffine(object):
    """Small affine transformation for face images."""
    
    def __init__(self, translate=0.05, scale=(0.95, 1.05), shear=2):
        self.translate = translate
        self.scale = scale
        self.shear = shear
    
    def __call__(self, img):
        # Small random affine transformation
        translate = (random.uniform(-self.translate, self.translate) * img.width,
                    random.uniform(-self.translate, self.translate) * img.height)
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear, self.shear)
        
        return F.affine(img, angle=0, translate=translate, scale=scale, shear=shear,
                       interpolation=F.InterpolationMode.BILINEAR)


class GaussianBlur(object):
    """Apply Gaussian blur with random kernel size."""
    
    def __init__(self, kernel_size_range=(1, 3), sigma_range=(0.1, 0.5)):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
    
    def __call__(self, img):
        kernel_size = random.randint(self.kernel_size_range[0], self.kernel_size_range[1])
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        return F.gaussian_blur(img, kernel_size, sigma)


def get_train_transforms(image_size=224, strong_augmentation=True):
    """
    Get training data transformations.
    
    Args:
        image_size: Target image size
        strong_augmentation: Whether to apply strong augmentation
    
    Returns:
        torchvision.transforms.Compose: Training transforms
    """
    transforms_list = []
    
    # Base transforms
    transforms_list.extend([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    if strong_augmentation:
        # Face-aware augmentations
        transforms_list.extend([
            FaceAwareRotation(degrees=5),
            FaceAwareColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            FaceAwareAffine(translate=0.05, scale=(0.95, 1.05), shear=2),
            transforms.RandomApply([GaussianBlur(kernel_size_range=(1, 3))], p=0.2),
        ])
    
    # Convert to tensor and normalize
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transforms_list)


def get_val_transforms(image_size=224):
    """
    Get validation data transformations.
    
    Args:
        image_size: Target image size
    
    Returns:
        torchvision.transforms.Compose: Validation transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_multi_scale_transforms(image_sizes=[224, 256, 448]):
    """
    Get multi-scale transforms for different image sizes.
    
    Args:
        image_sizes: List of image sizes to support
    
    Returns:
        dict: Dictionary mapping size to transforms
    """
    transforms_dict = {}
    
    for size in image_sizes:
        transforms_dict[f'train_{size}'] = get_train_transforms(size)
        transforms_dict[f'val_{size}'] = get_val_transforms(size)
    
    return transforms_dict


class TwinPairTransform(object):
    """Transform for twin pairs that ensures consistent augmentation."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img1, img2):
        # Apply the same random seed for consistent augmentation
        seed = random.randint(0, 2**32)
        
        # Transform first image
        random.seed(seed)
        torch.manual_seed(seed)
        img1_transformed = self.transform(img1)
        
        # Transform second image with same seed
        random.seed(seed)
        torch.manual_seed(seed)
        img2_transformed = self.transform(img2)
        
        return img1_transformed, img2_transformed


def get_transforms(config, is_training=True):
    """
    Get appropriate transforms based on configuration and training mode.
    
    Args:
        config: Configuration object
        is_training: Whether to get training transforms
    
    Returns:
        torchvision.transforms.Compose: Appropriate transforms
    """
    image_size = getattr(config.data, 'image_size', 224)
    
    if is_training:
        strong_aug = getattr(config.training, 'strong_augmentation', True)
        return get_train_transforms(image_size, strong_aug)
    else:
        return get_val_transforms(image_size)