import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import cv2
from PIL import Image
import os
from pathlib import Path
import json

class AttentionRollout:
    """
    Implements attention rollout for visualizing accumulated attention patterns.
    Based on "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020).
    """
    
    def __init__(self, model: nn.Module, discard_ratio: float = 0.9):
        """
        Initialize attention rollout.
        
        Args:
            model: DCAL model
            discard_ratio: Ratio of attention to discard (keep top (1-discard_ratio))
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.attention_maps = []
        self.hooks = []
        
    def _hook_fn(self, module, input, output):
        """Hook function to capture attention maps."""
        if hasattr(output, 'attention_weights'):
            self.attention_maps.append(output.attention_weights.detach().cpu())
    
    def register_hooks(self):
        """Register hooks to capture attention maps."""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_rollout(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout from attention maps.
        
        Args:
            attention_maps: List of attention maps from each layer
            
        Returns:
            Rolled-out attention map
        """
        # Initialize with identity matrix
        rollout = torch.eye(attention_maps[0].size(-1))
        
        for attention in attention_maps:
            # Average across heads
            attention = attention.mean(dim=1)  # [B, N, N]
            
            # Apply discard ratio
            if self.discard_ratio > 0:
                flat_attention = attention.view(-1, attention.size(-1))
                _, indices = flat_attention.topk(
                    int(flat_attention.size(-1) * (1 - self.discard_ratio)),
                    dim=-1
                )
                flat_attention = flat_attention.scatter(-1, indices, 0)
                attention = flat_attention.view(attention.size())
            
            # Add residual connection
            attention = attention + torch.eye(attention.size(-1))
            
            # Normalize
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            # Multiply with rollout
            rollout = torch.matmul(attention[0], rollout)
        
        return rollout
    
    def visualize_rollout(self, image: torch.Tensor, patch_size: int = 16) -> np.ndarray:
        """
        Visualize attention rollout on image.
        
        Args:
            image: Input image tensor [3, H, W]
            patch_size: Patch size for vision transformer
            
        Returns:
            Attention map overlaid on image
        """
        self.attention_maps = []
        self.register_hooks()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0))
        
        # Compute rollout
        rollout = self.compute_rollout(self.attention_maps)
        
        # Remove hooks
        self.remove_hooks()
        
        # Convert to spatial attention map
        H, W = image.size(1), image.size(2)
        num_patches = H // patch_size
        
        # Extract spatial attention (excluding CLS token)
        spatial_attention = rollout[0, 1:].view(num_patches, num_patches)
        
        # Resize to image size
        attention_map = F.interpolate(
            spatial_attention.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return attention_map


class GLCAVisualizer:
    """Visualizes Global-Local Cross-Attention (GLCA) patterns."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.glca_maps = []
        self.hooks = []
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture GLCA attention maps."""
        if hasattr(output, 'cross_attention'):
            self.glca_maps.append(output.cross_attention.detach().cpu())
    
    def register_hooks(self):
        """Register hooks for GLCA modules."""
        for name, module in self.model.named_modules():
            if 'glca' in name.lower():
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_global_local_attention(self, 
                                       image: torch.Tensor,
                                       patch_size: int = 16,
                                       top_k: int = 5) -> Dict[str, np.ndarray]:
        """
        Visualize global-local cross-attention patterns.
        
        Args:
            image: Input image tensor [3, H, W]
            patch_size: Patch size for vision transformer
            top_k: Number of top attended regions to highlight
            
        Returns:
            Dictionary containing attention visualizations
        """
        self.glca_maps = []
        self.register_hooks()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0))
        
        results = {}
        
        for i, glca_map in enumerate(self.glca_maps):
            # Average across heads
            attention = glca_map.mean(dim=1)  # [B, N, N]
            
            # Convert to spatial
            H, W = image.size(1), image.size(2)
            num_patches = H // patch_size
            
            # Global attention (CLS token attention to patches)
            global_attention = attention[0, 0, 1:].view(num_patches, num_patches)
            
            # Local attention (patch-to-patch attention)
            local_attention = attention[0, 1:, 1:].mean(dim=0).view(num_patches, num_patches)
            
            # Resize to image size
            global_map = F.interpolate(
                global_attention.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            local_map = F.interpolate(
                local_attention.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            results[f'glca_{i}_global'] = global_map
            results[f'glca_{i}_local'] = local_map
            
            # Top-k attended regions
            top_k_indices = torch.topk(attention[0, 0, 1:], top_k).indices
            top_k_map = torch.zeros_like(attention[0, 0, 1:])
            top_k_map[top_k_indices] = 1.0
            top_k_map = top_k_map.view(num_patches, num_patches)
            
            top_k_spatial = F.interpolate(
                top_k_map.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'
            ).squeeze().numpy()
            
            results[f'glca_{i}_top_k'] = top_k_spatial
        
        self.remove_hooks()
        return results


class PWCAVisualizer:
    """Visualizes Pair-Wise Cross-Attention (PWCA) patterns."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.pwca_maps = []
        self.hooks = []
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture PWCA attention maps."""
        if hasattr(output, 'pair_attention'):
            self.pwca_maps.append(output.pair_attention.detach().cpu())
    
    def register_hooks(self):
        """Register hooks for PWCA modules."""
        for name, module in self.model.named_modules():
            if 'pwca' in name.lower():
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_pair_attention(self, 
                               image1: torch.Tensor,
                               image2: torch.Tensor,
                               patch_size: int = 16) -> Dict[str, np.ndarray]:
        """
        Visualize pair-wise cross-attention between two images.
        
        Args:
            image1: First image tensor [3, H, W]
            image2: Second image tensor [3, H, W]
            patch_size: Patch size for vision transformer
            
        Returns:
            Dictionary containing pair attention visualizations
        """
        self.pwca_maps = []
        self.register_hooks()
        
        # Forward pass with paired images
        with torch.no_grad():
            batch = torch.stack([image1, image2], dim=0)
            _ = self.model(batch)
        
        results = {}
        
        for i, pwca_map in enumerate(self.pwca_maps):
            # Average across heads
            attention = pwca_map.mean(dim=1)  # [B, N, N]
            
            # Convert to spatial
            H, W = image1.size(1), image1.size(2)
            num_patches = H // patch_size
            
            # Cross-attention between images
            cross_attention_12 = attention[0, 1:, 1:].view(num_patches, num_patches)
            cross_attention_21 = attention[1, 1:, 1:].view(num_patches, num_patches)
            
            # Resize to image size
            cross_map_12 = F.interpolate(
                cross_attention_12.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            cross_map_21 = F.interpolate(
                cross_attention_21.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            results[f'pwca_{i}_cross_12'] = cross_map_12
            results[f'pwca_{i}_cross_21'] = cross_map_21
            
            # Attention difference (highlighting discriminative regions)
            diff_map = np.abs(cross_map_12 - cross_map_21)
            results[f'pwca_{i}_difference'] = diff_map
        
        self.remove_hooks()
        return results


class FaceAttentionVisualizer:
    """Face-specific attention visualization tools."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.rollout_viz = AttentionRollout(model)
        self.glca_viz = GLCAVisualizer(model)
        self.pwca_viz = PWCAVisualizer(model)
    
    def overlay_attention_on_face(self, 
                                 image: np.ndarray,
                                 attention_map: np.ndarray,
                                 alpha: float = 0.4,
                                 colormap: str = 'jet') -> np.ndarray:
        """
        Overlay attention map on face image.
        
        Args:
            image: Face image [H, W, 3] (0-255)
            attention_map: Attention map [H, W] (0-1)
            alpha: Transparency of attention overlay
            colormap: Colormap for attention visualization
            
        Returns:
            Image with attention overlay
        """
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        attention_colored = cmap(attention_map)[:, :, :3]  # Remove alpha channel
        attention_colored = (attention_colored * 255).astype(np.uint8)
        
        # Overlay on image
        overlay = cv2.addWeighted(image, 1 - alpha, attention_colored, alpha, 0)
        
        return overlay
    
    def highlight_facial_regions(self, 
                               image: np.ndarray,
                               attention_map: np.ndarray,
                               threshold: float = 0.7) -> np.ndarray:
        """
        Highlight highly attended facial regions.
        
        Args:
            image: Face image [H, W, 3] (0-255)
            attention_map: Attention map [H, W] (0-1)
            threshold: Threshold for highlighting
            
        Returns:
            Image with highlighted regions
        """
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Create mask for high attention regions
        high_attention_mask = attention_map > threshold
        
        # Create highlight overlay
        highlight = np.zeros_like(image)
        highlight[high_attention_mask] = [255, 255, 0]  # Yellow highlight
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.7, highlight, 0.3, 0)
        
        return result
    
    def compare_twin_attention(self, 
                             image1: np.ndarray,
                             image2: np.ndarray,
                             attention_map1: np.ndarray,
                             attention_map2: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of twin attention patterns.
        
        Args:
            image1: First twin image [H, W, 3]
            image2: Second twin image [H, W, 3]
            attention_map1: Attention map for first image
            attention_map2: Attention map for second image
            
        Returns:
            Side-by-side comparison image
        """
        # Overlay attention on both images
        overlay1 = self.overlay_attention_on_face(image1, attention_map1)
        overlay2 = self.overlay_attention_on_face(image2, attention_map2)
        
        # Create difference map
        diff_map = np.abs(attention_map1 - attention_map2)
        diff_overlay = self.overlay_attention_on_face(image1, diff_map, colormap='RdBu')
        
        # Concatenate horizontally
        comparison = np.hstack([overlay1, overlay2, diff_overlay])
        
        return comparison
    
    def generate_attention_summary(self, 
                                 image: torch.Tensor,
                                 patch_size: int = 16) -> Dict[str, np.ndarray]:
        """
        Generate comprehensive attention summary for a face image.
        
        Args:
            image: Face image tensor [3, H, W]
            patch_size: Patch size for vision transformer
            
        Returns:
            Dictionary of attention visualizations
        """
        results = {}
        
        # Convert tensor to numpy for visualization
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Attention rollout
        rollout_map = self.rollout_viz.visualize_rollout(image, patch_size)
        results['rollout'] = rollout_map
        results['rollout_overlay'] = self.overlay_attention_on_face(image_np, rollout_map)
        
        # GLCA attention
        glca_maps = self.glca_viz.visualize_global_local_attention(image, patch_size)
        for key, attention_map in glca_maps.items():
            results[key] = attention_map
            results[f'{key}_overlay'] = self.overlay_attention_on_face(image_np, attention_map)
        
        return results


class VerificationExplainer:
    """Explains verification decisions using attention patterns."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.face_viz = FaceAttentionVisualizer(model)
    
    def explain_verification_decision(self, 
                                    image1: torch.Tensor,
                                    image2: torch.Tensor,
                                    similarity_score: float,
                                    threshold: float = 0.5,
                                    patch_size: int = 16) -> Dict[str, any]:
        """
        Explain why the model made a specific verification decision.
        
        Args:
            image1: First face image [3, H, W]
            image2: Second face image [3, H, W]
            similarity_score: Computed similarity score
            threshold: Decision threshold
            patch_size: Patch size for vision transformer
            
        Returns:
            Dictionary containing explanation
        """
        decision = "Same Person" if similarity_score > threshold else "Different Person"
        confidence = abs(similarity_score - threshold)
        
        # Get attention patterns
        attention1 = self.face_viz.generate_attention_summary(image1, patch_size)
        attention2 = self.face_viz.generate_attention_summary(image2, patch_size)
        
        # Analyze attention differences
        rollout_diff = np.abs(attention1['rollout'] - attention2['rollout'])
        
        # Find most discriminative regions
        top_regions = self._find_discriminative_regions(rollout_diff)
        
        # Create visualizations
        comparison_viz = self.face_viz.compare_twin_attention(
            (image1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            (image2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            attention1['rollout'],
            attention2['rollout']
        )
        
        explanation = {
            'decision': decision,
            'similarity_score': similarity_score,
            'confidence': confidence,
            'threshold': threshold,
            'discriminative_regions': top_regions,
            'attention_comparison': comparison_viz,
            'attention_difference': rollout_diff
        }
        
        return explanation
    
    def _find_discriminative_regions(self, 
                                   attention_diff: np.ndarray,
                                   num_regions: int = 5) -> List[Dict[str, any]]:
        """
        Find the most discriminative regions in the attention difference map.
        
        Args:
            attention_diff: Attention difference map
            num_regions: Number of top regions to return
            
        Returns:
            List of discriminative regions with their properties
        """
        # Find connected components of high attention difference
        threshold = np.percentile(attention_diff, 90)
        binary_mask = attention_diff > threshold
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small regions
                # Calculate region properties
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Create mask for this region
                    mask = np.zeros_like(attention_diff)
                    cv2.fillPoly(mask, [contour], 1)
                    
                    # Calculate average attention difference
                    avg_diff = np.mean(attention_diff[mask == 1])
                    
                    regions.append({
                        'center': (cx, cy),
                        'area': cv2.contourArea(contour),
                        'avg_difference': avg_diff,
                        'contour': contour
                    })
        
        # Sort by average difference and return top regions
        regions.sort(key=lambda x: x['avg_difference'], reverse=True)
        return regions[:num_regions]


class AttentionAnalyzer:
    """Comprehensive attention analysis tools."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.explainer = VerificationExplainer(model)
    
    def analyze_attention_patterns(self, 
                                 dataset_loader,
                                 num_samples: int = 100) -> Dict[str, any]:
        """
        Analyze attention patterns across the dataset.
        
        Args:
            dataset_loader: DataLoader for the dataset
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        attention_stats = {
            'rollout_means': [],
            'rollout_stds': [],
            'glca_means': [],
            'glca_stds': [],
            'similarity_scores': [],
            'labels': []
        }
        
        face_viz = FaceAttentionVisualizer(self.model)
        
        sample_count = 0
        for batch in dataset_loader:
            if sample_count >= num_samples:
                break
            
            images, labels = batch
            
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                # Get attention summary
                attention_summary = face_viz.generate_attention_summary(images[i])
                
                # Collect statistics
                attention_stats['rollout_means'].append(attention_summary['rollout'].mean())
                attention_stats['rollout_stds'].append(attention_summary['rollout'].std())
                
                # Get GLCA statistics if available
                glca_keys = [k for k in attention_summary.keys() if 'glca' in k and 'overlay' not in k]
                if glca_keys:
                    glca_mean = np.mean([attention_summary[k].mean() for k in glca_keys])
                    glca_std = np.mean([attention_summary[k].std() for k in glca_keys])
                    attention_stats['glca_means'].append(glca_mean)
                    attention_stats['glca_stds'].append(glca_std)
                
                attention_stats['labels'].append(labels[i].item())
                sample_count += 1
        
        # Calculate summary statistics
        analysis_results = {
            'total_samples': sample_count,
            'rollout_statistics': {
                'mean_attention': np.mean(attention_stats['rollout_means']),
                'std_attention': np.mean(attention_stats['rollout_stds']),
                'variation_across_samples': np.std(attention_stats['rollout_means'])
            },
            'glca_statistics': {
                'mean_attention': np.mean(attention_stats['glca_means']) if attention_stats['glca_means'] else 0,
                'std_attention': np.mean(attention_stats['glca_stds']) if attention_stats['glca_stds'] else 0,
                'variation_across_samples': np.std(attention_stats['glca_means']) if attention_stats['glca_means'] else 0
            },
            'raw_data': attention_stats
        }
        
        return analysis_results
    
    def generate_attention_report(self, 
                                analysis_results: Dict[str, any],
                                output_dir: str = './attention_analysis') -> str:
        """
        Generate comprehensive attention analysis report.
        
        Args:
            analysis_results: Results from analyze_attention_patterns
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        self._create_attention_plots(analysis_results, output_dir)
        
        # Generate text report
        report_path = os.path.join(output_dir, 'attention_report.txt')
        with open(report_path, 'w') as f:
            f.write("DCAL Twin Faces - Attention Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total samples analyzed: {analysis_results['total_samples']}\n\n")
            
            f.write("Attention Rollout Statistics:\n")
            f.write(f"  Mean attention: {analysis_results['rollout_statistics']['mean_attention']:.4f}\n")
            f.write(f"  Std attention: {analysis_results['rollout_statistics']['std_attention']:.4f}\n")
            f.write(f"  Variation across samples: {analysis_results['rollout_statistics']['variation_across_samples']:.4f}\n\n")
            
            f.write("GLCA Statistics:\n")
            f.write(f"  Mean attention: {analysis_results['glca_statistics']['mean_attention']:.4f}\n")
            f.write(f"  Std attention: {analysis_results['glca_statistics']['std_attention']:.4f}\n")
            f.write(f"  Variation across samples: {analysis_results['glca_statistics']['variation_across_samples']:.4f}\n\n")
        
        return report_path
    
    def _create_attention_plots(self, 
                              analysis_results: Dict[str, any],
                              output_dir: str):
        """Create visualization plots for attention analysis."""
        plt.style.use('seaborn-v0_8')
        
        # Rollout attention distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(analysis_results['raw_data']['rollout_means'], bins=30, alpha=0.7)
        plt.xlabel('Mean Attention')
        plt.ylabel('Frequency')
        plt.title('Distribution of Rollout Attention Means')
        
        plt.subplot(2, 2, 2)
        plt.hist(analysis_results['raw_data']['rollout_stds'], bins=30, alpha=0.7)
        plt.xlabel('Attention Std')
        plt.ylabel('Frequency')
        plt.title('Distribution of Rollout Attention Stds')
        
        if analysis_results['raw_data']['glca_means']:
            plt.subplot(2, 2, 3)
            plt.hist(analysis_results['raw_data']['glca_means'], bins=30, alpha=0.7)
            plt.xlabel('Mean GLCA Attention')
            plt.ylabel('Frequency')
            plt.title('Distribution of GLCA Attention Means')
            
            plt.subplot(2, 2, 4)
            plt.hist(analysis_results['raw_data']['glca_stds'], bins=30, alpha=0.7)
            plt.xlabel('GLCA Attention Std')
            plt.ylabel('Frequency')
            plt.title('Distribution of GLCA Attention Stds')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_distributions.png'), dpi=300)
        plt.close()
        
        # Correlation plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(analysis_results['raw_data']['rollout_means'], 
                   analysis_results['raw_data']['rollout_stds'],
                   alpha=0.6)
        plt.xlabel('Mean Attention')
        plt.ylabel('Std Attention')
        plt.title('Attention Mean vs Std (Rollout)')
        
        if analysis_results['raw_data']['glca_means']:
            plt.subplot(1, 2, 2)
            plt.scatter(analysis_results['raw_data']['glca_means'], 
                       analysis_results['raw_data']['glca_stds'],
                       alpha=0.6)
            plt.xlabel('Mean GLCA Attention')
            plt.ylabel('Std GLCA Attention')
            plt.title('Attention Mean vs Std (GLCA)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_correlations.png'), dpi=300)
        plt.close() 