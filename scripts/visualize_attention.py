#!/usr/bin/env python3
"""
DCAL Twin Faces - Attention Visualization Script

This script provides comprehensive attention visualization and analysis tools for 
the DCAL twin faces verification model.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.siamese_dcal import SiameseDCAL
from data.dataset import TwinDataset
from data.transforms import get_transforms
from utils.config import Config
from utils.visualization import (
    AttentionRollout, GLCAVisualizer, PWCAVisualizer, 
    FaceAttentionVisualizer, VerificationExplainer, AttentionAnalyzer
)
from utils.checkpoint import CheckpointManager
from utils.logging import setup_logging
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DCAL Twin Faces - Attention Visualization')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./attention_analysis',
                      help='Output directory for visualizations')
    parser.add_argument('--mode', type=str, default='analyze',
                      choices=['analyze', 'visualize', 'explain', 'compare'],
                      help='Visualization mode')
    
    # Dataset options
    parser.add_argument('--data-split', type=str, default='test',
                      choices=['train', 'test', 'val'],
                      help='Dataset split to analyze')
    parser.add_argument('--num-samples', type=int, default=100,
                      help='Number of samples to analyze')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for processing')
    
    # Visualization options
    parser.add_argument('--attention-type', type=str, default='all',
                      choices=['rollout', 'glca', 'pwca', 'all'],
                      help='Type of attention to visualize')
    parser.add_argument('--save-individual', action='store_true',
                      help='Save individual attention maps')
    parser.add_argument('--colormap', type=str, default='jet',
                      help='Colormap for attention visualization')
    parser.add_argument('--overlay-alpha', type=float, default=0.4,
                      help='Transparency of attention overlay')
    
    # Comparison options
    parser.add_argument('--compare-pairs', type=str, nargs='+',
                      help='Specific image pairs to compare (format: img1_path img2_path)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Verification threshold for explanations')
    
    # Analysis options
    parser.add_argument('--generate-report', action='store_true',
                      help='Generate comprehensive analysis report')
    parser.add_argument('--export-stats', action='store_true',
                      help='Export attention statistics to JSON')
    
    return parser.parse_args()


def load_model_and_config(checkpoint_path: str, config_path: str) -> Tuple[nn.Module, Config]:
    """Load model and configuration from checkpoint."""
    print(f"Loading configuration from {config_path}")
    config = Config.from_file(config_path)
    
    print(f"Loading model from {checkpoint_path}")
    model = SiameseDCAL(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, config


def setup_data_loader(config: Config, split: str, batch_size: int) -> DataLoader:
    """Setup data loader for the specified split."""
    transforms = get_transforms(config, is_training=False)
    
    dataset = TwinDataset(
        config=config,
        split=split,
        transform=transforms,
        return_pairs=False  # For individual image analysis
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def analyze_attention_patterns(model: nn.Module, 
                             dataloader: DataLoader,
                             num_samples: int,
                             output_dir: str) -> Dict[str, any]:
    """Analyze attention patterns across the dataset."""
    print(f"Analyzing attention patterns on {num_samples} samples...")
    
    analyzer = AttentionAnalyzer(model)
    analysis_results = analyzer.analyze_attention_patterns(dataloader, num_samples)
    
    # Generate report
    report_path = analyzer.generate_attention_report(analysis_results, output_dir)
    print(f"Analysis report saved to: {report_path}")
    
    return analysis_results


def visualize_individual_attention(model: nn.Module,
                                 dataloader: DataLoader,
                                 attention_type: str,
                                 output_dir: str,
                                 num_samples: int = 20,
                                 save_individual: bool = False,
                                 colormap: str = 'jet',
                                 overlay_alpha: float = 0.4):
    """Visualize attention maps for individual images."""
    print(f"Visualizing {attention_type} attention for {num_samples} samples...")
    
    face_viz = FaceAttentionVisualizer(model)
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0
    device = next(model.parameters()).device
    
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break
            
        images, labels = batch
        images = images.to(device)
        
        for i in range(images.size(0)):
            if sample_count >= num_samples:
                break
                
            image = images[i]
            
            # Generate attention summary
            attention_summary = face_viz.generate_attention_summary(image)
            
            # Create visualization grid
            if attention_type == 'all':
                vis_keys = [k for k in attention_summary.keys() if 'overlay' in k]
            else:
                vis_keys = [k for k in attention_summary.keys() 
                          if attention_type in k and 'overlay' in k]
            
            if save_individual:
                # Save individual attention maps
                for key in vis_keys:
                    save_path = os.path.join(output_dir, f'sample_{sample_count:03d}_{key}.png')
                    plt.imsave(save_path, attention_summary[key])
            
            # Create combined visualization
            if len(vis_keys) > 0:
                fig, axes = plt.subplots(1, len(vis_keys), figsize=(5 * len(vis_keys), 5))
                if len(vis_keys) == 1:
                    axes = [axes]
                
                for idx, key in enumerate(vis_keys):
                    axes[idx].imshow(attention_summary[key])
                    axes[idx].set_title(key.replace('_', ' ').title())
                    axes[idx].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(output_dir, f'sample_{sample_count:03d}_combined.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            sample_count += 1
    
    print(f"Saved {sample_count} attention visualizations to {output_dir}")


def explain_verification_decisions(model: nn.Module,
                                 config: Config,
                                 output_dir: str,
                                 pairs: List[Tuple[str, str]] = None,
                                 threshold: float = 0.5,
                                 num_random_pairs: int = 10):
    """Explain verification decisions for specific pairs."""
    print("Explaining verification decisions...")
    
    explainer = VerificationExplainer(model)
    os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    transforms = get_transforms(config, is_training=False)
    
    # Use provided pairs or generate random pairs
    if pairs is None:
        # Generate random pairs from test set
        dataset = TwinDataset(config, split='test', transform=transforms, return_pairs=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        pair_count = 0
        for batch in dataloader:
            if pair_count >= num_random_pairs:
                break
                
            (image1, image2), label = batch
            image1, image2 = image1.to(device), image2.to(device)
            
            # Get similarity score
            with torch.no_grad():
                features1 = model.encode_single(image1)
                features2 = model.encode_single(image2)
                similarity = model.compute_similarity(features1, features2)
                similarity_score = similarity.item()
            
            # Generate explanation
            explanation = explainer.explain_verification_decision(
                image1.squeeze(0), image2.squeeze(0), similarity_score, threshold
            )
            
            # Save explanation
            save_explanation(explanation, output_dir, f'pair_{pair_count:03d}')
            pair_count += 1
    
    else:
        # Process provided pairs
        for pair_idx, (img1_path, img2_path) in enumerate(pairs):
            # Load and preprocess images
            image1 = load_and_preprocess_image(img1_path, transforms).to(device)
            image2 = load_and_preprocess_image(img2_path, transforms).to(device)
            
            # Get similarity score
            with torch.no_grad():
                features1 = model.encode_single(image1.unsqueeze(0))
                features2 = model.encode_single(image2.unsqueeze(0))
                similarity = model.compute_similarity(features1, features2)
                similarity_score = similarity.item()
            
            # Generate explanation
            explanation = explainer.explain_verification_decision(
                image1, image2, similarity_score, threshold
            )
            
            # Save explanation
            save_explanation(explanation, output_dir, f'custom_pair_{pair_idx:03d}')


def save_explanation(explanation: Dict[str, any], output_dir: str, filename: str):
    """Save explanation results to files."""
    # Save text explanation
    text_path = os.path.join(output_dir, f'{filename}_explanation.txt')
    with open(text_path, 'w') as f:
        f.write(f"Verification Decision: {explanation['decision']}\n")
        f.write(f"Similarity Score: {explanation['similarity_score']:.4f}\n")
        f.write(f"Threshold: {explanation['threshold']:.4f}\n")
        f.write(f"Confidence: {explanation['confidence']:.4f}\n\n")
        
        f.write("Top Discriminative Regions:\n")
        for i, region in enumerate(explanation['discriminative_regions']):
            f.write(f"  {i+1}. Center: {region['center']}, ")
            f.write(f"Area: {region['area']:.1f}, ")
            f.write(f"Avg Difference: {region['avg_difference']:.4f}\n")
    
    # Save visualization
    img_path = os.path.join(output_dir, f'{filename}_comparison.png')
    plt.imsave(img_path, explanation['attention_comparison'])
    
    # Save attention difference map
    diff_path = os.path.join(output_dir, f'{filename}_difference.png')
    plt.imsave(diff_path, explanation['attention_difference'], cmap='RdBu')


def load_and_preprocess_image(image_path: str, transforms) -> torch.Tensor:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    image = transforms(image)
    return image


def compare_attention_patterns(model: nn.Module,
                             config: Config,
                             output_dir: str,
                             comparison_pairs: List[Tuple[str, str]]):
    """Compare attention patterns between specific image pairs."""
    print("Comparing attention patterns...")
    
    face_viz = FaceAttentionVisualizer(model)
    os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    transforms = get_transforms(config, is_training=False)
    
    for pair_idx, (img1_path, img2_path) in enumerate(comparison_pairs):
        # Load images
        image1 = load_and_preprocess_image(img1_path, transforms).to(device)
        image2 = load_and_preprocess_image(img2_path, transforms).to(device)
        
        # Get attention summaries
        attention1 = face_viz.generate_attention_summary(image1)
        attention2 = face_viz.generate_attention_summary(image2)
        
        # Create comparison visualization
        image1_np = (image1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image2_np = (image2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        comparison_viz = face_viz.compare_twin_attention(
            image1_np, image2_np, attention1['rollout'], attention2['rollout']
        )
        
        # Save comparison
        save_path = os.path.join(output_dir, f'comparison_{pair_idx:03d}.png')
        plt.imsave(save_path, comparison_viz)
        
        print(f"Saved comparison {pair_idx+1} to {save_path}")


def export_attention_statistics(analysis_results: Dict[str, any], 
                              output_dir: str,
                              filename: str = 'attention_stats.json'):
    """Export attention statistics to JSON file."""
    output_path = os.path.join(output_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    exportable_results = {
        'total_samples': analysis_results['total_samples'],
        'rollout_statistics': analysis_results['rollout_statistics'],
        'glca_statistics': analysis_results['glca_statistics'],
        'raw_data': {
            'rollout_means': [float(x) for x in analysis_results['raw_data']['rollout_means']],
            'rollout_stds': [float(x) for x in analysis_results['raw_data']['rollout_stds']],
            'glca_means': [float(x) for x in analysis_results['raw_data']['glca_means']],
            'glca_stds': [float(x) for x in analysis_results['raw_data']['glca_stds']],
            'labels': [int(x) for x in analysis_results['raw_data']['labels']]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(exportable_results, f, indent=2)
    
    print(f"Attention statistics exported to {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level='INFO')
    
    # Load model and configuration
    model, config = load_model_and_config(args.checkpoint, args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data loader
    dataloader = setup_data_loader(config, args.data_split, args.batch_size)
    
    # Execute based on mode
    if args.mode == 'analyze':
        analysis_results = analyze_attention_patterns(
            model, dataloader, args.num_samples, args.output_dir
        )
        
        if args.export_stats:
            export_attention_statistics(analysis_results, args.output_dir)
    
    elif args.mode == 'visualize':
        visualize_individual_attention(
            model, dataloader, args.attention_type, args.output_dir,
            args.num_samples, args.save_individual, args.colormap, args.overlay_alpha
        )
    
    elif args.mode == 'explain':
        pairs = None
        if args.compare_pairs:
            pairs = [(args.compare_pairs[i], args.compare_pairs[i+1]) 
                    for i in range(0, len(args.compare_pairs), 2)]
        
        explain_verification_decisions(
            model, config, args.output_dir, pairs, args.threshold
        )
    
    elif args.mode == 'compare':
        if not args.compare_pairs:
            print("Error: --compare-pairs required for compare mode")
            return
        
        pairs = [(args.compare_pairs[i], args.compare_pairs[i+1]) 
                for i in range(0, len(args.compare_pairs), 2)]
        
        compare_attention_patterns(model, config, args.output_dir, pairs)
    
    print(f"Attention visualization complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 