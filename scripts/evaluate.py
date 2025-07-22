#!/usr/bin/env python3
"""
Evaluation script for DCAL Twin Face Verification.
Evaluates trained models on test datasets and generates performance reports.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import DCALEncoder, SiameseDCAL, VisionTransformer
from src.data import TwinDataset, TwinVerificationDataset, get_val_transforms
from src.training import calculate_verification_metrics, VerificationAnalyzer
from src.utils import Config, get_config, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DCAL Twin Face Verification Model')
    
    # Model and checkpoint
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/kaggle_config.yaml',
                       help='Path to configuration file')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for data')
    parser.add_argument('--test-dataset-info', type=str, default='data/test_dataset_infor.json',
                       help='Path to test dataset info file')
    parser.add_argument('--test-twin-pairs', type=str, default='data/test_twin_pairs.json',
                       help='Path to test twin pairs file')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size for evaluation')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    
    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=None,
                       help='Classification threshold (if None, uses optimal)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to CSV file')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save performance plots')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_model(model_path: str, config: dict, device: str) -> SiameseDCAL:
    """Load trained model from checkpoint."""
    model_config = config['model']
    
    # Create backbone
    # patch_size fallback: try model_config, else data config
    patch_size = model_config.get('patch_size', config['data'].get('patch_size', 16))
    backbone = VisionTransformer(
        img_size=config['data']['image_size'],
        patch_size=patch_size,
        in_channels=3,
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config.get('mlp_ratio', 4),
        dropout=model_config['dropout'],
        pretrained=False  # Don't load pretrained weights for evaluation
    )
    
    # Create DCAL encoder
    dcal_encoder = DCALEncoder(
        backbone=backbone,
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_sa_blocks=model_config['num_sa_blocks'],
        num_glca_blocks=model_config['num_glca_blocks'],
        num_pwca_blocks=model_config['num_pwca_blocks'],
        local_ratio_fgvc=model_config['local_ratio_fgvc'],
        local_ratio_reid=model_config['local_ratio_reid'],
        dropout=model_config['dropout']
    )
    
    # Create Siamese DCAL model
    model = SiameseDCAL(
        dcal_encoder=dcal_encoder,
        similarity_function=model_config['similarity_function'],
        feature_dim=model_config['feature_dim'],
        dropout=model_config['dropout'],
        temperature=model_config['temperature'],
        learnable_temperature=model_config['learnable_temperature']
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load model state
    model.load_state_dict(state_dict, strict=False)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    return model


def create_test_loader(config: dict, args: argparse.Namespace) -> DataLoader:
    """Create test data loader."""
    # Create transform
    transform = get_val_transforms(image_size=args.image_size)
    
    # Create dataset
    test_dataset = TwinVerificationDataset(
        dataset_info_path=args.test_dataset_info,
        twin_pairs_path=args.test_twin_pairs,
        data_root=args.data_root,
        transform=transform,
        same_person_only=False  # Include all pairs for comprehensive evaluation
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return test_loader


def evaluate_model(model: SiameseDCAL, test_loader: DataLoader, device: str) -> dict:
    """Evaluate model on test dataset."""
    model.eval()
    
    all_similarities = []
    all_labels = []
    all_person_pairs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            img1, img2, labels, person1, person2 = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Forward pass
            output = model(img1, img2)
            similarities = output['similarity'].cpu().numpy()
            
            # Store results
            all_similarities.extend(similarities)
            all_labels.extend(labels.cpu().numpy())
            all_person_pairs.extend(list(zip(person1, person2)))
    
    # Convert to numpy arrays
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    return {
        'similarities': all_similarities,
        'labels': all_labels,
        'person_pairs': all_person_pairs
    }


def generate_performance_report(results: dict, threshold: float = None, output_dir: str = None) -> dict:
    """Generate comprehensive performance report."""
    similarities = results['similarities']
    labels = results['labels']
    person_pairs = results['person_pairs']
    
    # Create verification analyzer
    analyzer = VerificationAnalyzer()
    
    # Analyze performance
    analysis = analyzer.analyze_verification_performance(
        similarities=similarities,
        labels=labels,
        person_pairs=person_pairs,
        calibrate_scores=True,
        optimize_thresholds=True
    )
    
    # Use specified threshold or optimal
    if threshold is None:
        threshold = analysis['optimal_thresholds']['f1']
    
    # Calculate metrics at threshold
    metrics = calculate_verification_metrics(labels, similarities, threshold)
    
    # Create summary report
    report = {
        'threshold': threshold,
        'metrics': metrics,
        'analysis': analysis,
        'dataset_stats': {
            'total_pairs': len(similarities),
            'positive_pairs': int(np.sum(labels == 1)),
            'negative_pairs': int(np.sum(labels == 0)),
            'positive_ratio': float(np.mean(labels == 1))
        }
    }
    
    # Save plots if requested
    if output_dir:
        plot_path = os.path.join(output_dir, 'performance_analysis.png')
        analyzer.plot_performance_analysis(similarities, labels, save_path=plot_path)
    
    return report


def save_predictions(results: dict, threshold: float, output_dir: str):
    """Save predictions to CSV file."""
    similarities = results['similarities']
    labels = results['labels']
    person_pairs = results['person_pairs']
    
    predictions = (similarities >= threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'person1': [pair[0] for pair in person_pairs],
        'person2': [pair[1] for pair in person_pairs],
        'true_label': labels,
        'similarity_score': similarities,
        'prediction': predictions,
        'correct': (predictions == labels).astype(int)
    })
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    
    # Save summary statistics
    summary_stats = {
        'accuracy': float(np.mean(predictions == labels)),
        'precision': float(np.sum((predictions == 1) & (labels == 1)) / np.sum(predictions == 1)) if np.sum(predictions == 1) > 0 else 0,
        'recall': float(np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)) if np.sum(labels == 1) > 0 else 0,
        'f1_score': float(2 * np.sum((predictions == 1) & (labels == 1)) / (np.sum(predictions == 1) + np.sum(labels == 1))) if (np.sum(predictions == 1) + np.sum(labels == 1)) > 0 else 0
    }
    
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Predictions saved to: {csv_path}")
    print(f"Summary statistics saved to: {os.path.join(output_dir, 'summary_stats.json')}")


def print_performance_summary(report: dict):
    """Print performance summary to console."""
    metrics = report['metrics']
    threshold = report['threshold']
    
    print("\n" + "="*60)
    print("DCAL Twin Face Verification - Evaluation Results")
    print("="*60)
    
    print(f"\nThreshold: {threshold:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"True Positive: {metrics['true_positive']}")
    print(f"False Positive: {metrics['false_positive']}")
    print(f"True Negative: {metrics['true_negative']}")
    print(f"False Negative: {metrics['false_negative']}")
    
    print(f"\nSensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    
    # Dataset statistics
    dataset_stats = report['dataset_stats']
    print(f"\nDataset Statistics:")
    print(f"Total Pairs: {dataset_stats['total_pairs']}")
    print(f"Positive Pairs: {dataset_stats['positive_pairs']}")
    print(f"Negative Pairs: {dataset_stats['negative_pairs']}")
    print(f"Positive Ratio: {dataset_stats['positive_ratio']:.4f}")
    
    # Optimal thresholds
    if 'optimal_thresholds' in report['analysis']:
        print(f"\nOptimal Thresholds:")
        for metric, threshold_val in report['analysis']['optimal_thresholds'].items():
            print(f"{metric}: {threshold_val:.4f}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging({
        'log_level': 'INFO',
        'log_file': os.path.join(args.output_dir, 'evaluation.log')
    })
    
    logger.info("Starting DCAL Twin Face Verification evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load configuration
    config = get_config(args.config)
    config_dict = config.to_dict()
    
    # Override config with command line arguments
    config_dict['data']['image_size'] = args.image_size
    config_dict['data']['data_root'] = args.data_root
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path, config_dict, device)
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test loader
    logger.info("Creating test data loader...")
    test_loader = create_test_loader(config_dict, args)
    logger.info(f"Test dataset: {len(test_loader.dataset)} pairs")
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Generate performance report
    logger.info("Generating performance report...")
    report = generate_performance_report(
        results, 
        threshold=args.threshold,
        output_dir=args.output_dir if args.save_plots else None
    )
    
    # Print summary
    print_performance_summary(report)
    
    # Save predictions if requested
    if args.save_predictions:
        logger.info("Saving predictions...")
        save_predictions(results, report['threshold'], args.output_dir)
    
    # Save full report
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Evaluation completed. Results saved to: {args.output_dir}")

    # --- Hard Pair (Twin) Evaluation ---
    # Load twin pairs for hard pair evaluation
    with open(args.test_twin_pairs, 'r') as f:
        twin_pairs = json.load(f)
    # Ensure tuple format for TwinSpecificEvaluator
    twin_pairs = [tuple(pair) for pair in twin_pairs]

    from src.training.verification import TwinSpecificEvaluator
    twin_evaluator = TwinSpecificEvaluator(twin_pairs)
    # Evaluate only on hard pairs (different people from same twin pair)
    hard_pair_metrics = twin_evaluator.evaluate_twin_difficulty(
        similarities=np.array(results['similarities']),
        labels=np.array(results['labels']),
        person_pairs=[(p1, p2) for p1, p2 in results['person_pairs']],
        threshold=report['threshold'] if 'threshold' in report else 0.5
    )
    if 'different_twins' in hard_pair_metrics:
        print("\n=== Hard Pair (Twin) Verification Metrics ===")
        for metric, value in hard_pair_metrics['different_twins'].items():
            print(f"{metric}: {value:.4f}")
    else:
        print("\nNo hard (twin) pairs found in the test set.")


if __name__ == '__main__':
    main() 