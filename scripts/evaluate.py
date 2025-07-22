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
    parser.add_argument('--hard-pairs-only', action='store_true',
                       help='Evaluate only on hard pairs (same person or twin persons)')
    
    return parser.parse_args()


def create_model(config: dict) -> SiameseDCAL:
    model_config = config['model']
    data_config = config['data']
    dcal_config = config['dcal']
    # Use 'num_layers' for VisionTransformer, fallback to 'depth' if needed
    num_layers = model_config.get('num_layers', model_config.get('depth', 12))
    # Prepare DCALEncoder arguments
    dcal_encoder = DCALEncoder(
        backbone_config=model_config.get('backbone', 'vit_base_patch16_224'),
        num_sa_blocks=dcal_config.get('num_sa_blocks', 12),
        num_glca_blocks=dcal_config.get('num_glca_blocks', 1),
        num_pwca_blocks=dcal_config.get('num_pwca_blocks', 12),
        local_ratio=dcal_config.get('local_ratio_fgvc', 0.1),
        embed_dim=model_config.get('embed_dim', 768),
        num_heads=model_config.get('num_heads', 12),
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        dropout=model_config.get('dropout', 0.1),
        pretrained=model_config.get('pretrained', True),
        pretrained_path=model_config.get('pretrained_path', None),
        num_layers=num_layers,
        use_dynamic_loss=dcal_config.get('use_dynamic_loss', True)
    )
    model = SiameseDCAL(
        dcal_encoder=dcal_encoder,
        similarity_function=model_config.get('similarity_function', 'cosine'),
        feature_dim=model_config.get('embed_dim', 768),
        dropout=model_config.get('dropout', 0.1),
        temperature=model_config.get('temperature', 0.07),
        learnable_temperature=model_config.get('learnable_temperature', True)
    )
    return model


def load_model(model_path: str, config: dict, device: str) -> SiameseDCAL:
    """Load trained model from checkpoint."""
    model = create_model(config)
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
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
        same_person_only=False,
        hard_pairs_only=getattr(args, 'hard_pairs_only', False)
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

    # Print main metrics
    print("\n=== Main Verification Metrics ===")
    try:
        auc = report['analysis']['roc_auc']
        eer = report['analysis']['eer']
        acc = report['analysis']['accuracy']
        print(f"AUC: {auc:.4f}" if auc is not None else "AUC: N/A")
        print(f"EER: {eer:.4f}" if eer is not None else "EER: N/A")
        print(f"Accuracy: {acc:.4f}" if acc is not None else "Accuracy: N/A")
    except Exception as e:
        print(f"[Warning] Could not compute main metrics: {e}")

    # Print hard pair metrics (if available)
    if 'different_twins' in hard_pair_metrics:
        print("\n=== Hard Pair (Twin) Verification Metrics ===")
        try:
            auc = hard_pair_metrics['different_twins'].get('roc_auc', None)
            eer = hard_pair_metrics['different_twins'].get('eer', None)
            acc = hard_pair_metrics['different_twins'].get('accuracy', None)
            print(f"AUC: {auc:.4f}" if auc is not None else "AUC: N/A")
            print(f"EER: {eer:.4f}" if eer is not None else "EER: N/A")
            print(f"Accuracy: {acc:.4f}" if acc is not None else "Accuracy: N/A")
        except Exception as e:
            print(f"[Warning] Could not compute hard pair metrics: {e}")
    else:
        print("\nNo hard (twin) pairs found in the test set.")


if __name__ == '__main__':
    main() 