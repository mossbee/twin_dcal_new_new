"""
Comprehensive evaluation suite for DCAL Twin Faces Verification.

This module provides evaluation tools including cross-validation, performance
benchmarking, error analysis, and performance profiling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from scipy import interpolate
from scipy.optimize import brentq
import logging
from tqdm import tqdm
import warnings
from collections import defaultdict
import psutil
import gc

from models.siamese_dcal import SiameseDCAL
from utils.config import Config
from data.dataset import TwinDataset
from data.transforms import get_transforms
from inference.predictor import DCALPredictor, BatchPredictor
from torch.utils.data import DataLoader, Subset


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and analysis.
    
    Provides detailed evaluation including ROC-AUC, EER, precision, recall,
    and various twin-specific metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: str = 'cuda',
        save_results: bool = True,
        results_dir: str = "./evaluation_results"
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Model to evaluate
            config: Configuration object
            device: Device to run evaluation on
            save_results: Whether to save evaluation results
            results_dir: Directory to save results
        """
        self.model = model
        self.config = config
        self.device = device
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        
        if save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize predictor
        self.predictor = BatchPredictor(model, config, device)
        
        # Evaluation metrics
        self.metrics = {}
        self.predictions = []
        self.ground_truth = []
        self.similarities = []
        
    def evaluate_dataset(
        self,
        test_loader: DataLoader,
        threshold: Optional[float] = None,
        save_predictions: bool = True,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            test_loader: Test data loader
            threshold: Classification threshold (uses model's if None)
            save_predictions: Whether to save predictions
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.predictions = []
        self.ground_truth = []
        self.similarities = []
        
        # Use model's optimal threshold if not provided
        if threshold is None:
            threshold = self.model.optimal_threshold.item()
        
        # Run evaluation
        iterator = tqdm(test_loader, desc="Evaluating", disable=not show_progress)
        
        with torch.no_grad():
            for batch in iterator:
                if len(batch) == 2:
                    (img1, img2), labels = batch
                else:
                    img1, img2, labels = batch
                
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(img1, img2)
                similarities = outputs['similarity']
                
                # Store results
                self.similarities.extend(similarities.cpu().numpy())
                self.ground_truth.extend(labels.cpu().numpy())
                self.predictions.extend((similarities > threshold).cpu().numpy())
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(threshold)
        
        # Save results
        if save_predictions and self.save_results:
            self._save_predictions()
        
        return self.metrics
    
    def _calculate_metrics(self, threshold: float) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        y_scores = np.array(self.similarities)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        
        # ROC metrics
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics['roc_auc_manual'] = np.trapz(tpr, fpr)
        except ValueError:
            metrics['roc_auc'] = 0.0
            metrics['roc_auc_manual'] = 0.0
        
        # Equal Error Rate (EER)
        metrics['eer'] = self._calculate_eer(y_true, y_scores)
        
        # Precision-Recall metrics
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        metrics['pr_auc'] = np.trapz(precision, recall)
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_negative_rate'] = fn / (tp + fn) if (tp + fn) > 0 else 0
            metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Threshold-specific metrics
        metrics['threshold'] = threshold
        metrics['num_samples'] = len(y_true)
        metrics['positive_samples'] = int(np.sum(y_true))
        metrics['negative_samples'] = int(len(y_true) - np.sum(y_true))
        
        # Distribution metrics
        metrics['similarity_mean'] = np.mean(y_scores)
        metrics['similarity_std'] = np.std(y_scores)
        metrics['similarity_min'] = np.min(y_scores)
        metrics['similarity_max'] = np.max(y_scores)
        
        # Twin-specific metrics
        metrics.update(self._calculate_twin_metrics(y_true, y_pred, y_scores))
        
        return metrics
    
    def _calculate_eer(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate Equal Error Rate."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        
        # Find EER
        try:
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        except ValueError:
            # Fallback method
            diff = np.abs(fpr - fnr)
            eer_idx = np.argmin(diff)
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return float(eer)
    
    def _calculate_twin_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate twin-specific evaluation metrics."""
        metrics = {}
        
        # Separate same-twin and different-twin predictions
        same_twin_mask = y_true == 1
        diff_twin_mask = y_true == 0
        
        if np.any(same_twin_mask):
            same_twin_scores = y_scores[same_twin_mask]
            same_twin_preds = y_pred[same_twin_mask]
            
            metrics['same_twin_accuracy'] = np.mean(same_twin_preds)
            metrics['same_twin_score_mean'] = np.mean(same_twin_scores)
            metrics['same_twin_score_std'] = np.std(same_twin_scores)
        
        if np.any(diff_twin_mask):
            diff_twin_scores = y_scores[diff_twin_mask]
            diff_twin_preds = y_pred[diff_twin_mask]
            
            metrics['diff_twin_accuracy'] = np.mean(1 - diff_twin_preds)
            metrics['diff_twin_score_mean'] = np.mean(diff_twin_scores)
            metrics['diff_twin_score_std'] = np.std(diff_twin_scores)
        
        # Separation metrics
        if np.any(same_twin_mask) and np.any(diff_twin_mask):
            same_scores = y_scores[same_twin_mask]
            diff_scores = y_scores[diff_twin_mask]
            
            # Score separation
            metrics['score_separation'] = np.mean(same_scores) - np.mean(diff_scores)
            
            # Overlap analysis
            same_min, same_max = np.min(same_scores), np.max(same_scores)
            diff_min, diff_max = np.min(diff_scores), np.max(diff_scores)
            
            overlap_start = max(same_min, diff_min)
            overlap_end = min(same_max, diff_max)
            
            if overlap_start < overlap_end:
                metrics['score_overlap'] = overlap_end - overlap_start
            else:
                metrics['score_overlap'] = 0.0
        
        return metrics
    
    def _save_predictions(self):
        """Save predictions to file."""
        results_data = {
            'predictions': self.predictions,
            'ground_truth': self.ground_truth,
            'similarities': self.similarities,
            'metrics': self.metrics
        }
        
        # Save as JSON
        json_path = self.results_dir / 'predictions.json'
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=float)
        
        # Save as CSV
        csv_path = self.results_dir / 'predictions.csv'
        df = pd.DataFrame({
            'ground_truth': self.ground_truth,
            'prediction': self.predictions,
            'similarity': self.similarities
        })
        df.to_csv(csv_path, index=False)
        
        logging.info(f"Predictions saved to {json_path} and {csv_path}")
    
    def plot_results(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate evaluation plots.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing plot figures
        """
        figures = {}
        
        # ROC Curve
        figures['roc'] = self._plot_roc_curve()
        
        # Precision-Recall Curve
        figures['pr'] = self._plot_pr_curve()
        
        # Similarity Distribution
        figures['similarity_dist'] = self._plot_similarity_distribution()
        
        # Confusion Matrix
        figures['confusion_matrix'] = self._plot_confusion_matrix()
        
        # Twin-specific analysis
        figures['twin_analysis'] = self._plot_twin_analysis()
        
        # Save plots
        if save_plots and self.save_results:
            for name, fig in figures.items():
                fig.savefig(self.results_dir / f'{name}.png', dpi=300, bbox_inches='tight')
                fig.savefig(self.results_dir / f'{name}.pdf', bbox_inches='tight')
        
        return figures
    
    def _plot_roc_curve(self) -> plt.Figure:
        """Plot ROC curve."""
        y_true = np.array(self.ground_truth)
        y_scores = np.array(self.similarities)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_pr_curve(self) -> plt.Figure:
        """Plot Precision-Recall curve."""
        y_true = np.array(self.ground_truth)
        y_scores = np.array(self.similarities)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = np.trapz(precision, recall)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_similarity_distribution(self) -> plt.Figure:
        """Plot similarity score distribution."""
        y_true = np.array(self.ground_truth)
        y_scores = np.array(self.similarities)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distributions
        same_twin_scores = y_scores[y_true == 1]
        diff_twin_scores = y_scores[y_true == 0]
        
        ax.hist(same_twin_scores, bins=50, alpha=0.7, label='Same Twin', density=True)
        ax.hist(diff_twin_scores, bins=50, alpha=0.7, label='Different Twin', density=True)
        
        # Add threshold line
        threshold = self.metrics.get('threshold', 0.5)
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
        
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title('Similarity Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_confusion_matrix(self) -> plt.Figure:
        """Plot confusion matrix."""
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Different Twin', 'Same Twin'])
        ax.set_yticklabels(['Different Twin', 'Same Twin'])
        
        return fig
    
    def _plot_twin_analysis(self) -> plt.Figure:
        """Plot twin-specific analysis."""
        y_true = np.array(self.ground_truth)
        y_scores = np.array(self.similarities)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Score distributions by class
        same_twin_scores = y_scores[y_true == 1]
        diff_twin_scores = y_scores[y_true == 0]
        
        axes[0, 0].boxplot([same_twin_scores, diff_twin_scores], labels=['Same Twin', 'Different Twin'])
        axes[0, 0].set_title('Score Distribution by Class')
        axes[0, 0].set_ylabel('Similarity Score')
        
        # Score vs Ground Truth
        axes[0, 1].scatter(y_true, y_scores, alpha=0.5)
        axes[0, 1].set_xlabel('Ground Truth')
        axes[0, 1].set_ylabel('Similarity Score')
        axes[0, 1].set_title('Similarity vs Ground Truth')
        
        # Error analysis
        errors = (y_true != np.array(self.predictions))
        error_scores = y_scores[errors]
        correct_scores = y_scores[~errors]
        
        axes[1, 0].hist(error_scores, bins=30, alpha=0.7, label='Errors', density=True)
        axes[1, 0].hist(correct_scores, bins=30, alpha=0.7, label='Correct', density=True)
        axes[1, 0].set_xlabel('Similarity Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Error Analysis')
        axes[1, 0].legend()
        
        # Threshold analysis
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        
        for thresh in thresholds:
            pred_thresh = (y_scores > thresh).astype(int)
            accuracies.append(accuracy_score(y_true, pred_thresh))
        
        axes[1, 1].plot(thresholds, accuracies)
        axes[1, 1].axvline(self.metrics.get('threshold', 0.5), color='red', linestyle='--')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Threshold vs Accuracy')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("DCAL Twin Faces Verification - Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Basic metrics
        report.append("Classification Metrics:")
        report.append(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        report.append(f"  Precision: {self.metrics['precision']:.4f}")
        report.append(f"  Recall: {self.metrics['recall']:.4f}")
        report.append(f"  F1-Score: {self.metrics['f1']:.4f}")
        report.append("")
        
        # ROC metrics
        report.append("ROC Metrics:")
        report.append(f"  ROC-AUC: {self.metrics['roc_auc']:.4f}")
        report.append(f"  Equal Error Rate: {self.metrics['eer']:.4f}")
        report.append(f"  PR-AUC: {self.metrics['pr_auc']:.4f}")
        report.append("")
        
        # Dataset statistics
        report.append("Dataset Statistics:")
        report.append(f"  Total Samples: {self.metrics['num_samples']}")
        report.append(f"  Positive Samples: {self.metrics['positive_samples']}")
        report.append(f"  Negative Samples: {self.metrics['negative_samples']}")
        report.append("")
        
        # Twin-specific metrics
        report.append("Twin-Specific Metrics:")
        if 'same_twin_accuracy' in self.metrics:
            report.append(f"  Same Twin Accuracy: {self.metrics['same_twin_accuracy']:.4f}")
        if 'diff_twin_accuracy' in self.metrics:
            report.append(f"  Different Twin Accuracy: {self.metrics['diff_twin_accuracy']:.4f}")
        if 'score_separation' in self.metrics:
            report.append(f"  Score Separation: {self.metrics['score_separation']:.4f}")
        report.append("")
        
        # Similarity statistics
        report.append("Similarity Statistics:")
        report.append(f"  Mean: {self.metrics['similarity_mean']:.4f}")
        report.append(f"  Std: {self.metrics['similarity_std']:.4f}")
        report.append(f"  Min: {self.metrics['similarity_min']:.4f}")
        report.append(f"  Max: {self.metrics['similarity_max']:.4f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        if self.save_results:
            report_path = self.results_dir / 'evaluation_report.txt'
            with open(report_path, 'w') as f:
                f.write(report_text)
            logging.info(f"Evaluation report saved to {report_path}")
        
        return report_text


class CrossValidationEvaluator:
    """
    Cross-validation evaluator for robust performance assessment.
    
    Provides k-fold cross-validation with comprehensive metrics.
    """
    
    def __init__(
        self,
        model_class: type,
        config: Config,
        device: str = 'cuda',
        k_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize cross-validation evaluator.
        
        Args:
            model_class: Model class to instantiate
            config: Configuration object
            device: Device to run evaluation on
            k_folds: Number of folds for cross-validation
            random_state: Random state for reproducibility
        """
        self.model_class = model_class
        self.config = config
        self.device = device
        self.k_folds = k_folds
        self.random_state = random_state
        
        # Results storage
        self.fold_results = []
        self.summary_results = {}
    
    def run_cross_validation(
        self,
        dataset: TwinDataset,
        train_function: Callable,
        save_results: bool = True,
        results_dir: str = "./cv_results"
    ) -> Dict[str, Any]:
        """
        Run k-fold cross-validation.
        
        Args:
            dataset: Dataset to evaluate on
            train_function: Function to train model
            save_results: Whether to save results
            results_dir: Directory to save results
            
        Returns:
            Dictionary containing cross-validation results
        """
        results_dir = Path(results_dir)
        if save_results:
            results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        
        # Get labels for stratification
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        
        # Run cross-validation
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            logging.info(f"Running fold {fold + 1}/{self.k_folds}")
            
            # Create train and validation sets
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            
            # Train model
            model = self.model_class(self.config)
            trained_model = train_function(model, train_dataset, val_dataset)
            
            # Evaluate
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            evaluator = ModelEvaluator(trained_model, self.config, self.device, False)
            metrics = evaluator.evaluate_dataset(val_loader, show_progress=False)
            
            # Store results
            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'metrics': metrics
            }
            fold_results.append(fold_result)
            
            # Save fold results
            if save_results:
                fold_dir = results_dir / f'fold_{fold + 1}'
                fold_dir.mkdir(exist_ok=True)
                
                with open(fold_dir / 'results.json', 'w') as f:
                    json.dump(fold_result, f, indent=2, default=float)
        
        # Calculate summary statistics
        self.fold_results = fold_results
        self.summary_results = self._calculate_summary_statistics()
        
        # Save summary results
        if save_results:
            with open(results_dir / 'cv_summary.json', 'w') as f:
                json.dump(self.summary_results, f, indent=2, default=float)
            
            # Generate report
            report = self._generate_cv_report()
            with open(results_dir / 'cv_report.txt', 'w') as f:
                f.write(report)
        
        return self.summary_results
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across folds."""
        # Collect metrics from all folds
        all_metrics = defaultdict(list)
        
        for fold_result in self.fold_results:
            metrics = fold_result['metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        # Calculate statistics
        summary = {}
        for metric, values in all_metrics.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values
            }
        
        return summary
    
    def _generate_cv_report(self) -> str:
        """Generate cross-validation report."""
        report = []
        report.append("DCAL Twin Faces Verification - Cross-Validation Report")
        report.append("=" * 70)
        report.append("")
        report.append(f"Number of folds: {self.k_folds}")
        report.append("")
        
        # Key metrics
        key_metrics = ['accuracy', 'roc_auc', 'eer', 'f1', 'precision', 'recall']
        
        for metric in key_metrics:
            if metric in self.summary_results:
                stats = self.summary_results[metric]
                report.append(f"{metric.upper()}:")
                report.append(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                report.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                report.append(f"  Median: {stats['median']:.4f}")
                report.append("")
        
        # Fold details
        report.append("Fold Details:")
        for fold_result in self.fold_results:
            fold_num = fold_result['fold']
            metrics = fold_result['metrics']
            report.append(f"  Fold {fold_num}:")
            report.append(f"    Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            report.append(f"    EER: {metrics['eer']:.4f}")
        
        return "\n".join(report)


class BenchmarkEvaluator:
    """
    Performance benchmarking evaluator.
    
    Provides comprehensive performance benchmarking including speed,
    memory usage, and accuracy trade-offs.
    """
    
    def __init__(self, config: Config, device: str = 'cuda'):
        """
        Initialize benchmark evaluator.
        
        Args:
            config: Configuration object
            device: Device to run benchmarks on
        """
        self.config = config
        self.device = device
        self.benchmark_results = {}
    
    def benchmark_model(
        self,
        model: nn.Module,
        test_data: List[Tuple[Any, Any]],
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            test_data: Test data for benchmarking
            batch_sizes: Batch sizes to test
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary containing benchmark results
        """
        results = {}
        
        for batch_size in batch_sizes:
            logging.info(f"Benchmarking batch size: {batch_size}")
            
            # Create predictor
            predictor = BatchPredictor(model, self.config, self.device)
            
            # Prepare batch data
            batch_data = test_data[:batch_size]
            
            # Warmup
            for _ in range(warmup_iterations):
                try:
                    predictor.predict_batch(batch_data, show_progress=False)
                except Exception as e:
                    logging.warning(f"Warmup failed for batch size {batch_size}: {e}")
                    break
            
            # Benchmark
            times = []
            memory_usage = []
            
            for _ in range(num_iterations):
                # Clear cache
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                # Measure memory before
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated(self.device)
                else:
                    mem_before = psutil.virtual_memory().used
                
                # Run inference
                start_time = time.time()
                try:
                    predictions = predictor.predict_batch(batch_data, show_progress=False)
                    end_time = time.time()
                    
                    # Measure memory after
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                        mem_after = torch.cuda.memory_allocated(self.device)
                    else:
                        mem_after = psutil.virtual_memory().used
                    
                    times.append(end_time - start_time)
                    memory_usage.append(mem_after - mem_before)
                    
                except Exception as e:
                    logging.warning(f"Benchmark failed for batch size {batch_size}: {e}")
                    break
            
            # Calculate statistics
            if times:
                results[batch_size] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'throughput': batch_size / np.mean(times),
                    'avg_memory': np.mean(memory_usage),
                    'max_memory': np.max(memory_usage),
                    'batch_size': batch_size
                }
        
        self.benchmark_results = results
        return results
    
    def plot_benchmark_results(self, save_plots: bool = True, output_dir: str = "./benchmark_results") -> Dict[str, Any]:
        """
        Plot benchmark results.
        
        Args:
            save_plots: Whether to save plots
            output_dir: Directory to save plots
            
        Returns:
            Dictionary containing plot figures
        """
        if not self.benchmark_results:
            logging.warning("No benchmark results to plot")
            return {}
        
        figures = {}
        
        # Extract data
        batch_sizes = list(self.benchmark_results.keys())
        avg_times = [self.benchmark_results[bs]['avg_time'] for bs in batch_sizes]
        throughputs = [self.benchmark_results[bs]['throughput'] for bs in batch_sizes]
        memory_usage = [self.benchmark_results[bs]['avg_memory'] / 1024**2 for bs in batch_sizes]  # MB
        
        # Time vs Batch Size
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(batch_sizes, avg_times, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Average Time (s)')
        ax.set_title('Inference Time vs Batch Size')
        ax.grid(True, alpha=0.3)
        figures['time_vs_batch'] = fig
        
        # Throughput vs Batch Size
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (samples/s)')
        ax.set_title('Throughput vs Batch Size')
        ax.grid(True, alpha=0.3)
        figures['throughput_vs_batch'] = fig
        
        # Memory Usage vs Batch Size
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(batch_sizes, memory_usage, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage vs Batch Size')
        ax.grid(True, alpha=0.3)
        figures['memory_vs_batch'] = fig
        
        # Combined plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(batch_sizes, avg_times, 'o-')
        axes[0, 0].set_title('Inference Time')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(batch_sizes, throughputs, 'o-', color='green')
        axes[0, 1].set_title('Throughput')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Samples/s')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(batch_sizes, memory_usage, 'o-', color='red')
        axes[1, 0].set_title('Memory Usage')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency plot (throughput per memory)
        efficiency = [t / m for t, m in zip(throughputs, memory_usage)]
        axes[1, 1].plot(batch_sizes, efficiency, 'o-', color='purple')
        axes[1, 1].set_title('Efficiency (Throughput/Memory)')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['combined'] = fig
        
        # Save plots
        if save_plots:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
                fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
        
        return figures


class ErrorAnalyzer:
    """
    Error analysis tool for understanding model failures.
    
    Provides detailed analysis of misclassified samples and failure modes.
    """
    
    def __init__(self, model: nn.Module, config: Config, device: str = 'cuda'):
        """
        Initialize error analyzer.
        
        Args:
            model: Model to analyze
            config: Configuration object
            device: Device to run analysis on
        """
        self.model = model
        self.config = config
        self.device = device
        self.error_analysis = {}
    
    def analyze_errors(
        self,
        test_loader: DataLoader,
        save_results: bool = True,
        output_dir: str = "./error_analysis"
    ) -> Dict[str, Any]:
        """
        Analyze model errors.
        
        Args:
            test_loader: Test data loader
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing error analysis results
        """
        # Run evaluation to get predictions
        evaluator = ModelEvaluator(self.model, self.config, self.device, False)
        metrics = evaluator.evaluate_dataset(test_loader, show_progress=True)
        
        # Analyze errors
        y_true = np.array(evaluator.ground_truth)
        y_pred = np.array(evaluator.predictions)
        y_scores = np.array(evaluator.similarities)
        
        # Find errors
        errors = (y_true != y_pred)
        error_indices = np.where(errors)[0]
        
        # Classify error types
        false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
        false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
        
        # Error analysis
        analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(y_true),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'fp_rate': len(false_positives) / len(y_true),
            'fn_rate': len(false_negatives) / len(y_true)
        }
        
        # Score distribution analysis
        if len(error_indices) > 0:
            error_scores = y_scores[error_indices]
            correct_scores = y_scores[~errors]
            
            analysis['error_score_stats'] = {
                'mean': np.mean(error_scores),
                'std': np.std(error_scores),
                'min': np.min(error_scores),
                'max': np.max(error_scores)
            }
            
            analysis['correct_score_stats'] = {
                'mean': np.mean(correct_scores),
                'std': np.std(correct_scores),
                'min': np.min(correct_scores),
                'max': np.max(correct_scores)
            }
        
        # Confidence analysis
        confidences = np.abs(y_scores - 0.5)
        error_confidences = confidences[errors]
        correct_confidences = confidences[~errors]
        
        analysis['confidence_analysis'] = {
            'error_confidence_mean': np.mean(error_confidences) if len(error_confidences) > 0 else 0,
            'correct_confidence_mean': np.mean(correct_confidences) if len(correct_confidences) > 0 else 0,
            'low_confidence_errors': np.sum(error_confidences < 0.1) if len(error_confidences) > 0 else 0,
            'high_confidence_errors': np.sum(error_confidences > 0.4) if len(error_confidences) > 0 else 0
        }
        
        self.error_analysis = analysis
        
        # Save results
        if save_results:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'error_analysis.json', 'w') as f:
                json.dump(analysis, f, indent=2, default=float)
        
        return analysis
    
    def plot_error_analysis(self, save_plots: bool = True, output_dir: str = "./error_analysis") -> Dict[str, Any]:
        """
        Plot error analysis results.
        
        Args:
            save_plots: Whether to save plots
            output_dir: Directory to save plots
            
        Returns:
            Dictionary containing plot figures
        """
        if not self.error_analysis:
            logging.warning("No error analysis results to plot")
            return {}
        
        figures = {}
        
        # Error distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        correct = 1 - self.error_analysis['error_rate']
        fp_rate = self.error_analysis['fp_rate']
        fn_rate = self.error_analysis['fn_rate']
        
        labels = ['Correct', 'False Positive', 'False Negative']
        sizes = [correct, fp_rate, fn_rate]
        colors = ['lightgreen', 'lightcoral', 'lightsalmon']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Error Distribution')
        figures['error_distribution'] = fig
        
        # Save plots
        if save_plots:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
                fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
        
        return figures 