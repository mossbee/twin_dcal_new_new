"""
Verification-specific features for twin face verification.
Includes similarity score calibration, threshold optimization, and cross-validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import calculate_eer, calculate_roc_auc, calculate_verification_metrics
from ..models import SiameseDCAL


class SimilarityCalibrator:
    """
    Calibrate similarity scores for better probability estimates.
    
    Uses Platt scaling (logistic regression) or isotonic regression
    to map similarity scores to calibrated probabilities.
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Initialize similarity calibrator.
        
        Args:
            method: Calibration method ('platt' or 'isotonic')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, similarities: np.ndarray, labels: np.ndarray):
        """
        Fit the calibrator on similarity scores and labels.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
        """
        if self.method == 'platt':
            # Platt scaling using logistic regression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(similarities.reshape(-1, 1), labels)
        elif self.method == 'isotonic':
            # Isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(similarities, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
    
    def predict_proba(self, similarities: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities from similarity scores.
        
        Args:
            similarities: Similarity scores [N]
        
        Returns:
            Calibrated probabilities [N]
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        if self.method == 'platt':
            return self.calibrator.predict_proba(similarities.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.predict(similarities)
    
    def transform(self, similarities: np.ndarray) -> np.ndarray:
        """
        Transform similarity scores to calibrated probabilities.
        
        Args:
            similarities: Similarity scores [N]
        
        Returns:
            Calibrated probabilities [N]
        """
        return self.predict_proba(similarities)


class ThresholdOptimizer:
    """
    Optimize classification threshold based on different criteria.
    """
    
    def __init__(self):
        self.optimal_thresholds = {}
    
    def optimize_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        metric: str = 'f1',
        return_score: bool = False
    ) -> Tuple[float, Optional[float]]:
        """
        Optimize threshold for a given metric.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
            metric: Metric to optimize ('f1', 'accuracy', 'eer', 'youden')
            return_score: Whether to return the score at optimal threshold
        
        Returns:
            Optimal threshold and optionally the score
        """
        if metric == 'eer':
            # Equal Error Rate optimization
            fpr, tpr, thresholds = roc_curve(labels, similarities)
            fnr = 1 - tpr
            
            # Find threshold where FPR = FNR
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            optimal_threshold = thresholds[eer_idx]
            
            if return_score:
                eer_score = fpr[eer_idx]  # EER value
                return optimal_threshold, eer_score
            else:
                return optimal_threshold
        
        elif metric == 'youden':
            # Youden's J statistic (TPR - FPR)
            fpr, tpr, thresholds = roc_curve(labels, similarities)
            youden_scores = tpr - fpr
            
            optimal_idx = np.argmax(youden_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            if return_score:
                return optimal_threshold, youden_scores[optimal_idx]
            else:
                return optimal_threshold
        
        else:
            # Grid search for other metrics
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
            
            best_score = 0
            best_threshold = 0.5
            
            # Use ROC thresholds for efficiency
            _, _, thresholds = roc_curve(labels, similarities)
            
            for threshold in thresholds:
                predictions = (similarities >= threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(labels, predictions, zero_division=0)
                elif metric == 'accuracy':
                    score = accuracy_score(labels, predictions)
                elif metric == 'precision':
                    score = precision_score(labels, predictions, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(labels, predictions, zero_division=0)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            if return_score:
                return best_threshold, best_score
            else:
                return best_threshold
    
    def optimize_multiple_thresholds(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        metrics: List[str] = ['f1', 'accuracy', 'eer', 'youden']
    ) -> Dict[str, float]:
        """
        Optimize thresholds for multiple metrics.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
            metrics: List of metrics to optimize
        
        Returns:
            Dictionary of metric -> optimal threshold
        """
        thresholds = {}
        
        for metric in metrics:
            thresholds[metric] = self.optimize_threshold(similarities, labels, metric)
        
        return thresholds


class CrossValidator:
    """
    Cross-validation for twin face verification.
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def cross_validate_model(
        self,
        model: SiameseDCAL,
        X: np.ndarray,
        y: np.ndarray,
        evaluation_fn: callable,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.
        
        Args:
            model: SiameseDCAL model
            X: Input features or data indices
            y: Labels
            evaluation_fn: Function to evaluate model performance
            **kwargs: Additional arguments for evaluation function
        
        Returns:
            Dictionary of metric -> list of scores across folds
        """
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            print(f"Fold {fold_idx + 1}/{self.n_splits}")
            
            # Get fold data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Evaluate model on this fold
            fold_metrics = evaluation_fn(model, X_train, y_train, X_val, y_val, **kwargs)
            fold_results.append(fold_metrics)
        
        # Aggregate results
        aggregated_results = {}
        for metric in fold_results[0].keys():
            aggregated_results[metric] = [fold[metric] for fold in fold_results]
        
        return aggregated_results
    
    def cross_validate_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        Cross-validate threshold optimization.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
            metric: Metric to optimize
        
        Returns:
            Dictionary with mean and std of optimal thresholds
        """
        threshold_optimizer = ThresholdOptimizer()
        optimal_thresholds = []
        
        for train_idx, val_idx in self.cv.split(similarities, labels):
            # Optimize threshold on training fold
            train_similarities = similarities[train_idx]
            train_labels = labels[train_idx]
            
            optimal_threshold = threshold_optimizer.optimize_threshold(
                train_similarities, train_labels, metric
            )
            optimal_thresholds.append(optimal_threshold)
        
        return {
            'mean_threshold': np.mean(optimal_thresholds),
            'std_threshold': np.std(optimal_thresholds),
            'thresholds': optimal_thresholds
        }


class TwinSpecificEvaluator:
    """
    Specialized evaluator for twin face verification.
    """
    
    def __init__(self, twin_pairs: List[Tuple[str, str]]):
        """
        Initialize twin-specific evaluator.
        
        Args:
            twin_pairs: List of twin pairs (person1, person2)
        """
        self.twin_pairs = twin_pairs
        self.twin_pair_map = {pair[0]: pair[1] for pair in twin_pairs}
        self.twin_pair_map.update({pair[1]: pair[0] for pair in twin_pairs})
    
    def evaluate_twin_difficulty(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        person_pairs: List[Tuple[str, str]],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate performance on different types of twin pairs.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
            person_pairs: List of person pairs for each sample
            threshold: Classification threshold
        
        Returns:
            Dictionary with detailed twin-specific metrics
        """
        results = {}
        
        # Categorize pairs
        same_twin_mask = []
        different_twin_mask = []
        twin_vs_nontwin_mask = []
        nontwin_vs_nontwin_mask = []
        
        for i, (person1, person2) in enumerate(person_pairs):
            if person1 == person2:
                # Same person (should not happen in verification)
                continue
            elif person1 in self.twin_pair_map and self.twin_pair_map[person1] == person2:
                # Different people from same twin pair
                different_twin_mask.append(i)
            elif (person1 in self.twin_pair_map) != (person2 in self.twin_pair_map):
                # One twin, one non-twin
                twin_vs_nontwin_mask.append(i)
            else:
                # Both non-twins or both twins from different pairs
                nontwin_vs_nontwin_mask.append(i)
        
        # Evaluate each category
        categories = {
            'different_twins': different_twin_mask,
            'twin_vs_nontwin': twin_vs_nontwin_mask,
            'nontwin_vs_nontwin': nontwin_vs_nontwin_mask
        }
        
        for category, mask in categories.items():
            if len(mask) > 0:
                cat_similarities = similarities[mask]
                cat_labels = labels[mask]
                
                # Calculate metrics for this category
                cat_metrics = calculate_verification_metrics(cat_labels, cat_similarities, threshold)
                results[category] = cat_metrics
        
        return results
    
    def plot_similarity_distributions(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        person_pairs: List[Tuple[str, str]],
        save_path: Optional[str] = None
    ):
        """
        Plot similarity score distributions for different pair types.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
            person_pairs: List of person pairs for each sample
            save_path: Path to save the plot
        """
        # Categorize pairs
        same_person_sims = []
        different_twin_sims = []
        different_nontwin_sims = []
        
        for i, (person1, person2) in enumerate(person_pairs):
            if labels[i] == 1:  # Same person
                same_person_sims.append(similarities[i])
            elif person1 in self.twin_pair_map and self.twin_pair_map[person1] == person2:
                # Different people from same twin pair
                different_twin_sims.append(similarities[i])
            else:
                # Different people (non-twins)
                different_nontwin_sims.append(similarities[i])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot distributions
        if same_person_sims:
            plt.hist(same_person_sims, bins=50, alpha=0.7, label='Same Person', density=True)
        if different_twin_sims:
            plt.hist(different_twin_sims, bins=50, alpha=0.7, label='Different Twins', density=True)
        if different_nontwin_sims:
            plt.hist(different_nontwin_sims, bins=50, alpha=0.7, label='Different Non-twins', density=True)
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Similarity Score Distributions by Pair Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class VerificationAnalyzer:
    """
    Comprehensive analyzer for twin face verification results.
    """
    
    def __init__(self):
        self.calibrator = SimilarityCalibrator()
        self.threshold_optimizer = ThresholdOptimizer()
    
    def analyze_verification_performance(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        person_pairs: Optional[List[Tuple[str, str]]] = None,
        calibrate_scores: bool = True,
        optimize_thresholds: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of verification performance.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
            person_pairs: List of person pairs (optional)
            calibrate_scores: Whether to calibrate similarity scores
            optimize_thresholds: Whether to optimize thresholds
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {}
        
        # Basic metrics
        results['basic_metrics'] = calculate_verification_metrics(labels, similarities)
        
        # Calibration
        if calibrate_scores:
            self.calibrator.fit(similarities, labels)
            calibrated_scores = self.calibrator.transform(similarities)
            results['calibrated_metrics'] = calculate_verification_metrics(labels, calibrated_scores)
            results['calibration_improvement'] = {
                'roc_auc': results['calibrated_metrics']['roc_auc'] - results['basic_metrics']['roc_auc'],
                'eer': results['basic_metrics']['eer'] - results['calibrated_metrics']['eer']
            }
        
        # Threshold optimization
        if optimize_thresholds:
            results['optimal_thresholds'] = self.threshold_optimizer.optimize_multiple_thresholds(
                similarities, labels
            )
        
        # Twin-specific analysis
        if person_pairs:
            twin_pairs = self._extract_twin_pairs(person_pairs)
            twin_evaluator = TwinSpecificEvaluator(twin_pairs)
            results['twin_specific'] = twin_evaluator.evaluate_twin_difficulty(
                similarities, labels, person_pairs
            )
        
        return results
    
    def _extract_twin_pairs(self, person_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Extract twin pairs from person pairs.
        
        Args:
            person_pairs: List of person pairs
        
        Returns:
            List of twin pairs
        """
        # Simple heuristic: consecutive IDs are twins
        twin_pairs = []
        processed = set()
        
        for person1, person2 in person_pairs:
            try:
                id1, id2 = int(person1), int(person2)
                if abs(id1 - id2) == 1:
                    pair = tuple(sorted([person1, person2]))
                    if pair not in processed:
                        twin_pairs.append(pair)
                        processed.add(pair)
            except ValueError:
                continue
        
        return twin_pairs
    
    def plot_performance_analysis(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive performance analysis.
        
        Args:
            similarities: Similarity scores [N]
            labels: Binary labels [N]
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True)
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(labels, similarities)
        avg_precision = average_precision_score(labels, similarities)
        
        axes[0, 1].plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc="lower left")
        axes[0, 1].grid(True)
        
        # Similarity Distribution
        axes[1, 0].hist(similarities[labels == 1], bins=50, alpha=0.7, label='Same Person', density=True)
        axes[1, 0].hist(similarities[labels == 0], bins=50, alpha=0.7, label='Different Person', density=True)
        axes[1, 0].set_xlabel('Similarity Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Similarity Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Threshold vs Metrics
        _, _, thresholds = roc_curve(labels, similarities)
        f1_scores = []
        accuracies = []
        
        from sklearn.metrics import f1_score, accuracy_score
        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            f1_scores.append(f1_score(labels, predictions, zero_division=0))
            accuracies.append(accuracy_score(labels, predictions))
        
        axes[1, 1].plot(thresholds, f1_scores, label='F1 Score')
        axes[1, 1].plot(thresholds, accuracies, label='Accuracy')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Threshold vs Performance Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 