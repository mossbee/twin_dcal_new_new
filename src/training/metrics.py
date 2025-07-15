"""
Evaluation metrics for twin face verification.
Implements EER, ROC-AUC, accuracy, and other verification metrics.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import brentq


def calculate_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) and its corresponding threshold.
    
    Args:
        labels: Ground truth binary labels
        scores: Prediction scores/probabilities
    
    Returns:
        Tuple of (EER, threshold)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Calculate EER
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, fnr)(x), 0., 1.)
    
    # Find threshold corresponding to EER
    eer_threshold = interpolate.interp1d(fpr, thresholds)(eer)
    
    return eer, eer_threshold


def calculate_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Calculate ROC-AUC score.
    
    Args:
        labels: Ground truth binary labels
        scores: Prediction scores/probabilities
    
    Returns:
        ROC-AUC score
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def calculate_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Args:
        labels: Ground truth binary labels
        predictions: Binary predictions
    
    Returns:
        Accuracy score
    """
    return accuracy_score(labels, predictions)


def calculate_precision_recall(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and average precision.
    
    Args:
        labels: Ground truth binary labels
        scores: Prediction scores/probabilities
    
    Returns:
        Tuple of (precision, recall, average_precision)
    """
    precision, recall, _ = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    
    return precision, recall, avg_precision


def find_optimal_threshold(labels: np.ndarray, scores: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal threshold for a given metric.
    
    Args:
        labels: Ground truth binary labels
        scores: Prediction scores/probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
    
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        
        if metric == 'f1':
            from sklearn.metrics import f1_score
            score = f1_score(labels, predictions)
        elif metric == 'accuracy':
            score = accuracy_score(labels, predictions)
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            score = precision_score(labels, predictions, zero_division=0)
        elif metric == 'recall':
            from sklearn.metrics import recall_score
            score = recall_score(labels, predictions, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def calculate_verification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive verification metrics.
    
    Args:
        labels: Ground truth binary labels
        scores: Prediction scores/probabilities
        threshold: Classification threshold (if None, uses optimal F1 threshold)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['roc_auc'] = calculate_roc_auc(labels, scores)
    metrics['eer'], metrics['eer_threshold'] = calculate_eer(labels, scores)
    
    # Precision-recall metrics
    _, _, metrics['average_precision'] = calculate_precision_recall(labels, scores)
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(labels, scores, metric='f1')
    
    metrics['threshold'] = threshold
    
    # Calculate metrics at threshold
    predictions = (scores >= threshold).astype(int)
    
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics['true_positive'] = tp
    metrics['false_positive'] = fp
    metrics['true_negative'] = tn
    metrics['false_negative'] = fn
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def calculate_twin_specific_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    person_pairs: List[Tuple[str, str]],
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate twin-specific verification metrics.
    
    Args:
        labels: Ground truth binary labels
        scores: Prediction scores/probabilities
        person_pairs: List of (person1, person2) tuples
        threshold: Classification threshold
    
    Returns:
        Dictionary of twin-specific metrics
    """
    metrics = {}
    
    # Basic verification metrics
    basic_metrics = calculate_verification_metrics(labels, scores, threshold)
    metrics.update(basic_metrics)
    
    # Twin-specific analysis
    if threshold is None:
        threshold = basic_metrics['threshold']
    
    predictions = (scores >= threshold).astype(int)
    
    # Analyze twin vs non-twin performance
    twin_mask = np.array([
        person1 != person2 and is_twin_pair(person1, person2)
        for person1, person2 in person_pairs
    ])
    
    if twin_mask.any():
        # Twin pairs (different twins)
        twin_labels = labels[twin_mask]
        twin_scores = scores[twin_mask]
        twin_predictions = predictions[twin_mask]
        
        if len(np.unique(twin_labels)) > 1:
            metrics['twin_accuracy'] = accuracy_score(twin_labels, twin_predictions)
            metrics['twin_roc_auc'] = calculate_roc_auc(twin_labels, twin_scores)
        
        # Non-twin pairs
        non_twin_mask = ~twin_mask
        non_twin_labels = labels[non_twin_mask]
        non_twin_scores = scores[non_twin_mask]
        non_twin_predictions = predictions[non_twin_mask]
        
        if len(np.unique(non_twin_labels)) > 1:
            metrics['non_twin_accuracy'] = accuracy_score(non_twin_labels, non_twin_predictions)
            metrics['non_twin_roc_auc'] = calculate_roc_auc(non_twin_labels, non_twin_scores)
    
    return metrics


def is_twin_pair(person1: str, person2: str) -> bool:
    """
    Check if two persons form a twin pair based on ID pattern.
    
    Args:
        person1: First person ID
        person2: Second person ID
    
    Returns:
        True if they are twins, False otherwise
    """
    # Simple heuristic: consecutive IDs are twins
    try:
        id1 = int(person1)
        id2 = int(person2)
        return abs(id1 - id2) == 1
    except ValueError:
        return False


class MetricsTracker:
    """
    Track and accumulate metrics during training/evaluation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.all_labels = []
        self.all_scores = []
        self.all_predictions = []
        self.all_person_pairs = []
        self.batch_metrics = []
    
    def update(
        self,
        labels: torch.Tensor,
        scores: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        person_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Update metrics with batch data.
        
        Args:
            labels: Ground truth labels
            scores: Prediction scores
            predictions: Binary predictions (optional)
            person_pairs: Person pairs for this batch (optional)
        """
        # Convert to numpy
        labels_np = labels.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        
        self.all_labels.extend(labels_np)
        self.all_scores.extend(scores_np)
        
        if predictions is not None:
            predictions_np = predictions.detach().cpu().numpy()
            self.all_predictions.extend(predictions_np)
        
        if person_pairs is not None:
            self.all_person_pairs.extend(person_pairs)
    
    def compute_metrics(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute accumulated metrics.
        
        Args:
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        if not self.all_labels:
            return {}
        
        labels = np.array(self.all_labels)
        scores = np.array(self.all_scores)
        
        # Calculate basic metrics
        metrics = calculate_verification_metrics(labels, scores, threshold)
        
        # Calculate twin-specific metrics if person pairs available
        if self.all_person_pairs:
            twin_metrics = calculate_twin_specific_metrics(
                labels, scores, self.all_person_pairs, threshold
            )
            metrics.update(twin_metrics)
        
        return metrics
    
    def get_best_threshold(self, metric: str = 'f1') -> float:
        """
        Get the best threshold for a given metric.
        
        Args:
            metric: Metric to optimize
        
        Returns:
            Best threshold
        """
        if not self.all_labels:
            return 0.5
        
        labels = np.array(self.all_labels)
        scores = np.array(self.all_scores)
        
        threshold, _ = find_optimal_threshold(labels, scores, metric)
        return threshold


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, title: str = "ROC Curve"):
    """
    Plot ROC curve.
    
    Args:
        labels: Ground truth labels
        scores: Prediction scores
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_precision_recall_curve(labels: np.ndarray, scores: np.ndarray, title: str = "Precision-Recall Curve"):
    """
    Plot Precision-Recall curve.
    
    Args:
        labels: Ground truth labels
        scores: Prediction scores
        title: Plot title
    """
    precision, recall, _ = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def print_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    target_names: List[str] = None
):
    """
    Print detailed classification report.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        target_names: Class names
    """
    from sklearn.metrics import classification_report
    
    if target_names is None:
        target_names = ['Different', 'Same']
    
    print(classification_report(labels, predictions, target_names=target_names)) 