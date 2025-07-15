"""
Inference module for DCAL Twin Faces Verification.

This module provides optimized inference pipelines, evaluation tools, and 
performance analysis utilities for the DCAL twin faces verification system.
"""

from .predictor import (
    DCALPredictor,
    BatchPredictor,
    OptimizedPredictor,
    ExportManager
)

from .evaluator import (
    ModelEvaluator,
    CrossValidationEvaluator,
    BenchmarkEvaluator,
    ErrorAnalyzer,
    PerformanceProfiler
)

__all__ = [
    # Predictors
    'DCALPredictor',
    'BatchPredictor', 
    'OptimizedPredictor',
    'ExportManager',
    
    # Evaluators
    'ModelEvaluator',
    'CrossValidationEvaluator',
    'BenchmarkEvaluator',
    'ErrorAnalyzer',
    'PerformanceProfiler'
] 