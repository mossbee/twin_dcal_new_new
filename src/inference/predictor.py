"""
Optimized inference pipeline for DCAL Twin Faces Verification.

This module provides optimized inference capabilities including batch processing,
model quantization, ONNX export, and TensorRT optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import json
import os
from pathlib import Path
import cv2
from PIL import Image
import logging
from tqdm import tqdm
import warnings

# Optional imports for optimization
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX not available. Export functionality will be limited.")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn("TensorRT not available. TensorRT optimization will be disabled.")

from models.siamese_dcal import SiameseDCAL
from utils.config import Config
from data.transforms import get_transforms


class DCALPredictor:
    """
    Base predictor class for DCAL twin face verification.
    
    Provides basic inference functionality with optimizations for single
    image pair predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: str = 'cuda',
        half_precision: bool = False,
        compile_model: bool = False
    ):
        """
        Initialize DCAL predictor.
        
        Args:
            model: Pre-trained DCAL model
            config: Configuration object
            device: Device to run inference on
            half_precision: Whether to use FP16 inference
            compile_model: Whether to compile model (PyTorch 2.0+)
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.half_precision = half_precision
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Apply optimizations
        if half_precision:
            self.model = self.model.half()
        
        if compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Setup transforms
        self.transform = get_transforms(config, is_training=False)
        
        # Performance tracking
        self.inference_times = []
        self.preprocessing_times = []
        
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image tensor
        """
        start_time = time.time()
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Convert to half precision if needed
        if self.half_precision:
            image_tensor = image_tensor.half()
        
        self.preprocessing_times.append(time.time() - start_time)
        
        return image_tensor
    
    def predict_pair(
        self,
        image1: Union[str, np.ndarray, Image.Image],
        image2: Union[str, np.ndarray, Image.Image],
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Predict similarity for a single image pair.
        
        Args:
            image1: First image
            image2: Second image
            return_features: Whether to return feature vectors
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Preprocess images
        img1_tensor = self.preprocess_image(image1).unsqueeze(0).to(self.device)
        img2_tensor = self.preprocess_image(image2).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                img1_tensor,
                img2_tensor,
                return_features=return_features,
                return_attention=return_attention
            )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Process outputs
        similarity = outputs['similarity'].cpu().numpy()[0]
        prediction = similarity > self.model.optimal_threshold.cpu().numpy()
        
        results = {
            'similarity': float(similarity),
            'prediction': bool(prediction),
            'confidence': float(abs(similarity - 0.5)),
            'inference_time': inference_time
        }
        
        if return_features:
            results['features1'] = outputs['features1'].cpu().numpy()
            results['features2'] = outputs['features2'].cpu().numpy()
        
        if return_attention:
            results['attention_maps'] = {
                k: v.cpu().numpy() for k, v in outputs['attention_maps'].items()
            }
        
        return results
    
    def predict_batch(
        self,
        image_pairs: List[Tuple[Any, Any]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict similarity for multiple image pairs.
        
        Args:
            image_pairs: List of (image1, image2) tuples
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of prediction results
        """
        results = []
        
        iterator = tqdm(
            range(0, len(image_pairs), batch_size),
            desc="Predicting",
            disable=not show_progress
        )
        
        for i in iterator:
            batch_pairs = image_pairs[i:i + batch_size]
            batch_results = self._predict_batch_internal(batch_pairs)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(self, batch_pairs: List[Tuple[Any, Any]]) -> List[Dict[str, Any]]:
        """Internal batch prediction method."""
        # Preprocess all images
        images1 = []
        images2 = []
        
        for img1, img2 in batch_pairs:
            images1.append(self.preprocess_image(img1))
            images2.append(self.preprocess_image(img2))
        
        # Stack into batches
        batch1 = torch.stack(images1).to(self.device)
        batch2 = torch.stack(images2).to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(batch1, batch2)
        inference_time = time.time() - start_time
        
        # Process outputs
        similarities = outputs['similarity'].cpu().numpy()
        predictions = similarities > self.model.optimal_threshold.cpu().numpy()
        
        results = []
        for i in range(len(batch_pairs)):
            results.append({
                'similarity': float(similarities[i]),
                'prediction': bool(predictions[i]),
                'confidence': float(abs(similarities[i] - 0.5)),
                'inference_time': inference_time / len(batch_pairs)
            })
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'avg_preprocessing_time': np.mean(self.preprocessing_times) if self.preprocessing_times else 0,
            'total_predictions': len(self.inference_times)
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times = []
        self.preprocessing_times = []


class BatchPredictor(DCALPredictor):
    """
    Optimized batch predictor for high-throughput inference.
    
    Provides additional optimizations for batch processing including
    dynamic batching and memory management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: str = 'cuda',
        half_precision: bool = False,
        compile_model: bool = False,
        max_batch_size: int = 64,
        memory_threshold: float = 0.8
    ):
        """
        Initialize batch predictor.
        
        Args:
            model: Pre-trained DCAL model
            config: Configuration object
            device: Device to run inference on
            half_precision: Whether to use FP16 inference
            compile_model: Whether to compile model
            max_batch_size: Maximum batch size
            memory_threshold: Memory usage threshold for batch size adjustment
        """
        super().__init__(model, config, device, half_precision, compile_model)
        
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.optimal_batch_size = self._find_optimal_batch_size()
        
    def _find_optimal_batch_size(self) -> int:
        """Find optimal batch size based on memory constraints."""
        if self.device.type == 'cpu':
            return self.max_batch_size
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > self.max_batch_size:
                break
            
            try:
                # Create dummy batch
                dummy_batch = torch.randn(
                    batch_size, 3, 224, 224,
                    device=self.device,
                    dtype=torch.half if self.half_precision else torch.float
                )
                
                # Test inference
                with torch.no_grad():
                    _ = self.model(dummy_batch, dummy_batch)
                
                # Check memory usage
                if self.device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated(self.device)
                    memory_total = torch.cuda.get_device_properties(self.device).total_memory
                    
                    if memory_used / memory_total > self.memory_threshold:
                        return max(1, batch_size // 2)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    return max(1, batch_size // 2)
                raise e
        
        return self.max_batch_size
    
    def predict_batch(
        self,
        image_pairs: List[Tuple[Any, Any]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        adaptive_batching: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict similarity for multiple image pairs with optimization.
        
        Args:
            image_pairs: List of (image1, image2) tuples
            batch_size: Batch size (uses optimal if None)
            show_progress: Whether to show progress bar
            adaptive_batching: Whether to use adaptive batch sizing
            
        Returns:
            List of prediction results
        """
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        results = []
        current_batch_size = batch_size
        
        iterator = tqdm(
            range(0, len(image_pairs), current_batch_size),
            desc="Batch Predicting",
            disable=not show_progress
        )
        
        for i in iterator:
            batch_pairs = image_pairs[i:i + current_batch_size]
            
            try:
                batch_results = self._predict_batch_internal(batch_pairs)
                results.extend(batch_results)
                
                # Increase batch size if successful and adaptive
                if adaptive_batching and current_batch_size < self.max_batch_size:
                    current_batch_size = min(current_batch_size * 2, self.max_batch_size)
                    
            except RuntimeError as e:
                if "out of memory" in str(e) and adaptive_batching:
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    torch.cuda.empty_cache()
                    
                    # Retry with smaller batch
                    batch_results = self._predict_batch_internal(batch_pairs[:current_batch_size])
                    results.extend(batch_results)
                    
                    # Process remaining items
                    remaining = batch_pairs[current_batch_size:]
                    if remaining:
                        remaining_results = self.predict_batch(
                            remaining, current_batch_size, False, False
                        )
                        results.extend(remaining_results)
                else:
                    raise e
        
        return results


class OptimizedPredictor(BatchPredictor):
    """
    Highly optimized predictor with all performance enhancements.
    
    Includes quantization, TensorRT optimization, and advanced caching.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: str = 'cuda',
        half_precision: bool = False,
        compile_model: bool = False,
        max_batch_size: int = 64,
        memory_threshold: float = 0.8,
        use_quantization: bool = False,
        use_tensorrt: bool = False,
        cache_size: int = 1000
    ):
        """
        Initialize optimized predictor.
        
        Args:
            model: Pre-trained DCAL model
            config: Configuration object
            device: Device to run inference on
            half_precision: Whether to use FP16 inference
            compile_model: Whether to compile model
            max_batch_size: Maximum batch size
            memory_threshold: Memory usage threshold
            use_quantization: Whether to use quantization
            use_tensorrt: Whether to use TensorRT
            cache_size: Size of feature cache
        """
        super().__init__(
            model, config, device, half_precision, compile_model,
            max_batch_size, memory_threshold
        )
        
        self.use_quantization = use_quantization
        self.use_tensorrt = use_tensorrt
        self.cache_size = cache_size
        
        # Feature cache for repeated images
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Apply optimizations
        if use_quantization:
            self._apply_quantization()
        
        if use_tensorrt and TENSORRT_AVAILABLE:
            self._apply_tensorrt()
    
    def _apply_quantization(self):
        """Apply dynamic quantization to model."""
        if self.device.type == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logging.info("Applied dynamic quantization")
        else:
            logging.warning("Quantization not supported on CUDA. Skipping.")
    
    def _apply_tensorrt(self):
        """Apply TensorRT optimization."""
        if not TENSORRT_AVAILABLE:
            logging.warning("TensorRT not available. Skipping TensorRT optimization.")
            return
        
        # This would require converting to TensorRT format
        # Implementation depends on specific TensorRT integration
        logging.info("TensorRT optimization would be applied here")
    
    def _get_image_hash(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """Get hash for image caching."""
        if isinstance(image, str):
            return f"file:{image}"
        elif isinstance(image, np.ndarray):
            return f"array:{hash(image.tobytes())}"
        else:
            return f"pil:{hash(image.tobytes())}"
    
    def _extract_features(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """Extract features with caching."""
        image_hash = self._get_image_hash(image)
        
        # Check cache
        if image_hash in self.feature_cache:
            self.cache_hits += 1
            return self.feature_cache[image_hash]
        
        # Extract features
        img_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_single(img_tensor)
        
        # Cache features
        if len(self.feature_cache) < self.cache_size:
            self.feature_cache[image_hash] = features
        
        self.cache_misses += 1
        return features
    
    def predict_pair(
        self,
        image1: Union[str, np.ndarray, Image.Image],
        image2: Union[str, np.ndarray, Image.Image],
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Predict similarity with caching optimization.
        
        Args:
            image1: First image
            image2: Second image
            return_features: Whether to return feature vectors
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Extract features with caching
        features1 = self._extract_features(image1)
        features2 = self._extract_features(image2)
        
        # Compute similarity
        with torch.no_grad():
            similarity = self.model.compute_similarity(features1, features2)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Process results
        similarity_score = similarity.cpu().numpy()[0]
        prediction = similarity_score > self.model.optimal_threshold.cpu().numpy()
        
        results = {
            'similarity': float(similarity_score),
            'prediction': bool(prediction),
            'confidence': float(abs(similarity_score - 0.5)),
            'inference_time': inference_time,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
        }
        
        if return_features:
            results['features1'] = features1.cpu().numpy()
            results['features2'] = features2.cpu().numpy()
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics including cache performance."""
        stats = super().get_performance_stats()
        
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            stats.update({
                'cache_hit_rate': self.cache_hits / total_cache_requests,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_size': len(self.feature_cache)
            })
        
        return stats
    
    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class ExportManager:
    """
    Model export manager for different formats.
    
    Supports PyTorch, ONNX, and TensorRT export formats.
    """
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Initialize export manager.
        
        Args:
            model: Model to export
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.model.eval()
    
    def export_pytorch(self, save_path: str, scripted: bool = True):
        """
        Export model in PyTorch format.
        
        Args:
            save_path: Path to save model
            scripted: Whether to use TorchScript
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if scripted:
            # Create example input
            example_input = (
                torch.randn(1, 3, 224, 224),
                torch.randn(1, 3, 224, 224)
            )
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, example_input)
            traced_model.save(str(save_path))
            logging.info(f"Exported TorchScript model to {save_path}")
        else:
            # Save state dict
            torch.save(self.model.state_dict(), save_path)
            logging.info(f"Exported PyTorch state dict to {save_path}")
    
    def export_onnx(
        self,
        save_path: str,
        opset_version: int = 11,
        dynamic_axes: bool = True,
        optimize: bool = True
    ):
        """
        Export model in ONNX format.
        
        Args:
            save_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes
            optimize: Whether to optimize the model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Please install onnx and onnxruntime.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create example input
        example_input = (
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224)
        )
        
        # Define dynamic axes
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input1': {0: 'batch_size'},
                'input2': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            example_input,
            str(save_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input1', 'input2'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict
        )
        
        # Optimize if requested
        if optimize:
            self._optimize_onnx(save_path)
        
        logging.info(f"Exported ONNX model to {save_path}")
    
    def _optimize_onnx(self, model_path: Path):
        """Optimize ONNX model."""
        try:
            # Load and optimize
            model = onnx.load(str(model_path))
            
            # Check model
            onnx.checker.check_model(model)
            
            # Apply optimizations
            optimized_model = onnx.optimizer.optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, str(model_path))
            logging.info(f"Optimized ONNX model saved to {model_path}")
            
        except Exception as e:
            logging.warning(f"ONNX optimization failed: {e}")
    
    def export_tensorrt(self, save_path: str, max_batch_size: int = 1):
        """
        Export model in TensorRT format.
        
        Args:
            save_path: Path to save TensorRT engine
            max_batch_size: Maximum batch size
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Please install TensorRT.")
        
        # This would require implementing TensorRT conversion
        # The actual implementation depends on specific TensorRT integration
        logging.info(f"TensorRT export would be implemented here for {save_path}")
    
    def benchmark_formats(
        self,
        test_data: List[Tuple[Any, Any]],
        export_dir: str = "./exports",
        iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different export formats.
        
        Args:
            test_data: Test data for benchmarking
            export_dir: Directory to save exported models
            iterations: Number of benchmark iterations
            
        Returns:
            Dictionary containing benchmark results
        """
        results = {}
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Original PyTorch model
        results['pytorch'] = self._benchmark_pytorch(test_data, iterations)
        
        # TorchScript model
        script_path = export_dir / "model_scripted.pt"
        self.export_pytorch(str(script_path), scripted=True)
        results['torchscript'] = self._benchmark_torchscript(str(script_path), test_data, iterations)
        
        # ONNX model
        if ONNX_AVAILABLE:
            onnx_path = export_dir / "model.onnx"
            self.export_onnx(str(onnx_path))
            results['onnx'] = self._benchmark_onnx(str(onnx_path), test_data, iterations)
        
        return results
    
    def _benchmark_pytorch(self, test_data: List[Tuple[Any, Any]], iterations: int) -> Dict[str, float]:
        """Benchmark original PyTorch model."""
        predictor = DCALPredictor(self.model, self.config)
        
        # Warmup
        for _ in range(10):
            predictor.predict_pair(test_data[0][0], test_data[0][1])
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.time()
            predictor.predict_pair(test_data[0][0], test_data[0][1])
            times.append(time.time() - start_time)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
    
    def _benchmark_torchscript(self, model_path: str, test_data: List[Tuple[Any, Any]], iterations: int) -> Dict[str, float]:
        """Benchmark TorchScript model."""
        model = torch.jit.load(model_path)
        model.eval()
        
        # Create transforms
        transform = get_transforms(self.config, is_training=False)
        
        # Preprocess test data
        img1 = transform(Image.open(test_data[0][0]) if isinstance(test_data[0][0], str) else test_data[0][0])
        img2 = transform(Image.open(test_data[0][1]) if isinstance(test_data[0][1], str) else test_data[0][1])
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(img1.unsqueeze(0), img2.unsqueeze(0))
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.time()
            with torch.no_grad():
                model(img1.unsqueeze(0), img2.unsqueeze(0))
            times.append(time.time() - start_time)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
    
    def _benchmark_onnx(self, model_path: str, test_data: List[Tuple[Any, Any]], iterations: int) -> Dict[str, float]:
        """Benchmark ONNX model."""
        session = ort.InferenceSession(model_path)
        
        # Create transforms
        transform = get_transforms(self.config, is_training=False)
        
        # Preprocess test data
        img1 = transform(Image.open(test_data[0][0]) if isinstance(test_data[0][0], str) else test_data[0][0])
        img2 = transform(Image.open(test_data[0][1]) if isinstance(test_data[0][1], str) else test_data[0][1])
        
        input1 = img1.unsqueeze(0).numpy()
        input2 = img2.unsqueeze(0).numpy()
        
        # Warmup
        for _ in range(10):
            session.run(None, {'input1': input1, 'input2': input2})
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.time()
            session.run(None, {'input1': input1, 'input2': input2})
            times.append(time.time() - start_time)
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        } 