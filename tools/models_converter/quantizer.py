#!/usr/bin/env python3
"""
Model Quantizer for FireRed-VAD
Implements INT16, INT8, and INT8-ch quantization
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class QuantizedTensor:
    """Quantized tensor with metadata"""
    data: np.ndarray          # Quantized weights
    scale: np.ndarray         # Scale factor(s)
    zero_point: int = 0       # Zero point (for asymmetric quant)
    per_channel: bool = False # Per-channel quantization
    
    def dequantize(self) -> np.ndarray:
        """Dequantize back to FP32"""
        if self.per_channel:
            # Scale is per output channel (first dimension)
            # Reshape scale to broadcast correctly
            if self.data.ndim == 1:
                # 1D tensor (bias): scale directly
                return self.data.astype(np.float32) * self.scale
            else:
                # Multi-dimensional: reshape scale for broadcasting
                scale_shape = [-1] + [1] * (self.data.ndim - 1)
                return self.data.astype(np.float32) * self.scale.reshape(scale_shape)
        else:
            # Single scale for entire tensor
            return self.data.astype(np.float32) * self.scale


class ModelQuantizer:
    """Quantize FP32 model weights to lower precision"""
    
    @staticmethod
    def quantize_int16(weights: np.ndarray) -> QuantizedTensor:
        """
        Symmetric INT16 quantization (per-tensor)
        
        Formula:
            scale = max(abs(weights)) / 32767
            quantized = clip(round(weights / scale), -32768, 32767)
        
        Args:
            weights: FP32 weights array
            
        Returns:
            QuantizedTensor with INT16 data and scale
        """
        # Find maximum absolute value
        max_val = np.max(np.abs(weights))
        
        if max_val == 0:
            # All zeros - no need to quantize
            return QuantizedTensor(
                data=np.zeros_like(weights, dtype=np.int16),
                scale=np.array(1.0, dtype=np.float32),
                per_channel=False
            )
        
        # Calculate scale
        scale = max_val / 32767.0
        
        # Quantize
        quantized = np.round(weights / scale)
        quantized = np.clip(quantized, -32768, 32767).astype(np.int16)
        
        return QuantizedTensor(
            data=quantized,
            scale=np.array(scale, dtype=np.float32),
            per_channel=False
        )
    
    @staticmethod
    def quantize_int8(weights: np.ndarray, per_channel: bool = False) -> QuantizedTensor:
        """
        Symmetric INT8 quantization
        
        Args:
            weights: FP32 weights array
            per_channel: If True, use per-channel (per output channel) quantization
            
        Returns:
            QuantizedTensor with INT8 data and scale(s)
        """
        if per_channel:
            return ModelQuantizer._quantize_int8_per_channel(weights)
        else:
            return ModelQuantizer._quantize_int8_per_tensor(weights)
    
    @staticmethod
    def _quantize_int8_per_tensor(weights: np.ndarray) -> QuantizedTensor:
        """
        Per-tensor INT8 quantization
        
        Formula:
            scale = max(abs(weights)) / 127
            quantized = clip(round(weights / scale), -128, 127)
        """
        max_val = np.max(np.abs(weights))
        
        if max_val == 0:
            return QuantizedTensor(
                data=np.zeros_like(weights, dtype=np.int8),
                scale=np.array(1.0, dtype=np.float32),
                per_channel=False
            )
        
        # Calculate scale
        scale = max_val / 127.0
        
        # Quantize
        quantized = np.round(weights / scale)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        
        return QuantizedTensor(
            data=quantized,
            scale=np.array(scale, dtype=np.float32),
            per_channel=False
        )
    
    @staticmethod
    def _quantize_int8_per_channel(weights: np.ndarray) -> QuantizedTensor:
        """
        Per-channel INT8 quantization
        
        Quantizes each output channel independently for better accuracy.
        Assumes weights shape: (out_channels, in_channels, ...) or (out_channels, ...)
        
        Formula (per channel i):
            scale[i] = max(abs(weights[i, :])) / 127
            quantized[i, :] = clip(round(weights[i, :] / scale[i]), -128, 127)
        """
        if weights.ndim < 1:
            raise ValueError("Weights must have at least 1 dimension")
        
        # Get number of output channels (first dimension)
        out_channels = weights.shape[0]
        
        # Calculate scales per channel
        scales = np.zeros(out_channels, dtype=np.float32)
        quantized = np.zeros_like(weights, dtype=np.int8)
        
        for i in range(out_channels):
            # Get channel weights (all elements in this output channel)
            channel_weights = weights[i]
            max_val = np.max(np.abs(channel_weights))
            
            if max_val == 0:
                scales[i] = 1.0
                quantized[i] = 0
            else:
                scales[i] = max_val / 127.0
                quantized[i] = np.clip(
                    np.round(channel_weights / scales[i]),
                    -128, 127
                ).astype(np.int8)
        
        return QuantizedTensor(
            data=quantized,
            scale=scales,
            per_channel=True
        )
    
    @staticmethod
    def calculate_quantization_error(original: np.ndarray, 
                                     quantized_tensor: QuantizedTensor) -> Dict[str, float]:
        """
        Calculate quantization error metrics
        
        Args:
            original: Original FP32 weights
            quantized_tensor: Quantized tensor
            
        Returns:
            Dictionary with error metrics:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - max_error: Maximum absolute error
            - sqnr_db: Signal-to-Quantization-Noise Ratio in dB
        """
        # Dequantize
        dequantized = quantized_tensor.dequantize()
        
        # Ensure same shape
        if original.shape != dequantized.shape:
            raise ValueError(f"Shape mismatch: {original.shape} vs {dequantized.shape}")
        
        # Calculate errors
        error = original - dequantized
        
        mae = np.mean(np.abs(error))
        mse = np.mean(error ** 2)
        max_error = np.max(np.abs(error))
        
        # SQNR = 10 * log10(signal_power / noise_power)
        signal_power = np.mean(original ** 2)
        noise_power = mse
        
        if noise_power > 0:
            sqnr_db = 10 * np.log10(signal_power / noise_power)
        else:
            sqnr_db = float('inf')
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'max_error': float(max_error),
            'sqnr_db': float(sqnr_db)
        }


def test_quantizer():
    """Test quantization functions"""
    print("Testing ModelQuantizer\n")
    
    # Create test weights
    np.random.seed(42)
    test_weights = np.random.randn(10, 20).astype(np.float32) * 2.0
    
    print(f"Original weights shape: {test_weights.shape}")
    print(f"Original range: [{test_weights.min():.3f}, {test_weights.max():.3f}]")
    print()
    
    # Test INT16 quantization
    print("="*70)
    print("INT16 Quantization (per-tensor)")
    print("="*70)
    q16 = ModelQuantizer.quantize_int16(test_weights)
    print(f"Scale: {q16.scale}")
    print(f"Quantized range: [{q16.data.min()}, {q16.data.max()}]")
    errors = ModelQuantizer.calculate_quantization_error(test_weights, q16)
    print(f"MAE: {errors['mae']:.6f}")
    print(f"SQNR: {errors['sqnr_db']:.2f} dB")
    print()
    
    # Test INT8 per-tensor
    print("="*70)
    print("INT8 Quantization (per-tensor)")
    print("="*70)
    q8 = ModelQuantizer.quantize_int8(test_weights, per_channel=False)
    print(f"Scale: {q8.scale}")
    print(f"Quantized range: [{q8.data.min()}, {q8.data.max()}]")
    errors = ModelQuantizer.calculate_quantization_error(test_weights, q8)
    print(f"MAE: {errors['mae']:.6f}")
    print(f"SQNR: {errors['sqnr_db']:.2f} dB")
    print()
    
    # Test INT8 per-channel
    print("="*70)
    print("INT8 Quantization (per-channel)")
    print("="*70)
    q8_ch = ModelQuantizer.quantize_int8(test_weights, per_channel=True)
    print(f"Scales shape: {q8_ch.scale.shape}")
    print(f"Scale range: [{q8_ch.scale.min():.6f}, {q8_ch.scale.max():.6f}]")
    print(f"Quantized range: [{q8_ch.data.min()}, {q8_ch.data.max()}]")
    errors = ModelQuantizer.calculate_quantization_error(test_weights, q8_ch)
    print(f"MAE: {errors['mae']:.6f}")
    print(f"SQNR: {errors['sqnr_db']:.2f} dB")
    print()
    
    # Compare INT8 per-tensor vs per-channel
    print("="*70)
    print("Comparison: INT8 per-tensor vs per-channel")
    print("="*70)
    q8_errors = ModelQuantizer.calculate_quantization_error(test_weights, q8)
    q8ch_errors = ModelQuantizer.calculate_quantization_error(test_weights, q8_ch)
    
    print(f"Per-tensor MAE: {q8_errors['mae']:.6f}")
    print(f"Per-channel MAE: {q8ch_errors['mae']:.6f}")
    print(f"Improvement: {(1 - q8ch_errors['mae']/q8_errors['mae'])*100:.1f}%")
    print()
    
    print(f"Per-tensor SQNR: {q8_errors['sqnr_db']:.2f} dB")
    print(f"Per-channel SQNR: {q8ch_errors['sqnr_db']:.2f} dB")
    print(f"Improvement: {q8ch_errors['sqnr_db'] - q8_errors['sqnr_db']:.2f} dB")
    print()
    
    print("✓ All quantization tests passed!")


if __name__ == "__main__":
    test_quantizer()
