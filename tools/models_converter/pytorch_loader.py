#!/usr/bin/env python3
"""
PyTorch Model Loader for FireRed-VAD

Loads PyTorch .pth.tar models and CMVN statistics.
Requires PyTorch for loading .pth files.
"""

import sys
import pickle
import zipfile
import struct
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


class PyTorchModelLoader:
    """Load and parse PyTorch .pth.tar models for FireRed-VAD"""
    
    def __init__(self, model_path: Path, cmvn_path: Path):
        """
        Initialize loader
        
        Args:
            model_path: Path to model.pth.tar file
            cmvn_path: Path to cmvn.ark file
        """
        self.model_path = Path(model_path)
        self.cmvn_path = Path(cmvn_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.cmvn_path.exists():
            raise FileNotFoundError(f"CMVN file not found: {cmvn_path}")
    
    def load(self) -> Dict[str, np.ndarray]:
        """
        Load PyTorch model and extract weights
        
        Returns:
            Dictionary mapping layer names to weight arrays (NumPy)
            
        Raises:
            RuntimeError: If model loading fails
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch")
        
        try:
            # Use PyTorch's load function
            # It handles the ZIP format and pickle properly
            print(f"  Loading with PyTorch...")
            # weights_only=False because FireRedTeam models are trusted
            # and contain argparse.Namespace objects
            state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Convert to NumPy arrays
            weights = self._extract_weights(state_dict)
            
            print(f"  Loaded {len(weights)} tensors")
            return weights
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _extract_weights(self, state_dict) -> Dict[str, np.ndarray]:
        """
        Extract and convert PyTorch tensors to NumPy arrays
        
        Args:
            state_dict: PyTorch state dictionary
            
        Returns:
            Dictionary mapping layer names to NumPy arrays
        """
        weights = {}
        
        # Handle different state_dict formats
        if isinstance(state_dict, dict):
            # Check for common wrapper keys
            if 'model_state_dict' in state_dict:
                # Wrapped format: {'model_state_dict': OrderedDict(...), 'args': ...}
                tensors = state_dict['model_state_dict']
            elif 'model' in state_dict:
                # Alternative wrapped format
                tensors = state_dict['model']
            elif 'state_dict' in state_dict:
                # Another common format
                tensors = state_dict['state_dict']
            else:
                # Direct format - assume it's already the state dict
                tensors = state_dict
                
            # Convert each tensor
            for name, tensor in tensors.items():
                try:
                    if hasattr(tensor, 'numpy'):
                        # PyTorch tensor
                        weights[name] = tensor.numpy()
                    elif hasattr(tensor, 'cpu'):
                        # PyTorch tensor on GPU
                        weights[name] = tensor.cpu().numpy()
                    elif isinstance(tensor, np.ndarray):
                        # Already NumPy
                        weights[name] = tensor
                    else:
                        # Unsupported type, skip
                        continue
                    
                    # For lookahead filters, we need to reverse the time axis
                    # because GGML's conv1d_dw is cross-correlation, and FSMN lookahead
                    # mathematically needs a reversed kernel to correctly slide into the future.
                    if 'lookahead' in name:
                        # shape is (channels, 1, kernel_size)
                        weights[name] = np.ascontiguousarray(weights[name][:, :, ::-1])

                except Exception as e:
                    print(f"  Warning: Failed to convert tensor '{name}': {e}")
                    continue
        else:
            raise RuntimeError(f"Unexpected state_dict type: {type(state_dict)}")
        
        if len(weights) == 0:
            raise RuntimeError("No tensors extracted from model")
        
        # Validate and print summary
        total_params = sum(w.size for w in weights.values())
        print(f"  Total parameters: {total_params:,}")
        
        # Print layer shapes (first 5)
        for i, (name, tensor) in enumerate(list(weights.items())[:5]):
            print(f"    {name}: {tensor.shape} ({tensor.dtype})")
        if len(weights) > 5:
            print(f"    ... and {len(weights) - 5} more")
        
        return weights
    
    def load_cmvn(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CMVN (Cepstral Mean and Variance Normalization) statistics
        
        Kaldi stores CMVN as statistics matrix in binary format:
        - Header: 15 bytes (BDM header + format markers)
        - Data: 2x81 matrix of float64 (double precision)
        - Row 0: sum of features (80) + frame count (1)
        - Row 1: sum of squares (80) + 0.0 (1)
        
        We convert stats to mean and variance:
        - mean = sum / count
        - variance = (sum_squares / count) - mean²
        
        Returns:
            Tuple of (mean, variance) arrays, shape (80,) as float32
        """
        try:
            with open(self.cmvn_path, 'rb') as f:
                content = f.read()
            
            print(f"  CMVN file size: {len(content)} bytes")
            
            # Validate file size (15 byte header + 2*81*8 bytes data = 1311 bytes)
            expected_size = 15 + 2 * 81 * 8
            if len(content) != expected_size:
                raise RuntimeError(f"Unexpected file size: {len(content)}, expected {expected_size}")
            
            # Kaldi binary format: 15 byte header, then 2x81 float64 matrix
            offset = 15
            n_values = 2 * 81
            
            # Read as float64 (double precision)
            doubles = struct.unpack(f'<{n_values}d', content[offset:offset+n_values*8])
            matrix = np.array(doubles, dtype=np.float64).reshape(2, 81)
            
            # Extract statistics
            feature_sum = matrix[0, :80]
            feature_sumsq = matrix[1, :80]
            frame_count = matrix[0, 80]
            
            print(f"  Frame count: {frame_count:.0f}")
            
            if frame_count <= 0:
                raise RuntimeError(f"Invalid frame count: {frame_count}")
            
            # Compute mean and variance from statistics
            mean = feature_sum / frame_count
            variance = (feature_sumsq / frame_count) - (mean * mean)
            
            # Ensure variance is positive (numerical stability)
            variance = np.maximum(variance, 1e-10)
            
            # Convert to float32
            mean = mean.astype(np.float32)
            variance = variance.astype(np.float32)
            
            # Validate
            if not np.all(np.isfinite(mean)):
                raise RuntimeError("CMVN mean contains invalid values (NaN/Inf)")
            if not np.all(np.isfinite(variance)):
                raise RuntimeError("CMVN variance contains invalid values (NaN/Inf)")
            if not np.all(variance > 0):
                bad_indices = np.where(variance <= 0)[0]
                raise RuntimeError(f"CMVN variance contains non-positive values at indices: {bad_indices}")
            
            print(f"  CMVN mean range: [{mean.min():.3f}, {mean.max():.3f}]")
            print(f"  CMVN variance range: [{variance.min():.3f}, {variance.max():.3f}]")
            print(f"  [OK] CMVN loaded successfully")
            
            return mean, variance
                
        except Exception as e:
            print(f"  ERROR: CMVN loading failed: {e}")
            raise RuntimeError(f"Failed to load CMVN: {e}")


def test_loader():
    """Test the loader with a real model"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pytorch_loader.py <model.pth.tar> <cmvn.ark>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    cmvn_path = Path(sys.argv[2])
    
    print(f"Testing PyTorchModelLoader")
    print(f"  Model: {model_path}")
    print(f"  CMVN: {cmvn_path}")
    print()
    
    try:
        loader = PyTorchModelLoader(model_path, cmvn_path)
        
        print("Loading model weights...")
        weights = loader.load()
        print(f"[OK] Loaded {len(weights)} tensors")
        print()
        
        print("Loading CMVN statistics...")
        mean, variance = loader.load_cmvn()
        print(f"[OK] CMVN shapes: mean={mean.shape}, variance={variance.shape}")
        print()
        
        print("SUCCESS: All loading tests passed!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_loader()
