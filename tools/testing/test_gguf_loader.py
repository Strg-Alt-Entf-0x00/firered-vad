#!/usr/bin/env python3
"""
GGUF Model Loader and Validator
Tests that generated GGUF files are valid and can be loaded
"""

import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class GGUFLoader:
    """Load and validate GGUF v3 files"""
    
    GGUF_MAGIC = 0x46554747  # "GGUF"
    GGUF_VERSION = 3
    
    # Value types
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12
    
    # Tensor types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_I8 = 24
    GGML_TYPE_I16 = 25
    
    def __init__(self, path: Path):
        """Initialize loader with GGUF file path"""
        self.path = Path(path)
        self.metadata = {}
        self.tensors = {}
        self.tensor_info = []
        
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {path}")
    
    def load(self) -> Dict:
        """
        Load and parse GGUF file
        
        Returns:
            Dictionary with metadata and tensor info
        """
        with open(self.path, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic: 0x{magic:08x}, expected 0x{self.GGUF_MAGIC:08x}")
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != self.GGUF_VERSION:
                raise ValueError(f"Unsupported GGUF version: {version}, expected {self.GGUF_VERSION}")
            
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata
            for _ in range(metadata_count):
                key, value = self._read_metadata_kv(f)
                self.metadata[key] = value
            
            # Read tensor info
            for _ in range(tensor_count):
                tensor_info = self._read_tensor_info(f)
                self.tensor_info.append(tensor_info)
            
            return {
                'metadata': self.metadata,
                'tensors': self.tensor_info,
                'tensor_count': tensor_count,
                'metadata_count': metadata_count
            }
    
    def _read_string(self, f) -> str:
        """Read length-prefixed string"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
    
    def _read_metadata_kv(self, f) -> Tuple[str, any]:
        """Read metadata key-value pair"""
        key = self._read_string(f)
        value_type = struct.unpack('<I', f.read(4))[0]
        
        if value_type == self.GGUF_TYPE_UINT8:
            value = struct.unpack('<B', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_INT8:
            value = struct.unpack('<b', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_UINT16:
            value = struct.unpack('<H', f.read(2))[0]
        elif value_type == self.GGUF_TYPE_INT16:
            value = struct.unpack('<h', f.read(2))[0]
        elif value_type == self.GGUF_TYPE_UINT32:
            value = struct.unpack('<I', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_INT32:
            value = struct.unpack('<i', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_FLOAT32:
            value = struct.unpack('<f', f.read(4))[0]
        elif value_type == self.GGUF_TYPE_BOOL:
            value = struct.unpack('<?', f.read(1))[0]
        elif value_type == self.GGUF_TYPE_STRING:
            value = self._read_string(f)
        elif value_type == self.GGUF_TYPE_ARRAY:
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            value = []
            for _ in range(array_len):
                if array_type == self.GGUF_TYPE_INT32:
                    value.append(struct.unpack('<i', f.read(4))[0])
                elif array_type == self.GGUF_TYPE_FLOAT32:
                    value.append(struct.unpack('<f', f.read(4))[0])
                elif array_type == self.GGUF_TYPE_STRING:
                    value.append(self._read_string(f))
        else:
            raise ValueError(f"Unsupported metadata type: {value_type}")
        
        return key, value
    
    def _read_tensor_info(self, f) -> Dict:
        """Read tensor information"""
        name = self._read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        
        # Read dimensions (reversed in GGML format)
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack('<Q', f.read(8))[0])
        dims = list(reversed(dims))  # Unreverse
        
        tensor_type = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        
        # Calculate size
        n_elements = 1
        for dim in dims:
            n_elements *= dim
        
        # Type sizes
        type_sizes = {
            self.GGML_TYPE_F32: 4,
            self.GGML_TYPE_F16: 2,
            self.GGML_TYPE_I8: 1,
            self.GGML_TYPE_I16: 2
        }
        
        type_size = type_sizes.get(tensor_type, 4)
        data_size = n_elements * type_size
        
        return {
            'name': name,
            'dims': dims,
            'type': tensor_type,
            'offset': offset,
            'size': data_size,
            'n_elements': n_elements
        }
    
    def get_type_name(self, type_id: int) -> str:
        """Get human-readable type name"""
        types = {
            self.GGML_TYPE_F32: "F32",
            self.GGML_TYPE_F16: "F16",
            self.GGML_TYPE_I8: "I8",
            self.GGML_TYPE_I16: "I16"
        }
        return types.get(type_id, f"Unknown({type_id})")
    
    def read_tensor(self, tensor_name: str) -> np.ndarray:
        """
        Read and dequantize tensor data from GGUF file
        
        Args:
            tensor_name: Name of tensor to read
            
        Returns:
            NumPy array with dequantized tensor data (always float32)
        """
        # Find tensor info
        tensor_info = None
        for t in self.tensor_info:
            if t['name'] == tensor_name:
                tensor_info = t
                break
        
        if tensor_info is None:
            raise ValueError(f"Tensor not found: {tensor_name}")
        
        # Open file and seek to tensor data
        with open(self.path, 'rb') as f:
            # Calculate data section start
            # Read header to get to data section
            f.read(4)  # magic
            f.read(4)  # version
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            
            # Skip metadata
            for _ in range(metadata_count):
                self._skip_metadata_kv(f)
            
            # Skip tensor info
            for _ in range(tensor_count):
                self._skip_tensor_info(f)
            
            # Now at start of tensor data (with alignment)
            alignment = 32
            current_pos = f.tell()
            aligned_pos = ((current_pos + alignment - 1) // alignment) * alignment
            f.seek(aligned_pos)
            
            data_start = aligned_pos
            
            # Seek to tensor offset
            f.seek(data_start + tensor_info['offset'])
            
            # Read tensor data based on type
            tensor_type = tensor_info['type']
            n_elements = tensor_info['n_elements']
            dims = tensor_info['dims']
            
            if tensor_type == self.GGML_TYPE_F32:
                # Float32 - just read directly
                data = np.fromfile(f, dtype=np.float32, count=n_elements)
            
            elif tensor_type == self.GGML_TYPE_I16:
                # INT16 - read and dequantize
                quant_data = np.fromfile(f, dtype=np.int16, count=n_elements)
                
                # Read scale (stored in metadata with key: quantization.{tensor_name}.scale)
                scale_key = f"quantization.{tensor_name}.scale"
                scale = self.metadata.get(scale_key, 1.0)
                
                # Dequantize
                data = quant_data.astype(np.float32) * scale
            
            elif tensor_type == self.GGML_TYPE_I8:
                # INT8 - read and dequantize
                quant_data = np.fromfile(f, dtype=np.int8, count=n_elements)
                
                # Check if per-channel (array with .scales) or per-tensor (single value with .scale)
                # Try per-channel first (INT8-CH)
                scale_key_ch = f"quantization.{tensor_name}.scales"
                scale = self.metadata.get(scale_key_ch, None)
                
                if scale is None:
                    # Try per-tensor (INT8)
                    scale_key = f"quantization.{tensor_name}.scale"
                    scale = self.metadata.get(scale_key, 1.0)
                
                if isinstance(scale, list) and len(scale) == 1:
                    # Single scale stored as list (shouldn't happen, but handle it)
                    data = quant_data.astype(np.float32) * scale[0]
                elif isinstance(scale, list):
                    # Per-channel quantization (INT8-CH)
                    scale_array = np.array(scale, dtype=np.float32)
                    
                    # Reshape to original dimensions
                    quant_reshaped = quant_data.reshape(dims)
                    
                    if len(dims) == 2 and len(scale_array) == dims[0]:
                        # 2D tensor (weight matrix): (out_channels, in_channels)
                        # Scale per output channel
                        data = quant_reshaped.astype(np.float32) * scale_array[:, np.newaxis]
                    elif len(dims) == 3 and len(scale_array) == dims[0]:
                        # 3D tensor (conv filter): (out_channels, height, width)
                        # Scale per output channel
                        data = quant_reshaped.astype(np.float32) * scale_array[:, np.newaxis, np.newaxis]
                    elif len(dims) == 1:
                        # 1D tensor (bias) with per-channel scale
                        # Should be per-element scale for bias
                        if len(scale_array) == len(quant_data):
                            data = quant_data.astype(np.float32) * scale_array
                        else:
                            # Fallback: use first scale or mean
                            data = quant_data.astype(np.float32) * scale_array[0]
                    else:
                        # Unknown shape - try broadcasting
                        data = quant_reshaped.astype(np.float32) * scale_array[0]
                    
                    # Flatten back to 1D (reshape will fix it later)
                    data = data.flatten()
                else:
                    # Per-tensor quantization (simple scalar scale)
                    data = quant_data.astype(np.float32) * scale
            
            else:
                raise ValueError(f"Unsupported tensor type: {tensor_type}")
            
            # Reshape to original dimensions
            return data.reshape(dims)
    
    def _skip_metadata_kv(self, f):
        """Skip one metadata key-value pair"""
        # Skip key
        length = struct.unpack('<Q', f.read(8))[0]
        f.read(length)
        
        # Skip value based on type
        value_type = struct.unpack('<I', f.read(4))[0]
        
        if value_type in [self.GGUF_TYPE_UINT8, self.GGUF_TYPE_INT8]:
            f.read(1)
        elif value_type in [self.GGUF_TYPE_UINT16, self.GGUF_TYPE_INT16]:
            f.read(2)
        elif value_type in [self.GGUF_TYPE_UINT32, self.GGUF_TYPE_INT32, self.GGUF_TYPE_FLOAT32]:
            f.read(4)
        elif value_type == self.GGUF_TYPE_BOOL:
            f.read(1)
        elif value_type == self.GGUF_TYPE_STRING:
            length = struct.unpack('<Q', f.read(8))[0]
            f.read(length)
        elif value_type == self.GGUF_TYPE_ARRAY:
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            for _ in range(array_len):
                if array_type in [self.GGUF_TYPE_INT32, self.GGUF_TYPE_FLOAT32]:
                    f.read(4)
                elif array_type == self.GGUF_TYPE_STRING:
                    length = struct.unpack('<Q', f.read(8))[0]
                    f.read(length)
        elif value_type in [self.GGUF_TYPE_UINT64, self.GGUF_TYPE_INT64, self.GGUF_TYPE_FLOAT64]:
            f.read(8)
    
    def _skip_tensor_info(self, f):
        """Skip one tensor info entry"""
        # Skip name
        length = struct.unpack('<Q', f.read(8))[0]
        f.read(length)
        
        # Skip dimensions
        n_dims = struct.unpack('<I', f.read(4))[0]
        f.read(8 * n_dims)
        
        # Skip type and offset
        f.read(4)  # type
        f.read(8)  # offset


def validate_gguf(path: Path) -> Dict:
    """
    Validate GGUF file and return summary
    
    Args:
        path: Path to GGUF file
        
    Returns:
        Validation results dictionary
    """
    loader = GGUFLoader(path)
    
    try:
        data = loader.load()
        
        # Calculate total size
        total_params = sum(t['n_elements'] for t in data['tensors'])
        total_size_mb = path.stat().st_size / (1024 * 1024)
        
        # Get quantization info
        quant = loader.metadata.get('firered.quantization', 'unknown')
        mode = loader.metadata.get('firered.mode', 'unknown')
        
        # Check for required metadata
        required_keys = [
            'general.architecture',
            'general.name',
            'firered.mode',
            'firered.quantization',
            'firered.sample_rate',
            'firered.frame_size_ms',
            'firered.feature_dim'
        ]
        
        missing_keys = [k for k in required_keys if k not in loader.metadata]
        
        return {
            'valid': len(missing_keys) == 0,
            'path': str(path),
            'file_size_mb': total_size_mb,
            'tensor_count': data['tensor_count'],
            'metadata_count': data['metadata_count'],
            'total_parameters': total_params,
            'quantization': quant,
            'mode': mode,
            'missing_keys': missing_keys,
            'metadata': loader.metadata,
            'tensors': data['tensors'],
            'error': None
        }
        
    except Exception as e:
        return {
            'valid': False,
            'path': str(path),
            'error': str(e)
        }


def test_all_models(gguf_dir: Path) -> List[Dict]:
    """Test all GGUF models in directory"""
    results = []
    
    gguf_files = sorted(gguf_dir.glob("*.gguf"))
    
    print(f"\n{'='*70}")
    print(f"GGUF Model Validation")
    print(f"{'='*70}")
    print(f"Directory: {gguf_dir}")
    print(f"Found {len(gguf_files)} GGUF files")
    print(f"{'='*70}\n")
    
    for gguf_file in gguf_files:
        print(f"Testing: {gguf_file.name}")
        result = validate_gguf(gguf_file)
        results.append(result)
        
        if result['valid']:
            print(f"  ✓ Valid")
            print(f"  Size: {result['file_size_mb']:.2f} MB")
            print(f"  Tensors: {result['tensor_count']}")
            print(f"  Parameters: {result['total_parameters']:,}")
            print(f"  Quantization: {result['quantization']}")
            print(f"  Mode: {result['mode']}")
        else:
            print(f"  ✗ Invalid: {result.get('error', 'Unknown error')}")
            if result.get('missing_keys'):
                print(f"  Missing metadata: {', '.join(result['missing_keys'])}")
        print()
    
    # Summary
    print(f"{'='*70}")
    print(f"Validation Summary")
    print(f"{'='*70}")
    
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    
    print(f"Total models: {total_count}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {total_count - valid_count}")
    
    if valid_count == total_count:
        print(f"\n✓ All models are valid!")
    elif valid_count > 0:
        print(f"\n⚠ Some models failed validation")
        for r in results:
            if not r['valid']:
                print(f"  ✗ {Path(r['path']).name}")
    else:
        print(f"\n✗ All models failed validation")
    
    print(f"{'='*70}\n")
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF Model Loader and Validator")
    parser.add_argument('--input', '-i', type=str, help='Single GGUF file to test')
    parser.add_argument('--dir', '-d', type=str, default='../gguf_models', help='Directory with GGUF files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.input:
        # Test single file
        path = Path(args.input)
        print(f"Testing single file: {path}")
        result = validate_gguf(path)
        
        if result['valid']:
            print(f"\n✓ Valid GGUF file")
            print(f"  Size: {result['file_size_mb']:.2f} MB")
            print(f"  Tensors: {result['tensor_count']}")
            print(f"  Parameters: {result['total_parameters']:,}")
            
            if args.verbose:
                print(f"\nMetadata:")
                for key, value in result['metadata'].items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  {key}: [{len(value)} items]")
                    else:
                        print(f"  {key}: {value}")
                
                print(f"\nTensors:")
                loader = GGUFLoader(path)
                for t in result['tensors']:
                    print(f"  {t['name']}: {t['dims']} ({loader.get_type_name(t['type'])})")
        else:
            print(f"\n✗ Invalid: {result['error']}")
            sys.exit(1)
    else:
        # Test all files in directory
        gguf_dir = Path(args.dir).resolve()
        results = test_all_models(gguf_dir)
        
        # Exit with error if any failed
        if not all(r['valid'] for r in results):
            sys.exit(1)


if __name__ == "__main__":
    main()
