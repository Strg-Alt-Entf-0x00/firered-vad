#!/usr/bin/env python3
"""
GGUF v3 Writer for FireRed-VAD

Writes proper GGUF format files with metadata and tensors.
"""

import struct
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np

from gguf_types import (
    GGUF_MAGIC, GGUF_VERSION, GGUF_DEFAULT_ALIGNMENT,
    GGUFValueType, GGMLQuantizationType, GGML_TYPE_SIZE
)


class GGUFWriter:
    """Write GGUF v3 format files"""
    
    def __init__(self, output_path: Path, arch: str = "firered-vad"):
        """
        Initialize GGUF writer
        
        Args:
            output_path: Path to output .gguf file
            arch: Architecture name (default: firered-vad)
        """
        self.output_path = Path(output_path)
        self.arch = arch
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict] = []
        
        # Always set architecture
        self.add_metadata("general.architecture", arch)
    
    def add_metadata(self, key: str, value: Any):
        """
        Add metadata key-value pair
        
        Args:
            key: Metadata key (string)
            value: Value (int, float, string, bool, or array)
        """
        self.metadata[key] = value
    
    def add_tensor(self, name: str, data: np.ndarray, 
                   quant_type: GGMLQuantizationType = GGMLQuantizationType.F32):
        """
        Add tensor to GGUF file
        
        Args:
            name: Tensor name
            data: NumPy array
            quant_type: GGML quantization type
        """
        # Ensure contiguous memory layout
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Convert to appropriate dtype
        if quant_type == GGMLQuantizationType.F32:
            data = data.astype(np.float32)
        elif quant_type == GGMLQuantizationType.F16:
            data = data.astype(np.float16)
        elif quant_type == GGMLQuantizationType.I8:
            data = data.astype(np.int8)
        elif quant_type == GGMLQuantizationType.I16:
            data = data.astype(np.int16)
        elif quant_type == GGMLQuantizationType.I32:
            data = data.astype(np.int32)
        
        self.tensors.append({
            'name': name,
            'data': data,
            'shape': data.shape,
            'type': quant_type
        })
    
    def write(self):
        """Write complete GGUF file"""
        print(f"  Writing GGUF to: {self.output_path}")
        
        with open(self.output_path, 'wb') as f:
            # 1. Header
            self._write_header(f)
            
            # 2. Metadata section
            self._write_metadata(f)
            
            # 3. Tensor info section (with placeholder offsets)
            tensor_info_positions = self._write_tensor_info(f)
            
            # 4. Alignment padding to data section
            current_pos = f.tell()
            padding = (GGUF_DEFAULT_ALIGNMENT - (current_pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
            f.write(b'\x00' * padding)
            
            # Remember where tensor data starts
            data_start = f.tell()
            
            # 5. Write tensor data and collect actual offsets
            actual_offsets = self._write_tensor_data(f, data_start)
            
            # 6. Go back and update tensor offsets
            self._update_tensor_offsets(f, tensor_info_positions, actual_offsets)
        
        file_size = self.output_path.stat().st_size
        print(f"  Written {len(self.tensors)} tensors ({file_size:,} bytes)")
    
    def _write_header(self, f):
        """Write GGUF header"""
        # Magic number: "GGUF" (4 bytes)
        f.write(struct.pack('<I', GGUF_MAGIC))
        
        # Version: 3 (4 bytes)
        f.write(struct.pack('<I', GGUF_VERSION))
        
        # Tensor count (8 bytes)
        f.write(struct.pack('<Q', len(self.tensors)))
        
        # Metadata count (8 bytes)
        f.write(struct.pack('<Q', len(self.metadata)))
    
    def _write_metadata(self, f):
        """Write metadata key-value pairs"""
        for key, value in self.metadata.items():
            # Write key
            self._write_string(f, key)
            
            # Write value with type
            self._write_value(f, value)
    
    def _write_string(self, f, s: str):
        """Write string with length prefix"""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f, value: Any):
        """Write typed value"""
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUFValueType.BOOL))
            f.write(struct.pack('<?', value))
        elif isinstance(value, int):
            if value < 0:
                f.write(struct.pack('<I', GGUFValueType.INT32))
                f.write(struct.pack('<i', value))
            else:
                f.write(struct.pack('<I', GGUFValueType.UINT32))
                f.write(struct.pack('<I', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUFValueType.FLOAT32))
            f.write(struct.pack('<f', value))
        elif isinstance(value, str):
            f.write(struct.pack('<I', GGUFValueType.STRING))
            self._write_string(f, value)
        elif isinstance(value, (list, tuple)):
            f.write(struct.pack('<I', GGUFValueType.ARRAY))
            # Array type and count
            if len(value) > 0:
                # Detect array element type
                elem_type = type(value[0])
                if elem_type == int:
                    f.write(struct.pack('<I', GGUFValueType.INT32))
                elif elem_type == float:
                    f.write(struct.pack('<I', GGUFValueType.FLOAT32))
                elif elem_type == str:
                    f.write(struct.pack('<I', GGUFValueType.STRING))
                else:
                    raise ValueError(f"Unsupported array element type: {elem_type}")
            else:
                f.write(struct.pack('<I', GGUFValueType.INT32))  # Default
            
            # Array length
            f.write(struct.pack('<Q', len(value)))
            
            # Array elements
            for elem in value:
                if isinstance(elem, int):
                    f.write(struct.pack('<i', elem))
                elif isinstance(elem, float):
                    f.write(struct.pack('<f', elem))
                elif isinstance(elem, str):
                    self._write_string(f, elem)
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")
    
    def _write_tensor_info(self, f) -> List[int]:
        """
        Write tensor information section
        
        Returns:
            List of file positions where each tensor's offset is written
        """
        offset_positions = []
        
        for tensor in self.tensors:
            # Tensor name
            self._write_string(f, tensor['name'])
            
            # Number of dimensions
            n_dims = len(tensor['shape'])
            f.write(struct.pack('<I', n_dims))
            
            # Dimensions (reversed for GGML format)
            for dim in reversed(tensor['shape']):
                f.write(struct.pack('<Q', dim))
            
            # Tensor type
            f.write(struct.pack('<I', tensor['type']))
            
            # Offset (placeholder, will update later)
            offset_positions.append(f.tell())
            f.write(struct.pack('<Q', 0))
        
        return offset_positions
    
    def _write_tensor_data(self, f, data_start: int) -> List[int]:
        """
        Write tensor data section
        
        Args:
            data_start: File position where tensor data section starts
            
        Returns:
            List of actual offsets for each tensor (relative to data_start)
        """
        actual_offsets = []
        
        for i, tensor in enumerate(self.tensors):
            # Calculate offset relative to data_start
            offset = f.tell() - data_start
            actual_offsets.append(offset)
            
            # Write tensor data
            data = tensor['data']
            f.write(data.tobytes())
            
            # Align next tensor to 32 bytes (except for last tensor)
            if i < len(self.tensors) - 1:
                current_pos = f.tell()
                padding = (GGUF_DEFAULT_ALIGNMENT - (current_pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
                if padding > 0:
                    f.write(b'\x00' * padding)
        
        return actual_offsets
    
    def _update_tensor_offsets(self, f, offset_positions: List[int], actual_offsets: List[int]):
        """
        Go back and update tensor offsets in tensor info section
        
        Args:
            f: File handle
            offset_positions: List of file positions where offsets are written
            actual_offsets: List of actual offset values
        """
        for pos, offset in zip(offset_positions, actual_offsets):
            f.seek(pos)
            f.write(struct.pack('<Q', offset))
    

def test_writer():
    """Test GGUF writer"""
    import sys
    
    print("Testing GGUFWriter")
    
    try:
        writer = GGUFWriter(Path("test_output.gguf"), arch="test-model")
        
        # Add metadata
        writer.add_metadata("test.int", 42)
        writer.add_metadata("test.float", 3.14)
        writer.add_metadata("test.string", "Hello GGUF")
        writer.add_metadata("test.bool", True)
        writer.add_metadata("test.array", [1, 2, 3, 4, 5])
        
        # Add tensors
        tensor1 = np.random.randn(10, 20).astype(np.float32)
        tensor2 = np.random.randn(5).astype(np.float32)
        
        writer.add_tensor("test.weight", tensor1)
        writer.add_tensor("test.bias", tensor2)
        
        # Write
        writer.write()
        
        print("✓ GGUF file written successfully")
        print(f"  File size: {Path('test_output.gguf').stat().st_size} bytes")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_writer()
