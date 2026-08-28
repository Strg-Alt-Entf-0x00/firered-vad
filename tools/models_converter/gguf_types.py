#!/usr/bin/env python3
"""
GGUF Format Types and Constants

Based on GGUF v3 specification.
"""

from enum import IntEnum, Enum


# GGUF Magic Number
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# Alignment for tensor data
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    """GGUF metadata value types"""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantizationType(IntEnum):
    """GGML Quantization Types"""
    F32 = 0    # float32
    F16 = 1    # float16
    Q4_0 = 2   # 4-bit quantization
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8   # 8-bit quantization
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24    # int8
    I16 = 25   # int16
    I32 = 26   # int32
    I64 = 27   # int64
    F64 = 28   # float64
    IQ1_M = 29


# Type size mapping (bytes per element)
GGML_TYPE_SIZE = {
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F16: 2,
    GGMLQuantizationType.I8: 1,
    GGMLQuantizationType.I16: 2,
    GGMLQuantizationType.I32: 4,
    GGMLQuantizationType.I64: 8,
    GGMLQuantizationType.F64: 8,
}


# Standard metadata keys for FireRed-VAD
FIRERED_METADATA_KEYS = {
    # General
    'general.architecture': 'firered-vad',
    'general.file_type': 'GGUF v3',
    
    # Model specific
    'firered.mode': None,  # 'standard', 'streaming', or 'aed'
    'firered.sample_rate': 16000,
    'firered.frame_size_ms': 30,
    'firered.feature_dim': 80,
    'firered.quantization': None,  # 'fp32', 'int16', 'int8', 'int8-ch'
    
    # Architecture
    'firered.dfsmn_blocks': 8,
    'firered.memory_order': 20,
}


def get_quantization_name(quant_type: GGMLQuantizationType) -> str:
    """Get human-readable quantization name"""
    mapping = {
        GGMLQuantizationType.F32: 'fp32',
        GGMLQuantizationType.F16: 'fp16',
        GGMLQuantizationType.I8: 'int8',
        GGMLQuantizationType.I16: 'int16',
        GGMLQuantizationType.I32: 'int32',
    }
    return mapping.get(quant_type, f'quant_{quant_type}')


def get_firered_mode_from_filename(filename: str) -> str:
    """Extract FireRed mode from filename"""
    filename = filename.lower()
    if 'stream' in filename:
        return 'streaming'
    elif 'aed' in filename:
        return 'aed'
    else:
        return 'standard'
