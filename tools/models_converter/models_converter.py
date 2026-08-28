#!/usr/bin/env python3
"""
FireRed-VAD Model Converter v3
Complete PyTorch to GGUF converter with all quantizations: FP32, INT16, INT8, INT8-ch

Usage:
    python models_converter_v3.py --input vad --quant fp32
    python models_converter_v3.py --input vad --quant all
    python models_converter_v3.py --all --quant all
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

# Import our modules
from pytorch_loader import PyTorchModelLoader
from gguf_writer import GGUFWriter
from gguf_types import GGMLQuantizationType
from quantizer import ModelQuantizer, QuantizedTensor


# Model and quantization configurations
MODEL_TYPES = {
    "vad": {
        "name_base": "firered-vad",
        "pth_dir": "vad",
        "description": "Standard VAD (Non-Streaming)",
        "mode": "standard"
    },
    "stream-vad": {
        "name_base": "firered-stream-vad",
        "pth_dir": "stream-vad",
        "description": "Streaming VAD",
        "mode": "streaming"
    },
    "aed": {
        "name_base": "firered-aed",  # Consistent with ESP32-P4 naming
        "pth_dir": "aed",
        "description": "Audio Event Detection",
        "mode": "aed"
    }
}

QUANTIZATIONS = {
    "fp32": {
        "ggml_type": GGMLQuantizationType.F32,
        "description": "32-bit floating point (no quantization)"
    },
    "int16": {
        "ggml_type": GGMLQuantizationType.I16,
        "description": "16-bit integer (symmetric quantization)"
    },
    "int8": {
        "ggml_type": GGMLQuantizationType.I8,
        "description": "8-bit integer per-tensor (symmetric)"
    },
    "int8-ch": {
        "ggml_type": GGMLQuantizationType.I8,
        "description": "8-bit integer per-channel (symmetric)"
    }
}


def quantize_weights(weights: Dict[str, np.ndarray], 
                    quant_type: str):
    """
    Quantize model weights
    
    Args:
        weights: Dictionary of FP32 weights
        quant_type: Quantization type (fp32, int16, int8, int8-ch)
        
    Returns:
        Tuple of (quantized_weights_dict, stats_dict)
    """
    quantized = {}
    stats = {}
    
    for name, tensor in weights.items():
        if quant_type == "fp32":
            # No quantization - wrap in QuantizedTensor for consistency
            quantized[name] = QuantizedTensor(
                data=tensor,
                scale=np.array(1.0, dtype=np.float32),
                per_channel=False
            )
            stats[name] = {"quantization": "none"}
            
        elif quant_type == "int16":
            q_tensor = ModelQuantizer.quantize_int16(tensor)
            quantized[name] = q_tensor
            error = ModelQuantizer.calculate_quantization_error(tensor, q_tensor)
            stats[name] = {
                "quantization": "int16",
                "scale": float(q_tensor.scale),
                **error
            }
            
        elif quant_type == "int8":
            q_tensor = ModelQuantizer.quantize_int8(tensor, per_channel=False)
            quantized[name] = q_tensor
            error = ModelQuantizer.calculate_quantization_error(tensor, q_tensor)
            stats[name] = {
                "quantization": "int8-per-tensor",
                "scale": float(q_tensor.scale),
                **error
            }
            
        elif quant_type == "int8-ch":
            q_tensor = ModelQuantizer.quantize_int8(tensor, per_channel=True)
            quantized[name] = q_tensor
            error = ModelQuantizer.calculate_quantization_error(tensor, q_tensor)
            stats[name] = {
                "quantization": "int8-per-channel",
                "scale_min": float(q_tensor.scale.min()),
                "scale_max": float(q_tensor.scale.max()),
                "scale_mean": float(q_tensor.scale.mean()),
                **error
            }
    
    return quantized, stats


def convert_model(model_type: str, quant_type: str, 
                 pth_dir: Path, output_dir: Path,
                 save_debug: bool = False) -> bool:
    """
    Convert single model with specific quantization
    
    Args:
        model_type: Model type (vad, stream-vad, aed)
        quant_type: Quantization (fp32, int16, int8, int8-ch)
        pth_dir: PTH models directory
        output_dir: GGUF output directory
        save_debug: Save debug JSON with quantization stats
        
    Returns:
        True if successful
    """
    config = MODEL_TYPES[model_type]
    model_dir = pth_dir / config["pth_dir"]
    model_name = f"{config['name_base']}-{quant_type}"
    output_path = output_dir / f"{model_name}.gguf"
    
    print(f"\n{'='*70}")
    print(f"Converting: {model_name}")
    print(f"  Type: {config['description']}")
    print(f"  Quantization: {QUANTIZATIONS[quant_type]['description']}")
    print(f"{'='*70}")
    
    try:
        # 1. Load PyTorch model
        print("\n[1/4] Loading PyTorch model...")
        loader = PyTorchModelLoader(
            model_dir / "model.pth.tar",
            model_dir / "cmvn.ark"
        )
        
        weights_fp32 = loader.load()
        mean, variance = loader.load_cmvn()
        
        # 2. Quantize weights
        print(f"\n[2/4] Quantizing to {quant_type.upper()}...")
        quantized_weights, quant_stats = quantize_weights(weights_fp32, quant_type)
        
        # Print quantization summary
        if quant_type != "fp32":
            all_mae = [s.get('mae', 0) for s in quant_stats.values() if 'mae' in s]
            all_sqnr = [s.get('sqnr_db', 0) for s in quant_stats.values() if 'sqnr_db' in s]
            if all_mae and all_sqnr:
                print(f"  Average MAE: {np.mean(all_mae):.6f}")
                print(f"  Average SQNR: {np.mean(all_sqnr):.2f} dB")
        
        # 3. Create GGUF writer
        print("\n[3/4] Creating GGUF file...")
        writer = GGUFWriter(output_path, arch="firered-vad")
        
        # Add metadata
        writer.add_metadata("general.name", model_name)
        writer.add_metadata("firered.mode", config["mode"])
        writer.add_metadata("firered.quantization", quant_type)
        writer.add_metadata("firered.sample_rate", 16000)
        writer.add_metadata("firered.frame_size_ms", 30)
        writer.add_metadata("firered.feature_dim", 80)
        
        # Add CMVN as metadata
        writer.add_metadata("firered.cmvn_mean", mean.tolist())
        writer.add_metadata("firered.cmvn_variance", variance.tolist())
        
        # 4. Add all tensors
        print(f"\n[4/4] Writing {len(quantized_weights)} tensors...")
        for name, q_tensor in quantized_weights.items():
            # Add scale(s) to metadata if quantized
            if quant_type != "fp32":
                if q_tensor.per_channel:
                    # Per-channel: store scales array
                    writer.add_metadata(f"quantization.{name}.scales", q_tensor.scale.tolist())
                else:
                    # Per-tensor: single scale
                    writer.add_metadata(f"quantization.{name}.scale", float(q_tensor.scale))
            
            # Write tensor data
            ggml_type = QUANTIZATIONS[quant_type]["ggml_type"]
            writer.add_tensor(name, q_tensor.data, ggml_type)
        
        # Write file
        writer.write()
        
        # Save debug JSON if requested
        if save_debug:
            debug_path = output_dir / f"{model_name}-debug.json"
            debug_data = {
                "model_name": model_name,
                "model_type": model_type,
                "quantization": quant_type,
                "tensors": quant_stats,
                "summary": {
                    "total_tensors": len(quantized_weights),
                    "total_parameters": sum(t.data.size for t in quantized_weights.values())
                }
            }
            
            if quant_type != "fp32":
                all_mae = [s.get('mae', 0) for s in quant_stats.values() if 'mae' in s]
                all_sqnr = [s.get('sqnr_db', 0) for s in quant_stats.values() if 'sqnr_db' in s]
                debug_data["summary"]["average_mae"] = float(np.mean(all_mae))
                debug_data["summary"]["average_sqnr_db"] = float(np.mean(all_sqnr))
            
            with open(debug_path, 'w') as f:
                json.dump(debug_data, f, indent=2)
            print(f"  Debug JSON: {debug_path.name}")
        
        # Validate
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ SUCCESS: {output_path.name} ({file_size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_all(pth_dir: Path, output_dir: Path, 
                model_types: List[str], quant_types: List[str],
                save_debug: bool = False) -> None:
    """Convert multiple models with multiple quantizations"""
    print("\n" + "="*70)
    print("FireRed-VAD Model Converter v3 - All Quantizations")
    print("="*70)
    print(f"Models: {', '.join(model_types)}")
    print(f"Quantizations: {', '.join(quant_types)}")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    total = len(model_types) * len(quant_types)
    current = 0
    
    for model_type in model_types:
        for quant_type in quant_types:
            current += 1
            print(f"\n[{current}/{total}]")
            
            key = f"{model_type}-{quant_type}"
            success = convert_model(model_type, quant_type, pth_dir, output_dir, save_debug)
            results[key] = success
    
    # Summary
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    
    successful = sum(results.values())
    
    print(f"\nTotal: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    if successful == total:
        print("\n✓ All models converted successfully!")
    elif successful > 0:
        print("\n⚠ Some models failed:")
        for key, success in results.items():
            if not success:
                print(f"  ✗ {key}")
    else:
        print("\n✗ All conversions failed")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FireRed-VAD Model Converter v3 - Full Quantization Support",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        choices=list(MODEL_TYPES.keys()),
        help='Convert specific model type'
    )
    
    parser.add_argument(
        '--quant', '-q',
        type=str,
        choices=list(QUANTIZATIONS.keys()) + ['all'],
        default='fp32',
        help='Quantization type (default: fp32)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Convert all model types'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../pth_models',
        help='PTH models directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../gguf_models',
        help='GGUF output directory'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save debug JSON files with quantization statistics'
    )
    
    args = parser.parse_args()
    
    if not args.input and not args.all:
        print("ERROR: Specify --input <model> or --all")
        parser.print_help()
        sys.exit(1)
    
    pth_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Determine which models to convert
    if args.all:
        model_types = list(MODEL_TYPES.keys())
    else:
        model_types = [args.input]
    
    # Determine which quantizations to use
    if args.quant == 'all':
        quant_types = list(QUANTIZATIONS.keys())
    else:
        quant_types = [args.quant]
    
    # Convert
    convert_all(pth_dir, output_dir, model_types, quant_types, args.debug)


if __name__ == "__main__":
    main()
