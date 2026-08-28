#!/usr/bin/env python3
"""
Compare Quantization Quality
Loads debug JSON files and generates comparison reports
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import statistics


def load_debug_json(path: Path) -> Dict:
    """Load debug JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def analyze_quantization_quality(debug_data: Dict) -> Dict:
    """Analyze quantization quality from debug JSON"""
    tensors = debug_data.get('tensors', {})
    summary = debug_data.get('summary', {})
    
    # Collect all metrics
    all_mae = []
    all_sqnr = []
    all_max_error = []
    
    for tensor_name, stats in tensors.items():
        if 'mae' in stats:
            all_mae.append(stats['mae'])
        if 'sqnr_db' in stats:
            sqnr = stats['sqnr_db']
            if sqnr != float('inf'):
                all_sqnr.append(sqnr)
        if 'max_error' in stats:
            all_max_error.append(stats['max_error'])
    
    result = {
        'model': debug_data.get('model_name', 'unknown'),
        'quantization': debug_data.get('quantization', 'unknown'),
        'tensor_count': len(tensors),
        'total_parameters': summary.get('total_parameters', 0)
    }
    
    if all_mae:
        result['mae_mean'] = statistics.mean(all_mae)
        result['mae_median'] = statistics.median(all_mae)
        result['mae_min'] = min(all_mae)
        result['mae_max'] = max(all_mae)
    
    if all_sqnr:
        result['sqnr_mean'] = statistics.mean(all_sqnr)
        result['sqnr_median'] = statistics.median(all_sqnr)
        result['sqnr_min'] = min(all_sqnr)
        result['sqnr_max'] = max(all_sqnr)
    
    if all_max_error:
        result['max_error_mean'] = statistics.mean(all_max_error)
        result['max_error_worst'] = max(all_max_error)
    
    return result


def compare_model_type(gguf_dir: Path, model_type: str) -> Dict:
    """Compare all quantizations for a specific model type"""
    pattern = f"firered-{model_type}-*-debug.json"
    debug_files = sorted(gguf_dir.glob(pattern))
    
    results = {}
    
    for debug_file in debug_files:
        data = load_debug_json(debug_file)
        quant_type = data.get('quantization', 'unknown')
        analysis = analyze_quantization_quality(data)
        results[quant_type] = analysis
    
    return results


def print_comparison_table(model_type: str, results: Dict):
    """Print comparison table for a model type"""
    print(f"\n{'='*80}")
    print(f"Model: {model_type.upper()}")
    print(f"{'='*80}")
    
    # Get quantization types in order
    quant_order = ['fp32', 'int16', 'int8', 'int8-ch']
    quants = [q for q in quant_order if q in results]
    
    if not quants:
        print("No quantization data found")
        return
    
    # Print header
    print(f"\n{'Quantization':<12} {'Parameters':<12} {'MAE (Mean)':<15} {'SQNR (Mean)':<15} {'Max Error':<15}")
    print(f"{'-'*12} {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
    
    for quant in quants:
        data = results[quant]
        params = f"{data['total_parameters']:,}"
        
        if 'mae_mean' in data:
            mae = f"{data['mae_mean']:.6f}"
        else:
            mae = "N/A"
        
        if 'sqnr_mean' in data:
            sqnr = f"{data['sqnr_mean']:.2f} dB"
        else:
            sqnr = "N/A"
        
        if 'max_error_worst' in data:
            max_err = f"{data['max_error_worst']:.6f}"
        else:
            max_err = "N/A"
        
        print(f"{quant:<12} {params:<12} {mae:<15} {sqnr:<15} {max_err:<15}")
    
    # Print detailed statistics for quantized versions
    print(f"\n{'Detailed Statistics (Quantized Only)':^80}")
    print(f"{'-'*80}")
    
    for quant in quants:
        if quant == 'fp32':
            continue
        
        data = results[quant]
        print(f"\n{quant.upper()}:")
        
        if 'mae_mean' in data:
            print(f"  MAE:")
            print(f"    Mean:   {data['mae_mean']:.6f}")
            print(f"    Median: {data['mae_median']:.6f}")
            print(f"    Min:    {data['mae_min']:.6f}")
            print(f"    Max:    {data['mae_max']:.6f}")
        
        if 'sqnr_mean' in data:
            print(f"  SQNR:")
            print(f"    Mean:   {data['sqnr_mean']:.2f} dB")
            print(f"    Median: {data['sqnr_median']:.2f} dB")
            print(f"    Min:    {data['sqnr_min']:.2f} dB")
            print(f"    Max:    {data['sqnr_max']:.2f} dB")


def compare_all_models(gguf_dir: Path):
    """Compare all model types"""
    model_types = ['vad', 'stream-vad', 'aed-vad']
    
    print(f"\n{'='*80}")
    print(f"Quantization Quality Comparison")
    print(f"{'='*80}")
    print(f"Directory: {gguf_dir}")
    print(f"{'='*80}")
    
    for model_type in model_types:
        results = compare_model_type(gguf_dir, model_type)
        if results:
            print_comparison_table(model_type, results)
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"Summary & Recommendations")
    print(f"{'='*80}")
    
    print("""
Quantization Trade-offs:

INT16 (16-bit):
  ✓ Nearly lossless quality (SQNR >80 dB)
  ✓ 50% size reduction
  ✓ Best for desktop/server applications
  ✓ Recommended when quality is priority

INT8 Per-Tensor:
  ✓ 75% size reduction
  ✓ Acceptable quality (SQNR ~40 dB)
  ✓ Simple implementation
  ○ Moderate accuracy loss

INT8 Per-Channel:
  ✓ 75% size reduction (same as per-tensor)
  ✓ Better quality than per-tensor (SQNR ~42 dB)
  ✓ 2-3 dB improvement in SQNR
  ✓ Best INT8 option
  ✓ Recommended for embedded systems

Recommendation:
  - Desktop/Server: Use INT16 or FP32
  - Embedded/Mobile: Use INT8-CH
  - Ultra-constrained: Use INT8 per-tensor
""")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Quantization Quality")
    parser.add_argument('--dir', '-d', type=str, default='../gguf_models',
                       help='Directory with debug JSON files')
    parser.add_argument('--model', '-m', type=str,
                       choices=['vad', 'stream-vad', 'aed-vad'],
                       help='Compare specific model type only')
    
    args = parser.parse_args()
    
    gguf_dir = Path(args.dir).resolve()
    
    if args.model:
        results = compare_model_type(gguf_dir, args.model)
        print_comparison_table(args.model, results)
    else:
        compare_all_models(gguf_dir)


if __name__ == "__main__":
    main()
