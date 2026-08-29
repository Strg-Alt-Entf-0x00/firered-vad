#!/usr/bin/env python3
"""
Weight-Level Golden Test: Compare GGUF weights against PyTorch weights
This validates the conversion process without requiring inference implementation
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models_converter.pytorch_loader import PyTorchModelLoader
from test_gguf_loader import GGUFLoader


class WeightFidelityTester:
    """Compare GGUF weights against original PyTorch weights"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
    
    def load_pytorch_weights(self, pth_path: Path, cmvn_path: Path) -> Dict[str, np.ndarray]:
        """Load original PyTorch model weights"""
        if self.verbose:
            print(f"Loading PyTorch model: {pth_path}")
        
        loader = PyTorchModelLoader(pth_path, cmvn_path)
        weights = loader.load()
        
        if self.verbose:
            print(f"  Loaded {len(weights)} tensors")
            total_params = sum(w.size for w in weights.values())
            print(f"  Total parameters: {total_params:,}")
        
        return weights
    
    def load_gguf_weights(self, gguf_path: Path) -> Tuple[Dict[str, np.ndarray], str]:
        """Load GGUF model weights and quantization type"""
        if self.verbose:
            print(f"Loading GGUF model: {gguf_path}")
        
        loader = GGUFLoader(gguf_path)
        loader.load()  # Load metadata and tensor info
        
        # Get quantization type from metadata
        # Try the standard firered.quantization key first
        quant_type = loader.metadata.get('firered.quantization', None)
        
        if quant_type is None:
            # Fallback: infer from filename
            if 'fp32' in gguf_path.name:
                quant_type = 'fp32'
            elif 'int16' in gguf_path.name:
                quant_type = 'int16'
            elif 'int8-ch' in gguf_path.name:
                quant_type = 'int8-ch'
            elif 'int8' in gguf_path.name:
                quant_type = 'int8'
        
        # Load and dequantize tensors
        weights = {}
        for tensor_info in loader.tensor_info:
            name = tensor_info['name']
            try:
                weights[name] = loader.read_tensor(name)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to read tensor {name}: {e}")
        
        if self.verbose:
            print(f"  Loaded {len(weights)} tensors ({quant_type})")
            total_params = sum(w.size for w in weights.values())
            print(f"  Total parameters: {total_params:,}")
        
        return weights, quant_type
    
    def compare_weights(
        self,
        pytorch_weights: Dict[str, np.ndarray],
        gguf_weights: Dict[str, np.ndarray],
        quant_type: str
    ) -> Dict:
        """
        Compare PyTorch weights vs GGUF weights
        
        Returns:
            Dict with metrics: mae, max_error, mse, matching_tensors, etc.
        """
        metrics = {
            'quant_type': quant_type,
            'pytorch_tensors': len(pytorch_weights),
            'gguf_tensors': len(gguf_weights),
            'matching_tensors': 0,
            'mismatched_shapes': [],
            'missing_in_gguf': [],
            'missing_in_pytorch': [],
            'per_tensor_metrics': {},
            'overall_mae': 0.0,
            'overall_max_error': 0.0,
            'overall_mse': 0.0,
        }
        
        # Check tensor names match
        pytorch_names = set(pytorch_weights.keys())
        gguf_names = set(gguf_weights.keys())
        
        metrics['missing_in_gguf'] = list(pytorch_names - gguf_names)
        metrics['missing_in_pytorch'] = list(gguf_names - pytorch_names)
        common_names = pytorch_names & gguf_names
        
        if self.verbose:
            print(f"\nComparing weights:")
            print(f"  PyTorch tensors: {len(pytorch_names)}")
            print(f"  GGUF tensors: {len(gguf_names)}")
            print(f"  Common tensors: {len(common_names)}")
            if metrics['missing_in_gguf']:
                print(f"  [!] Missing in GGUF: {metrics['missing_in_gguf']}")
            if metrics['missing_in_pytorch']:
                print(f"  [!] Missing in PyTorch: {metrics['missing_in_pytorch']}")
        
        # Compare common tensors
        all_maes = []
        all_max_errors = []
        all_mses = []
        
        for name in sorted(common_names):
            pt_weight = pytorch_weights[name]
            gg_weight = gguf_weights[name]
            
            # Check shapes match
            if pt_weight.shape != gg_weight.shape:
                metrics['mismatched_shapes'].append({
                    'name': name,
                    'pytorch_shape': pt_weight.shape,
                    'gguf_shape': gg_weight.shape
                })
                continue
            
            # Calculate metrics
            mae = np.mean(np.abs(pt_weight - gg_weight))
            max_error = np.max(np.abs(pt_weight - gg_weight))
            mse = np.mean((pt_weight - gg_weight) ** 2)
            
            metrics['per_tensor_metrics'][name] = {
                'mae': float(mae),
                'max_error': float(max_error),
                'mse': float(mse),
                'shape': pt_weight.shape,
                'size': pt_weight.size
            }
            
            all_maes.append(mae)
            all_max_errors.append(max_error)
            all_mses.append(mse)
            
            metrics['matching_tensors'] += 1
        
        # Overall metrics
        if all_maes:
            metrics['overall_mae'] = float(np.mean(all_maes))
            metrics['overall_max_error'] = float(np.max(all_max_errors))
            metrics['overall_mse'] = float(np.mean(all_mses))
            
            # Calculate SQNR
            signal_power = sum(
                np.sum(pytorch_weights[name] ** 2)
                for name in common_names
                if name not in [m['name'] for m in metrics['mismatched_shapes']]
            )
            noise_power = sum(
                np.sum((pytorch_weights[name] - gguf_weights[name]) ** 2)
                for name in common_names
                if name not in [m['name'] for m in metrics['mismatched_shapes']]
            )
            
            if noise_power > 0:
                sqnr_db = 10 * np.log10(signal_power / noise_power)
                metrics['overall_sqnr_db'] = float(sqnr_db)
            else:
                metrics['overall_sqnr_db'] = float('inf')
        
        return metrics
    
    def test_model(
        self,
        model_type: str,
        quant_type: str,
        pth_dir: Path,
        gguf_dir: Path
    ) -> Dict:
        """
        Test one model variant
        
        Args:
            model_type: vad, stream-vad, or aed
            quant_type: fp32, int16, int8, or int8-ch
            pth_dir: Directory containing PTH models
            gguf_dir: Directory containing GGUF models
            
        Returns:
            Test results dict
        """
        # Map model types to directory names
        model_map = {
            'vad': 'vad',
            'stream-vad': 'stream-vad',
            'aed': 'aed'
        }
        
        # Map model types to GGUF filename patterns (consistent with ESP32-P4)
        gguf_name_map = {
            'vad': 'firered-vad',
            'stream-vad': 'firered-stream-vad',
            'aed': 'firered-aed'  # No extra "-vad" suffix
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Paths
        pth_model_dir = pth_dir / model_map[model_type]
        pth_path = pth_model_dir / "model.pth.tar"
        cmvn_path = pth_model_dir / "cmvn.ark"
        
        gguf_basename = gguf_name_map[model_type]
        gguf_filename = f"{gguf_basename}-{quant_type}.gguf"
        gguf_path = gguf_dir / gguf_filename
        
        # Check files exist
        if not pth_path.exists():
            return {
                'error': f"PyTorch model not found: {pth_path}",
                'model_type': model_type,
                'quant_type': quant_type
            }
        
        if not gguf_path.exists():
            return {
                'error': f"GGUF model not found: {gguf_path}",
                'model_type': model_type,
                'quant_type': quant_type
            }
        
        print(f"\n{'='*70}")
        print(f"Testing: {model_type} ({quant_type})")
        print(f"{'='*70}")
        
        # Load weights
        try:
            pytorch_weights = self.load_pytorch_weights(pth_path, cmvn_path)
            gguf_weights, detected_quant = self.load_gguf_weights(gguf_path)
            
            # Verify quantization type
            if detected_quant and detected_quant != quant_type:
                print(f"[!] Warning: Expected {quant_type}, detected {detected_quant}")
            
            # Compare
            metrics = self.compare_weights(pytorch_weights, gguf_weights, quant_type)
            metrics['model_type'] = model_type
            metrics['pth_path'] = str(pth_path)
            metrics['gguf_path'] = str(gguf_path)
            
            # Print summary
            self.print_results(metrics)
            
            return metrics
            
        except Exception as e:
            import traceback
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'model_type': model_type,
                'quant_type': quant_type
            }
    
    def print_results(self, metrics: Dict):
        """Print test results"""
        print(f"\nResults:")
        print(f"  Matching tensors: {metrics['matching_tensors']}/{metrics['pytorch_tensors']}")
        
        if metrics.get('mismatched_shapes'):
            print(f"  [!] Mismatched shapes: {len(metrics['mismatched_shapes'])}")
        
        if metrics['matching_tensors'] > 0:
            print(f"\n  Overall Quality Metrics:")
            print(f"    MAE:       {metrics['overall_mae']:.6e}")
            print(f"    Max Error: {metrics['overall_max_error']:.6e}")
            print(f"    MSE:       {metrics['overall_mse']:.6e}")
            if 'overall_sqnr_db' in metrics:
                sqnr = metrics['overall_sqnr_db']
                if np.isinf(sqnr):
                    print(f"    SQNR:      inf dB (perfect)")
                else:
                    print(f"    SQNR:      {sqnr:.2f} dB")
            
            # Pass/fail thresholds
            quant = metrics['quant_type']
            thresholds = {
                'fp32': {'mae': 1e-5, 'sqnr': 80},
                'int16': {'mae': 1e-3, 'sqnr': 70},
                'int8-ch': {'mae': 5e-3, 'sqnr': 40},
                'int8': {'mae': 1e-2, 'sqnr': 30},
            }
            
            if quant in thresholds:
                thresh = thresholds[quant]
                mae_pass = metrics['overall_mae'] < thresh['mae']
                sqnr_pass = metrics.get('overall_sqnr_db', 0) > thresh['sqnr']
                
                print(f"\n  Thresholds for {quant}:")
                print(f"    MAE < {thresh['mae']:.0e}: {'[OK] PASS' if mae_pass else '[X] FAIL'}")
                print(f"    SQNR > {thresh['sqnr']} dB: {'[OK] PASS' if sqnr_pass else '[X] FAIL'}")
                
                if mae_pass and sqnr_pass:
                    print(f"\n  [OK] Overall: PASS")
                else:
                    print(f"\n  [X] Overall: FAIL")
    
    def test_all(self, pth_dir: Path, gguf_dir: Path) -> list:
        """Test all model variants"""
        model_types = ['vad', 'stream-vad', 'aed']
        quant_types = ['fp32', 'int16', 'int8', 'int8-ch']
        
        results = []
        for model_type in model_types:
            for quant_type in quant_types:
                result = self.test_model(model_type, quant_type, pth_dir, gguf_dir)
                results.append(result)
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Weight-Level Golden Test')
    parser.add_argument('--model', choices=['vad', 'stream-vad', 'aed', 'all'], default='all',
                        help='Model type to test')
    parser.add_argument('--quant', choices=['fp32', 'int16', 'int8', 'int8-ch', 'all'], default='all',
                        help='Quantization type to test')
    parser.add_argument('--pth-dir', type=Path, default=None,
                        help='Directory containing PTH models (default: ../../pth_models)')
    parser.add_argument('--gguf-dir', type=Path, default=None,
                        help='Directory containing GGUF models (default: ../../models_gguf)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Default directories
    if args.pth_dir is None:
        args.pth_dir = Path(__file__).parent.parent.parent / "models_pth"
    if args.gguf_dir is None:
        args.gguf_dir = Path(__file__).parent.parent.parent / "models_gguf"
    
    # Verify directories exist
    if not args.pth_dir.exists():
        print(f"Error: PTH directory not found: {args.pth_dir}")
        sys.exit(1)
    if not args.gguf_dir.exists():
        print(f"Error: GGUF directory not found: {args.gguf_dir}")
        sys.exit(1)
    
    print(f"FireRed-VAD Weight Fidelity Test")
    print(f"{'='*70}")
    print(f"PyTorch models: {args.pth_dir}")
    print(f"GGUF models:    {args.gguf_dir}")
    
    tester = WeightFidelityTester(verbose=args.verbose)
    
    # Determine what to test
    if args.model == 'all' and args.quant == 'all':
        results = tester.test_all(args.pth_dir, args.gguf_dir)
    else:
        models = ['vad', 'stream-vad', 'aed'] if args.model == 'all' else [args.model]
        quants = ['fp32', 'int16', 'int8', 'int8-ch'] if args.quant == 'all' else [args.quant]
        
        results = []
        for model in models:
            for quant in quants:
                result = tester.test_model(model, quant, args.pth_dir, args.gguf_dir)
                results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    passed = sum(1 for r in results if 'error' not in r and r.get('matching_tensors', 0) > 0)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Errors: {errors}")
    
    if errors > 0:
        print(f"\nTests with errors:")
        for r in results:
            if 'error' in r:
                print(f"  - {r['model_type']} ({r['quant_type']}): {r['error']}")
    
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()

