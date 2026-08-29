#!/usr/bin/env python3
"""
Inference-Level Golden Test: Compare GGUF inference outputs vs PyTorch (via OmniVAD)

This tests the COMPLETE pipeline:
- PyTorch models inference (via OmniVAD - uses original .pth.tar files)
- GGUF models inference (TODO: needs C++ integration or Python GGUF runner)

IMPORTANT: This validates that GGUF outputs match PyTorch outputs frame-by-frame.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

try:
    import omnivad
except ImportError:
    print("Error: omnivad not installed")
    print("Install with: pip install omnivad")
    sys.exit(1)


class InferenceGoldenTester:
    """Compare GGUF inference outputs against PyTorch inference (golden reference)"""
    
    def __init__(self, pth_models_dir: Path, verbose: bool = False):
        """
        Initialize tester
        
        Args:
            pth_models_dir: Directory containing original PTH models
            verbose: Verbose output
        """
        self.pth_models_dir = pth_models_dir
        self.verbose = verbose
        
        # Initialize OmniVAD models (uses PyTorch .pth.tar files)
        print("Loading PyTorch models via OmniVAD...")
        
        # OmniVAD will automatically download or use local models
        # We'll point it to our pth_models directory
        self.vad_pytorch = None
        self.stream_vad_pytorch = None
        self.aed_pytorch = None
    
    def load_pytorch_models(self):
        """Load PyTorch models using OmniVAD"""
        try:
            # Standard VAD
            print("  Loading Standard VAD...")
            self.vad_pytorch = omnivad.OmniVAD(
                model_dir=str(self.pth_models_dir / "vad"),
                mode="vad"
            )
            
            # Streaming VAD
            print("  Loading Streaming VAD...")
            self.stream_vad_pytorch = omnivad.OmniVAD(
                model_dir=str(self.pth_models_dir / "stream-vad"),
                mode="stream_vad"
            )
            
            # AED
            print("  Loading AED...")
            self.aed_pytorch = omnivad.OmniVAD(
                model_dir=str(self.pth_models_dir / "aed"),
                mode="aed"
            )
            
            print("✓ All PyTorch models loaded")
            
        except Exception as e:
            print(f"✗ Failed to load PyTorch models: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def run_pytorch_inference(
        self,
        audio_path: Path,
        model_type: str
    ) -> np.ndarray:
        """
        Run PyTorch inference on audio file
        
        Args:
            audio_path: Path to audio file
            model_type: vad, stream-vad, or aed
            
        Returns:
            Frame-level probabilities array
        """
        if model_type == "vad":
            model = self.vad_pytorch
        elif model_type == "stream-vad":
            model = self.stream_vad_pytorch
        elif model_type == "aed":
            model = self.aed_pytorch
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Run inference
        result = model.detect(str(audio_path))
        
        # Extract frame-level probabilities
        # OmniVAD returns dict with timestamps, we need raw frame probs
        # This depends on OmniVAD API - might need adjustment
        
        return result
    
    def compare_inference(
        self,
        pytorch_output: np.ndarray,
        gguf_output: np.ndarray,
        model_type: str,
        quant_type: str
    ) -> Dict:
        """
        Compare PyTorch vs GGUF inference outputs
        
        Args:
            pytorch_output: PyTorch inference result (golden)
            gguf_output: GGUF inference result
            model_type: vad, stream-vad, or aed
            quant_type: fp32, int16, int8, int8-ch
            
        Returns:
            Metrics dict
        """
        # Ensure same shape
        if pytorch_output.shape != gguf_output.shape:
            return {
                'error': f"Shape mismatch: PyTorch {pytorch_output.shape} vs GGUF {gguf_output.shape}",
                'model_type': model_type,
                'quant_type': quant_type
            }
        
        # Calculate metrics
        mae = np.mean(np.abs(pytorch_output - gguf_output))
        max_error = np.max(np.abs(pytorch_output - gguf_output))
        mse = np.mean((pytorch_output - gguf_output) ** 2)
        
        # Correlation
        if pytorch_output.size > 1:
            correlation = np.corrcoef(pytorch_output.flatten(), gguf_output.flatten())[0, 1]
        else:
            correlation = 1.0
        
        # SQNR
        signal_power = np.sum(pytorch_output ** 2)
        noise_power = np.sum((pytorch_output - gguf_output) ** 2)
        
        if noise_power > 0:
            sqnr_db = 10 * np.log10(signal_power / noise_power)
        else:
            sqnr_db = float('inf')
        
        metrics = {
            'model_type': model_type,
            'quant_type': quant_type,
            'mae': float(mae),
            'max_error': float(max_error),
            'mse': float(mse),
            'correlation': float(correlation),
            'sqnr_db': float(sqnr_db),
            'num_frames': pytorch_output.shape[0]
        }
        
        return metrics
    
    def test_audio_file(
        self,
        audio_path: Path,
        model_type: str,
        quant_type: str,
        gguf_dir: Path
    ) -> Dict:
        """
        Test one audio file with one model variant
        
        Args:
            audio_path: Path to test audio
            model_type: vad, stream-vad, or aed
            quant_type: fp32, int16, int8, int8-ch
            gguf_dir: Directory containing GGUF models
            
        Returns:
            Test results
        """
        print(f"\n{'='*70}")
        print(f"Testing: {audio_path.name}")
        print(f"Model: {model_type} ({quant_type})")
        print(f"{'='*70}")
        
        # Run PyTorch inference (golden reference)
        print("Running PyTorch inference (golden reference)...")
        pytorch_output = self.run_pytorch_inference(audio_path, model_type)
        
        # Run GGUF inference
        # TODO: This needs C++ integration or Python GGUF inference wrapper
        print("⚠️ GGUF inference not yet implemented")
        print("   Requires: C++ loader or Python GGUF inference wrapper")
        
        return {
            'status': 'pytorch_only',
            'pytorch_output': pytorch_output,
            'note': 'GGUF inference requires C++ integration'
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference-Level Golden Test')
    parser.add_argument('--audio', type=Path, required=True,
                        help='Audio file to test')
    parser.add_argument('--model', choices=['vad', 'stream-vad', 'aed'], default='vad',
                        help='Model type')
    parser.add_argument('--quant', choices=['fp32', 'int16', 'int8', 'int8-ch'], default='fp32',
                        help='Quantization type')
    parser.add_argument('--pth-dir', type=Path, default=None,
                        help='Directory containing PTH models')
    parser.add_argument('--gguf-dir', type=Path, default=None,
                        help='Directory containing GGUF models')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Default directories
    if args.pth_dir is None:
        args.pth_dir = Path(__file__).parent.parent.parent / "models_pth"
    if args.gguf_dir is None:
        args.gguf_dir = Path(__file__).parent.parent.parent / "models_gguf"
    
    # Verify audio file exists
    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    print(f"FireRed-VAD Inference Golden Test")
    print(f"{'='*70}")
    print(f"Audio: {args.audio}")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quant}")
    print(f"PTH models: {args.pth_dir}")
    print(f"GGUF models: {args.gguf_dir}")
    print()
    
    # Create tester
    tester = InferenceGoldenTester(args.pth_dir, verbose=args.verbose)
    
    # Load PyTorch models
    tester.load_pytorch_models()
    
    # Run test
    result = tester.test_audio_file(
        args.audio,
        args.model,
        args.quant,
        args.gguf_dir
    )
    
    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
