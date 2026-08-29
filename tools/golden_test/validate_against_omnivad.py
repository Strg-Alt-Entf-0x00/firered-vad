#!/usr/bin/env python3
"""
Validate PyTorch DFSMN Implementation Against Official OmniVAD

This script performs scientific validation of our PyTorch implementation
by comparing frame-by-frame outputs against the official OmniVAD library.

Purpose:
- Verify correctness of our DFSMN implementation
- Measure numerical accuracy (MAE, max error, correlation)
- Identify any implementation discrepancies
- Validate on multiple audio samples

Scientific Approach:
1. Load same model in both implementations
2. Process same audio file
3. Extract frame-level probabilities
4. Statistical comparison (MAE, SQNR, correlation)
5. Threshold analysis (agreement at different confidence levels)
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from inference_pytorch import FireRedVADPyTorch

# Try to import omnivad
try:
    import omnivad
    OMNIVAD_AVAILABLE = True
except ImportError:
    print("[ERROR] omnivad not installed")
    print("Install with: pip install omnivad")
    sys.exit(1)


class ValidationTester:
    """Compare our implementation against official OmniVAD"""
    
    def __init__(self, pth_dir: Path, audio_dir: Path, verbose: bool = False):
        """
        Initialize validation tester
        
        Args:
            pth_dir: Directory containing PTH models
            audio_dir: Directory containing test audio files
            verbose: Verbose output
        """
        self.pth_dir = pth_dir
        self.audio_dir = audio_dir
        self.verbose = verbose
        self.results = []
    
    def load_omnivad_model(self, model_type: str):
        """
        Load official OmniVAD model
        
        Args:
            model_type: 'vad', 'stream-vad', or 'aed'
            
        Returns:
            OmniVAD model instance
        """
        print(f"\n[OmniVAD] Loading {model_type}...")
        
        # Map our names to OmniVAD names
        omnivad_modes = {
            'vad': 'vad',
            'stream-vad': 'stream_vad',
            'aed': 'aed'
        }
        
        mode = omnivad_modes.get(model_type)
        if not mode:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # OmniVAD will use the models in pth_models/ if available
        model = omnivad.OmniVAD(
            model_dir=str(self.pth_dir / model_type.replace('-', '_')),
            mode=mode
        )
        
        print(f"[OK] OmniVAD {model_type} loaded")
        return model
    
    def run_omnivad_inference(
        self,
        model,
        audio_path: Path
    ) -> np.ndarray:
        """
        Run OmniVAD inference on audio file
        
        Args:
            model: OmniVAD model instance
            audio_path: Path to audio file
            
        Returns:
            Frame-level probabilities (n_frames, n_classes)
        """
        if self.verbose:
            print(f"  [OmniVAD] Processing {audio_path.name}...")
        
        # Run inference
        result = model.detect(str(audio_path))
        
        # Extract frame-level probabilities
        # OmniVAD returns dict with various info, we need raw frame probs
        if isinstance(result, dict):
            # Check for frame_probs key (might vary by version)
            if 'frame_probs' in result:
                probs = np.array(result['frame_probs'])
            elif 'probabilities' in result:
                probs = np.array(result['probabilities'])
            else:
                # Fallback: try to extract from timestamps
                print("[WARN] Could not find frame probabilities in OmniVAD output")
                print(f"Available keys: {result.keys()}")
                return None
        else:
            # Direct array return
            probs = np.array(result)
        
        if self.verbose:
            print(f"  [OmniVAD] Output shape: {probs.shape}")
        
        return probs
    
    def run_our_inference(
        self,
        model_type: str,
        audio_path: Path
    ) -> np.ndarray:
        """
        Run our PyTorch implementation inference
        
        Args:
            model_type: 'vad', 'stream-vad', or 'aed'
            audio_path: Path to audio file
            
        Returns:
            Frame-level probabilities (n_frames, n_classes)
        """
        if self.verbose:
            print(f"  [Ours] Processing {audio_path.name}...")
        
        # Load our model
        from inference_pytorch import load_model
        model = load_model(model_type, self.pth_dir)
        
        # Run inference
        predictions, sample_rate = model.predict_file(audio_path)
        
        if self.verbose:
            print(f"  [Ours] Output shape: {predictions.shape}")
        
        return predictions
    
    def compare_outputs(
        self,
        ours: np.ndarray,
        omnivad: np.ndarray,
        model_type: str,
        audio_name: str
    ) -> Dict:
        """
        Statistical comparison of outputs
        
        Args:
            ours: Our implementation output
            omnivad: OmniVAD output
            model_type: Model type
            audio_name: Audio file name
            
        Returns:
            Comparison metrics dict
        """
        print(f"\n{'='*70}")
        print(f"Comparing: {model_type} on {audio_name}")
        print(f"{'='*70}")
        
        # Check if OmniVAD output is valid
        if omnivad is None:
            return {
                'model': model_type,
                'audio': audio_name,
                'status': 'error',
                'error': 'OmniVAD output not available'
            }
        
        # Align frame counts (might differ slightly)
        min_frames = min(ours.shape[0], omnivad.shape[0])
        
        if ours.shape[0] != omnivad.shape[0]:
            print(f"[WARN] Frame count mismatch:")
            print(f"  Ours: {ours.shape[0]} frames")
            print(f"  OmniVAD: {omnivad.shape[0]} frames")
            print(f"  Using first {min_frames} frames for comparison")
        
        ours_aligned = ours[:min_frames]
        omnivad_aligned = omnivad[:min_frames]
        
        # Flatten for single-class models
        if ours_aligned.shape[1] == 1:
            ours_aligned = ours_aligned.flatten()
        if omnivad_aligned.shape[1] == 1:
            omnivad_aligned = omnivad_aligned.flatten()
        
        # Calculate metrics
        mae = np.mean(np.abs(ours_aligned - omnivad_aligned))
        max_error = np.max(np.abs(ours_aligned - omnivad_aligned))
        mse = np.mean((ours_aligned - omnivad_aligned) ** 2)
        rmse = np.sqrt(mse)
        
        # Correlation
        if ours_aligned.size > 1:
            correlation = np.corrcoef(ours_aligned.flatten(), omnivad_aligned.flatten())[0, 1]
        else:
            correlation = 1.0
        
        # SQNR (Signal-to-Quantization-Noise Ratio)
        signal_power = np.sum(omnivad_aligned ** 2)
        noise_power = np.sum((ours_aligned - omnivad_aligned) ** 2)
        
        if noise_power > 0:
            sqnr_db = 10 * np.log10(signal_power / noise_power)
        else:
            sqnr_db = float('inf')
        
        # Threshold agreement (at various confidence levels)
        thresholds = [0.3, 0.5, 0.7, 0.9]
        agreements = {}
        
        for thresh in thresholds:
            ours_binary = (ours_aligned > thresh).astype(int)
            omni_binary = (omnivad_aligned > thresh).astype(int)
            agreement = np.mean(ours_binary == omni_binary)
            agreements[f'thresh_{thresh}'] = float(agreement)
        
        # Print results
        print(f"\nNumerical Accuracy:")
        print(f"  MAE:         {mae:.6f}")
        print(f"  Max Error:   {max_error:.6f}")
        print(f"  RMSE:        {rmse:.6f}")
        print(f"  Correlation: {correlation:.6f}")
        print(f"  SQNR:        {sqnr_db:.2f} dB")
        
        print(f"\nThreshold Agreement:")
        for thresh in thresholds:
            agreement = agreements[f'thresh_{thresh}']
            print(f"  @ {thresh:.1f}: {agreement:.2%}")
        
        # Determine pass/fail
        # Criteria: MAE < 0.01 and correlation > 0.95
        passed = mae < 0.01 and correlation > 0.95
        status_icon = "[OK]" if passed else "[FAIL]"
        print(f"\nValidation: {status_icon}")
        
        if not passed:
            print("[WARN] Implementation differs from OmniVAD")
            if mae >= 0.01:
                print(f"  - MAE too high: {mae:.6f} >= 0.01")
            if correlation <= 0.95:
                print(f"  - Correlation too low: {correlation:.6f} <= 0.95")
        
        # Return metrics
        return {
            'model': model_type,
            'audio': audio_name,
            'status': 'success',
            'passed': passed,
            'num_frames': min_frames,
            'frame_mismatch': ours.shape[0] != omnivad.shape[0],
            'mae': float(mae),
            'max_error': float(max_error),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'sqnr_db': float(sqnr_db),
            'threshold_agreements': agreements
        }
    
    def validate_model_on_audio(
        self,
        model_type: str,
        audio_path: Path
    ) -> Dict:
        """
        Validate one model on one audio file
        
        Args:
            model_type: 'vad', 'stream-vad', or 'aed'
            audio_path: Path to audio file
            
        Returns:
            Validation results dict
        """
        try:
            # Run both implementations
            ours = self.run_our_inference(model_type, audio_path)
            
            # Load and run OmniVAD
            omnivad_model = self.load_omnivad_model(model_type)
            omnivad_output = self.run_omnivad_inference(omnivad_model, audio_path)
            
            # Compare
            result = self.compare_outputs(ours, omnivad_output, model_type, audio_path.name)
            
            return result
            
        except Exception as e:
            print(f"\n[ERROR] Validation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'model': model_type,
                'audio': audio_path.name,
                'status': 'error',
                'error': str(e)
            }
    
    def run_validation_suite(self):
        """Run complete validation suite"""
        print("="*70)
        print("PyTorch DFSMN Implementation Validation")
        print("Comparing against Official OmniVAD Library")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PTH models: {self.pth_dir}")
        print(f"Audio files: {self.audio_dir}")
        print()
        
        # Select representative audio files for validation
        audio_files = [
            'speech-mic-test.wav',
            'speech-welcome-constant-volume.wav',
            'noise-constant-white.wav',
            'negative-birds.wav'
        ]
        
        # Validate VAD and Stream-VAD on these files
        models = ['vad', 'stream-vad']
        
        print(f"Running {len(models) * len(audio_files)} validation tests...\n")
        
        for model_type in models:
            for audio_name in audio_files:
                audio_path = self.audio_dir / audio_name
                
                if not audio_path.exists():
                    print(f"[SKIP] Audio file not found: {audio_name}")
                    continue
                
                result = self.validate_model_on_audio(model_type, audio_path)
                self.results.append(result)
        
        # Summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        passed_count = sum(1 for r in self.results if r.get('passed', False))
        error_count = sum(1 for r in self.results if r['status'] == 'error')
        
        print(f"Total tests: {len(self.results)}")
        print(f"Completed: {success_count}")
        print(f"Passed (MAE<0.01, corr>0.95): {passed_count}/{success_count}")
        print(f"Errors: {error_count}")
        print()
        
        if passed_count == success_count and error_count == 0:
            print("[OK] ALL VALIDATIONS PASSED!")
            print("Our implementation matches OmniVAD within acceptable tolerances.")
        elif passed_count > 0:
            print("[PARTIAL] Some validations passed, others failed.")
            print("Review individual results for details.")
        else:
            print("[FAIL] No validations passed.")
            print("Implementation differs significantly from OmniVAD.")
        
        # Print failed tests
        failed = [r for r in self.results if r['status'] == 'success' and not r.get('passed', False)]
        if failed:
            print("\nFailed validations:")
            for r in failed:
                print(f"  - {r['model']} on {r['audio']}: MAE={r['mae']:.6f}, corr={r['correlation']:.6f}")
    
    def save_results(self):
        """Save validation results to file"""
        output_dir = Path(__file__).parent.parent.parent / ".docs"
        output_file = output_dir / "OMNIVAD_VALIDATION.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# OmniVAD Validation Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            
            success_count = sum(1 for r in self.results if r['status'] == 'success')
            passed_count = sum(1 for r in self.results if r.get('passed', False))
            error_count = sum(1 for r in self.results if r['status'] == 'error')
            
            f.write(f"- Total tests: {len(self.results)}\n")
            f.write(f"- Completed: {success_count}\n")
            f.write(f"- Passed: {passed_count}/{success_count}\n")
            f.write(f"- Errors: {error_count}\n\n")
            
            # Results table
            f.write("## Validation Results\n\n")
            f.write("| Model | Audio | MAE | Max Error | Correlation | SQNR (dB) | Status |\n")
            f.write("|-------|-------|-----|-----------|-------------|-----------|--------|\n")
            
            for r in self.results:
                if r['status'] == 'success':
                    model = r['model']
                    audio = r['audio']
                    mae = r['mae']
                    max_err = r['max_error']
                    corr = r['correlation']
                    sqnr = r['sqnr_db']
                    status = "[OK]" if r.get('passed', False) else "[FAIL]"
                    
                    f.write(f"| {model} | {audio} | {mae:.6f} | {max_err:.6f} | {corr:.6f} | {sqnr:.2f} | {status} |\n")
            
            f.write("\n")
            
            # Raw results JSON
            f.write("## Raw Results (JSON)\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.results, indent=2))
            f.write("\n```\n")
        
        print(f"\n[OK] Results saved to: {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate against OmniVAD')
    parser.add_argument('--pth-dir', type=Path, default=None,
                        help='Directory containing PTH models')
    parser.add_argument('--audio-dir', type=Path, default=None,
                        help='Directory containing test audio files')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Default directories
    if args.pth_dir is None:
        args.pth_dir = Path(__file__).parent.parent.parent / "models_pth"
    if args.audio_dir is None:
        args.audio_dir = Path(__file__).parent.parent.parent / "example_wave"
    
    # Create tester
    tester = ValidationTester(args.pth_dir, args.audio_dir, verbose=args.verbose)
    
    # Run validation
    tester.run_validation_suite()


if __name__ == "__main__":
    main()
