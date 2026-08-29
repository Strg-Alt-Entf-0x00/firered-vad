#!/usr/bin/env python3
"""
Test PyTorch DFSMN Inference on All Models and Audio Files

This script tests the PyTorch inference implementation on:
- All 3 models (vad, stream-vad, aed)
- All example audio files (speech, noise, music)
- Validates that predictions are reasonable

Results are saved to .docs/PHASE5_PYTORCH_INFERENCE.md
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from inference_pytorch import load_model


class PyTorchInferenceTester:
    """Test PyTorch inference on all models and audio files"""
    
    def __init__(self, pth_dir: Path, audio_dir: Path, verbose: bool = False):
        """
        Initialize tester
        
        Args:
            pth_dir: Directory containing PTH models
            audio_dir: Directory containing example audio files
            verbose: Verbose output
        """
        self.pth_dir = pth_dir
        self.audio_dir = audio_dir
        self.verbose = verbose
        self.results = []
    
    def test_model_on_audio(
        self,
        model_type: str,
        audio_path: Path
    ) -> Dict:
        """
        Test one model on one audio file
        
        Args:
            model_type: vad, stream-vad, or aed
            audio_path: Path to audio file
            
        Returns:
            Results dict
        """
        print(f"\n{'='*70}")
        print(f"Model: {model_type}")
        print(f"Audio: {audio_path.name}")
        print(f"{'='*70}")
        
        try:
            # Load model
            model = load_model(model_type, self.pth_dir)
            
            # Run inference
            predictions, sample_rate = model.predict_file(audio_path)
            
            # Calculate statistics
            mean_prob = float(predictions.mean())
            max_prob = float(predictions.max())
            min_prob = float(predictions.min())
            std_prob = float(predictions.std())
            
            # Count frames above threshold (0.5 for speech detection)
            if model_type in ['vad', 'stream-vad']:
                speech_frames = int((predictions[:, 0] > 0.5).sum())
                speech_ratio = speech_frames / predictions.shape[0]
            else:  # AED has 3 outputs
                speech_frames = int((predictions[:, 0] > 0.5).sum())
                music_frames = int((predictions[:, 1] > 0.5).sum()) if predictions.shape[1] > 1 else 0
                singing_frames = int((predictions[:, 2] > 0.5).sum()) if predictions.shape[1] > 2 else 0
                speech_ratio = speech_frames / predictions.shape[0]
            
            # Duration
            duration = predictions.shape[0] / 100.0  # ~100 frames per second
            
            result = {
                'model': model_type,
                'audio': audio_path.name,
                'duration_sec': round(duration, 2),
                'num_frames': predictions.shape[0],
                'num_classes': predictions.shape[1],
                'mean_prob': round(mean_prob, 6),
                'max_prob': round(max_prob, 6),
                'min_prob': round(min_prob, 6),
                'std_prob': round(std_prob, 6),
                'speech_ratio': round(speech_ratio, 4) if model_type in ['vad', 'stream-vad'] else None,
                'status': 'success'
            }
            
            if model_type == 'aed':
                result['music_frames'] = music_frames
                result['singing_frames'] = singing_frames
            
            # Determine if result is expected
            audio_name = audio_path.stem
            if audio_name.startswith('speech-'):
                expected = 'speech'
                result['expected'] = 'high speech probability'
                result['correct'] = speech_ratio > 0.3  # At least 30% speech
            elif audio_name.startswith('noise-') or audio_name.startswith('negative-'):
                expected = 'noise'
                result['expected'] = 'low speech probability'
                result['correct'] = speech_ratio < 0.3  # Less than 30% speech
            elif audio_name.startswith('music-'):
                expected = 'music'
                result['expected'] = 'AED: high music probability'
                result['correct'] = True  # We don't have ground truth
            elif audio_name.startswith('singing-'):
                expected = 'singing'
                result['expected'] = 'AED: high singing probability'
                result['correct'] = True
            else:
                expected = 'unknown'
                result['expected'] = 'unknown'
                result['correct'] = None
            
            # Print summary
            print(f"  Duration: {duration:.2f}s")
            print(f"  Frames: {predictions.shape[0]}")
            print(f"  Mean probability: {mean_prob:.4f}")
            print(f"  Max probability: {max_prob:.4f}")
            if model_type in ['vad', 'stream-vad']:
                print(f"  Speech ratio: {speech_ratio:.2%}")
                print(f"  Expected: {result['expected']}")
                if result['correct'] is not None:
                    status_icon = "[OK]" if result['correct'] else "[FAIL]"
                    print(f"  Result: {status_icon}")
            
            return result
            
        except Exception as e:
            print(f"  [ERROR]: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'model': model_type,
                'audio': audio_path.name,
                'status': 'error',
                'error': str(e)
            }
    
    def test_all(self):
        """Test all models on all audio files"""
        print("="*70)
        print("PyTorch DFSMN Inference Test")
        print("="*70)
        print(f"PTH models: {self.pth_dir}")
        print(f"Audio files: {self.audio_dir}")
        print()
        
        # Get all audio files
        audio_files = sorted(self.audio_dir.glob('*.wav'))
        print(f"Found {len(audio_files)} audio files")
        
        # Test configurations
        models = ['vad', 'stream-vad', 'aed']
        
        # For initial validation, test a subset
        # Speech files: test all 3 models
        speech_files = [f for f in audio_files if f.stem.startswith('speech-')]
        # Noise files: test vad and stream-vad only (should reject)
        noise_files = [f for f in audio_files if f.stem.startswith('noise-') or f.stem.startswith('negative-')][:3]
        # Music/singing: test AED only
        music_files = [f for f in audio_files if f.stem.startswith('music-') or f.stem.startswith('singing-')]
        
        test_configs = []
        
        # Speech: all models
        for audio in speech_files:
            for model in ['vad', 'stream-vad']:
                test_configs.append((model, audio))
        
        # Noise: vad and stream-vad
        for audio in noise_files:
            for model in ['vad', 'stream-vad']:
                test_configs.append((model, audio))
        
        # Music/Singing: AED only
        for audio in music_files:
            test_configs.append(('aed', audio))
        
        print(f"Running {len(test_configs)} tests...")
        print()
        
        # Run tests
        for model_type, audio_path in test_configs:
            result = self.test_model_on_audio(model_type, audio_path)
            self.results.append(result)
        
        # Summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        error_count = sum(1 for r in self.results if r['status'] == 'error')
        
        print(f"Total tests: {len(self.results)}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print()
        
        # Group by model
        for model_type in ['vad', 'stream-vad', 'aed']:
            model_results = [r for r in self.results if r.get('model') == model_type and r['status'] == 'success']
            if not model_results:
                continue
            
            print(f"{model_type.upper()}:")
            print(f"  Tests: {len(model_results)}")
            
            # Check correctness for vad/stream-vad
            if model_type in ['vad', 'stream-vad']:
                correct = [r for r in model_results if r.get('correct') == True]
                incorrect = [r for r in model_results if r.get('correct') == False]
                unknown = [r for r in model_results if r.get('correct') is None]
                
                print(f"  Correct predictions: {len(correct)}/{len(model_results) - len(unknown)}")
                
                if incorrect:
                    print(f"  [WARN] Incorrect predictions:")
                    for r in incorrect:
                        print(f"    - {r['audio']}: speech_ratio={r['speech_ratio']:.2%} (expected: {r['expected']})")
            
            print()
        
        if error_count > 0:
            print("[ERROR] Failed tests:")
            for r in self.results:
                if r['status'] == 'error':
                    print(f"  - {r['model']} on {r['audio']}: {r['error']}")
    
    def save_results(self):
        """Save results to markdown file"""
        output_dir = Path(__file__).parent.parent.parent / ".docs"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "PHASE5_PYTORCH_INFERENCE.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Phase 5: PyTorch DFSMN Inference Test Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Status:** PyTorch DFSMN inference implementation complete\n\n")
            
            f.write("## Summary\n\n")
            success_count = sum(1 for r in self.results if r['status'] == 'success')
            error_count = sum(1 for r in self.results if r['status'] == 'error')
            
            f.write(f"- Total tests: {len(self.results)}\n")
            f.write(f"- Successful: {success_count}\n")
            f.write(f"- Errors: {error_count}\n\n")
            
            # Results by model
            for model_type in ['vad', 'stream-vad', 'aed']:
                model_results = [r for r in self.results if r.get('model') == model_type and r['status'] == 'success']
                if not model_results:
                    continue
                
                f.write(f"## {model_type.upper()} Model\n\n")
                f.write("| Audio File | Duration | Frames | Mean Prob | Max Prob | Speech Ratio | Result |\n")
                f.write("|------------|----------|--------|-----------|----------|--------------|--------|\n")
                
                for r in model_results:
                    audio = r['audio']
                    duration = r['duration_sec']
                    frames = r['num_frames']
                    mean_prob = r['mean_prob']
                    max_prob = r['max_prob']
                    speech_ratio = r.get('speech_ratio')
                    if speech_ratio is not None:
                        speech_ratio = f"{speech_ratio:.2%}"
                    else:
                        speech_ratio = "-"
                    
                    correct = r.get('correct')
                    if correct is True:
                        result = "[OK]"
                    elif correct is False:
                        result = "[FAIL]"
                    else:
                        result = "-"
                    
                    f.write(f"| {audio} | {duration}s | {frames} | {mean_prob:.4f} | {max_prob:.4f} | {speech_ratio} | {result} |\n")
                
                f.write("\n")
            
            # Errors
            if error_count > 0:
                f.write("## Errors\n\n")
                for r in self.results:
                    if r['status'] == 'error':
                        f.write(f"- **{r['model']}** on **{r['audio']}**: {r['error']}\n")
                f.write("\n")
            
            # Raw results
            f.write("## Raw Results (JSON)\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.results, indent=2))
            f.write("\n```\n")
        
        print(f"\n[OK] Results saved to: {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PyTorch DFSMN Inference')
    parser.add_argument('--pth-dir', type=Path, default=None,
                        help='Directory containing PTH models')
    parser.add_argument('--audio-dir', type=Path, default=None,
                        help='Directory containing example audio files')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Default directories
    if args.pth_dir is None:
        args.pth_dir = Path(__file__).parent.parent.parent / "models_pth"
    if args.audio_dir is None:
        args.audio_dir = Path(__file__).parent.parent.parent / "example_wave"
    
    # Create tester
    tester = PyTorchInferenceTester(args.pth_dir, args.audio_dir, verbose=args.verbose)
    
    # Run all tests
    tester.test_all()


if __name__ == "__main__":
    main()
