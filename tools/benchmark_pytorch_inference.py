#!/usr/bin/env python3
"""
Performance Benchmarking for PyTorch DFSMN Inference

Scientific performance measurement:
- Latency (ms per frame)
- Throughput (frames/sec)
- Memory usage (peak/average)
- CPU utilization
- Comparison across models and quantizations

All measurements include:
- Sample size (n measurements)
- Mean, median, std deviation
- Min/max values
- 95th percentile
- Timestamp for reproducibility
"""

import sys
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import tracemalloc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "golden_test"))
from inference_pytorch import load_model


class PerformanceBenchmark:
    """Scientific performance benchmarking"""
    
    def __init__(self, pth_dir: Path, audio_dir: Path, n_runs: int = 10):
        """
        Initialize benchmarker
        
        Args:
            pth_dir: Directory containing PTH models
            audio_dir: Directory containing test audio
            n_runs: Number of measurements per test (default: 10 for statistical significance)
        """
        self.pth_dir = pth_dir
        self.audio_dir = audio_dir
        self.n_runs = n_runs
        self.results = []
        
        # Get system info
        self.system_info = {
            'cpu': self._get_cpu_info(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_cpu_info(self) -> str:
        """Get CPU model name"""
        try:
            import platform
            return platform.processor()
        except:
            return "Unknown CPU"
    
    def measure_latency(
        self,
        model_type: str,
        audio_path: Path,
        n_runs: int = None
    ) -> Dict:
        """
        Measure inference latency
        
        Args:
            model_type: 'vad', 'stream-vad', or 'aed'
            audio_path: Path to test audio
            n_runs: Number of measurements
            
        Returns:
            Latency statistics dict
        """
        if n_runs is None:
            n_runs = self.n_runs
        
        print(f"\n[Benchmark] {model_type} on {audio_path.name}")
        print(f"  Measuring latency ({n_runs} runs)...")
        
        # Load model (once)
        model = load_model(model_type, self.pth_dir)
        
        # Load audio (once)
        predictions_ref, sample_rate = model.predict_file(audio_path)
        num_frames = predictions_ref.shape[0]
        duration_sec = num_frames / 100.0  # ~100 frames/sec
        
        # Warm-up run (exclude from measurements)
        _ = model.predict_file(audio_path)
        
        # Measure latencies
        latencies = []
        
        for i in range(n_runs):
            start = time.perf_counter()
            predictions, _ = model.predict_file(audio_path)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        latencies = np.array(latencies)
        
        # Calculate statistics
        stats = {
            'model': model_type,
            'audio': audio_path.name,
            'audio_duration_sec': duration_sec,
            'num_frames': num_frames,
            'n_runs': n_runs,
            'latency_ms': {
                'mean': float(np.mean(latencies)),
                'median': float(np.median(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
                'p95': float(np.percentile(latencies, 95))
            },
            'latency_per_frame_ms': {
                'mean': float(np.mean(latencies) / num_frames),
                'median': float(np.median(latencies) / num_frames)
            },
            'throughput_frames_per_sec': {
                'mean': float(num_frames / (np.mean(latencies) / 1000)),
                'median': float(num_frames / (np.median(latencies) / 1000))
            },
            'realtime_factor': {
                'mean': float(duration_sec / (np.mean(latencies) / 1000)),
                'median': float(duration_sec / (np.median(latencies) / 1000))
            }
        }
        
        # Print results
        print(f"  Latency (total):")
        print(f"    Mean: {stats['latency_ms']['mean']:.2f} ms ± {stats['latency_ms']['std']:.2f}")
        print(f"    Median: {stats['latency_ms']['median']:.2f} ms")
        print(f"    Range: [{stats['latency_ms']['min']:.2f}, {stats['latency_ms']['max']:.2f}] ms")
        print(f"  Latency (per frame):")
        print(f"    Mean: {stats['latency_per_frame_ms']['mean']:.3f} ms/frame")
        print(f"  Throughput:")
        print(f"    Mean: {stats['throughput_frames_per_sec']['mean']:.1f} frames/sec")
        print(f"  Realtime Factor:")
        print(f"    Mean: {stats['realtime_factor']['mean']:.2f}x")
        print(f"    (1.0x = realtime, >1.0x = faster than realtime)")
        
        return stats
    
    def measure_memory(
        self,
        model_type: str,
        audio_path: Path
    ) -> Dict:
        """
        Measure memory usage
        
        Args:
            model_type: 'vad', 'stream-vad', or 'aed'
            audio_path: Path to test audio
            
        Returns:
            Memory statistics dict
        """
        print(f"\n[Benchmark] Memory usage for {model_type}")
        
        # Start memory tracking
        tracemalloc.start()
        
        # Measure baseline
        baseline_mem = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        # Load model
        model = load_model(model_type, self.pth_dir)
        after_load_mem = psutil.Process().memory_info().rss / (1024**2)
        
        # Run inference
        predictions, sample_rate = model.predict_file(audio_path)
        after_inference_mem = psutil.Process().memory_info().rss / (1024**2)
        
        # Get peak memory from tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        stats = {
            'model': model_type,
            'audio': audio_path.name,
            'memory_mb': {
                'baseline': float(baseline_mem),
                'after_model_load': float(after_load_mem),
                'after_inference': float(after_inference_mem),
                'model_size_estimate': float(after_load_mem - baseline_mem),
                'inference_overhead': float(after_inference_mem - after_load_mem),
                'peak_tracemalloc': float(peak / (1024**2))
            }
        }
        
        print(f"  Baseline: {stats['memory_mb']['baseline']:.1f} MB")
        print(f"  After model load: {stats['memory_mb']['after_model_load']:.1f} MB")
        print(f"  After inference: {stats['memory_mb']['after_inference']:.1f} MB")
        print(f"  Model size (estimate): {stats['memory_mb']['model_size_estimate']:.1f} MB")
        print(f"  Inference overhead: {stats['memory_mb']['inference_overhead']:.1f} MB")
        print(f"  Peak (tracemalloc): {stats['memory_mb']['peak_tracemalloc']:.1f} MB")
        
        return stats
    
    def measure_cpu_utilization(
        self,
        model_type: str,
        audio_path: Path,
        duration_sec: float = 5.0
    ) -> Dict:
        """
        Measure CPU utilization during inference
        
        Args:
            model_type: Model type
            audio_path: Test audio
            duration_sec: Measurement duration
            
        Returns:
            CPU statistics dict
        """
        print(f"\n[Benchmark] CPU utilization for {model_type} ({duration_sec}s)")
        
        # Load model
        model = load_model(model_type, self.pth_dir)
        
        # Start CPU monitoring
        process = psutil.Process()
        cpu_samples = []
        
        start_time = time.time()
        
        # Run inference repeatedly and sample CPU
        while (time.time() - start_time) < duration_sec:
            # Run inference
            _ = model.predict_file(audio_path)
            
            # Sample CPU
            cpu_percent = process.cpu_percent(interval=0.1)
            cpu_samples.append(cpu_percent)
        
        cpu_samples = np.array(cpu_samples)
        
        stats = {
            'model': model_type,
            'audio': audio_path.name,
            'measurement_duration_sec': duration_sec,
            'n_samples': len(cpu_samples),
            'cpu_percent': {
                'mean': float(np.mean(cpu_samples)),
                'median': float(np.median(cpu_samples)),
                'std': float(np.std(cpu_samples)),
                'min': float(np.min(cpu_samples)),
                'max': float(np.max(cpu_samples))
            }
        }
        
        print(f"  CPU usage: {stats['cpu_percent']['mean']:.1f}% ± {stats['cpu_percent']['std']:.1f}%")
        print(f"  Range: [{stats['cpu_percent']['min']:.1f}%, {stats['cpu_percent']['max']:.1f}%]")
        
        return stats
    
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite"""
        print("="*70)
        print("Performance Benchmark Suite")
        print("PyTorch DFSMN Inference")
        print("="*70)
        print(f"Date: {self.system_info['timestamp']}")
        print(f"CPU: {self.system_info['cpu']}")
        print(f"RAM: {self.system_info['memory_total_gb']:.1f} GB")
        print(f"Python: {self.system_info['python_version']}")
        print()
        
        # Select test audio (medium length for balanced measurement)
        test_audio = self.audio_dir / "speech-mic-test.wav"
        
        if not test_audio.exists():
            print(f"[ERROR] Test audio not found: {test_audio}")
            return
        
        # Benchmark all models
        models = ['vad', 'stream-vad', 'aed']
        
        for model_type in models:
            print(f"\n{'='*70}")
            print(f"Benchmarking: {model_type.upper()}")
            print(f"{'='*70}")
            
            # Latency
            latency_stats = self.measure_latency(model_type, test_audio)
            self.results.append(('latency', latency_stats))
            
            # Memory
            memory_stats = self.measure_memory(model_type, test_audio)
            self.results.append(('memory', memory_stats))
            
            # CPU utilization (shorter duration for efficiency)
            cpu_stats = self.measure_cpu_utilization(model_type, test_audio, duration_sec=3.0)
            self.results.append(('cpu', cpu_stats))
        
        # Save results
        self.save_results()
        self.print_comparison()
    
    def print_comparison(self):
        """Print comparison table across models"""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        # Extract latency results
        latency_results = [r[1] for r in self.results if r[0] == 'latency']
        
        if latency_results:
            print("\nLatency (Mean):")
            print("| Model | Total (ms) | Per Frame (ms) | Throughput (fps) | RT Factor |")
            print("|-------|------------|----------------|------------------|-----------|")
            
            for stats in latency_results:
                model = stats['model']
                total_ms = stats['latency_ms']['mean']
                per_frame_ms = stats['latency_per_frame_ms']['mean']
                throughput = stats['throughput_frames_per_sec']['mean']
                rt_factor = stats['realtime_factor']['mean']
                
                print(f"| {model:11s} | {total_ms:10.2f} | {per_frame_ms:14.3f} | {throughput:16.1f} | {rt_factor:9.2f}x |")
        
        # Extract memory results
        memory_results = [r[1] for r in self.results if r[0] == 'memory']
        
        if memory_results:
            print("\nMemory Usage:")
            print("| Model | Model Size (MB) | Inference Overhead (MB) | Peak (MB) |")
            print("|-------|-----------------|-------------------------|-----------|")
            
            for stats in memory_results:
                model = stats['model']
                model_size = stats['memory_mb']['model_size_estimate']
                inference = stats['memory_mb']['inference_overhead']
                peak = stats['memory_mb']['peak_tracemalloc']
                
                print(f"| {model:11s} | {model_size:15.1f} | {inference:23.1f} | {peak:9.1f} |")
        
        # Extract CPU results
        cpu_results = [r[1] for r in self.results if r[0] == 'cpu']
        
        if cpu_results:
            print("\nCPU Utilization:")
            print("| Model | Mean (%) | Std (%) | Range (%) |")
            print("|-------|----------|---------|-----------|")
            
            for stats in cpu_results:
                model = stats['model']
                mean = stats['cpu_percent']['mean']
                std = stats['cpu_percent']['std']
                min_cpu = stats['cpu_percent']['min']
                max_cpu = stats['cpu_percent']['max']
                
                print(f"| {model:11s} | {mean:8.1f} | {std:7.1f} | [{min_cpu:.1f}, {max_cpu:.1f}] |")
    
    def save_results(self):
        """Save benchmark results"""
        output_dir = Path(__file__).parent.parent / ".docs"
        output_file = output_dir / "PERFORMANCE_BENCHMARK.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Performance Benchmark Results\n\n")
            f.write(f"**Date:** {self.system_info['timestamp']}\n\n")
            
            f.write("## System Information\n\n")
            f.write(f"- **CPU:** {self.system_info['cpu']}\n")
            f.write(f"- **RAM:** {self.system_info['memory_total_gb']:.1f} GB\n")
            f.write(f"- **Python:** {self.system_info['python_version']}\n\n")
            
            f.write("## Results\n\n")
            
            # Latency
            f.write("### Latency\n\n")
            latency_results = [r[1] for r in self.results if r[0] == 'latency']
            if latency_results:
                f.write("| Model | Total (ms) | Per Frame (ms) | Throughput (fps) | RT Factor |\n")
                f.write("|-------|------------|----------------|------------------|-----------|\\n")
                
                for stats in latency_results:
                    model = stats['model']
                    total_ms = stats['latency_ms']['mean']
                    per_frame_ms = stats['latency_per_frame_ms']['mean']
                    throughput = stats['throughput_frames_per_sec']['mean']
                    rt_factor = stats['realtime_factor']['mean']
                    
                    f.write(f"| {model} | {total_ms:.2f} | {per_frame_ms:.3f} | {throughput:.1f} | {rt_factor:.2f}x |\n")
                
                f.write("\n")
            
            # Memory
            f.write("### Memory\n\n")
            memory_results = [r[1] for r in self.results if r[0] == 'memory']
            if memory_results:
                f.write("| Model | Model Size (MB) | Inference Overhead (MB) | Peak (MB) |\n")
                f.write("|-------|-----------------|-------------------------|-----------|\\n")
                
                for stats in memory_results:
                    model = stats['model']
                    model_size = stats['memory_mb']['model_size_estimate']
                    inference = stats['memory_mb']['inference_overhead']
                    peak = stats['memory_mb']['peak_tracemalloc']
                    
                    f.write(f"| {model} | {model_size:.1f} | {inference:.1f} | {peak:.1f} |\n")
                
                f.write("\n")
            
            # CPU
            f.write("### CPU Utilization\n\n")
            cpu_results = [r[1] for r in self.results if r[0] == 'cpu']
            if cpu_results:
                f.write("| Model | Mean (%) | Std (%) | Range (%) |\n")
                f.write("|-------|----------|---------|-----------|\\n")
                
                for stats in cpu_results:
                    model = stats['model']
                    mean = stats['cpu_percent']['mean']
                    std = stats['cpu_percent']['std']
                    min_cpu = stats['cpu_percent']['min']
                    max_cpu = stats['cpu_percent']['max']
                    
                    f.write(f"| {model} | {mean:.1f} | {std:.1f} | [{min_cpu:.1f}, {max_cpu:.1f}] |\n")
                
                f.write("\n")
            
            # Raw JSON
            f.write("## Raw Data (JSON)\n\n")
            f.write("```json\n")
            json_data = {
                'system_info': self.system_info,
                'results': [(metric, data) for metric, data in self.results]
            }
            f.write(json.dumps(json_data, indent=2))
            f.write("\n```\n")
        
        print(f"\n[OK] Results saved to: {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark PyTorch Inference')
    parser.add_argument('--pth-dir', type=Path, default=None,
                        help='Directory containing PTH models')
    parser.add_argument('--audio-dir', type=Path, default=None,
                        help='Directory containing test audio')
    parser.add_argument('--n-runs', type=int, default=10,
                        help='Number of latency measurements (default: 10)')
    
    args = parser.parse_args()
    
    # Default directories
    if args.pth_dir is None:
        args.pth_dir = Path(__file__).parent.parent / "models_pth"
    if args.audio_dir is None:
        args.audio_dir = Path(__file__).parent.parent / "example_wave"
    
    # Create benchmarker
    benchmarker = PerformanceBenchmark(args.pth_dir, args.audio_dir, n_runs=args.n_runs)
    
    # Run benchmarks
    benchmarker.run_benchmark_suite()


if __name__ == "__main__":
    main()
