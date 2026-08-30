# FireRed-VAD (C++ / GGUF) - High-Performance Inference Engine

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/Strg-Alt-Entf-0x00/firered-vad)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)]()
[![Parity](https://img.shields.io/badge/PyTorch%20Parity-100%25-success.svg)]()

A **zero-dependency, memory-safe, and highly optimized C++ inference engine** for [FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD). We provide a complete end-to-end pipeline including native Kaldi Fbank extraction, exact CMVN normalization, and GGML-based DFSMN graph execution.

This repository allows you to run FireRedVAD on edge devices, desktops, and servers without PyTorch, Torchaudio, or Kaldi, achieving up to **~53x realtime** processing speed on a single CPU core.

## 🎯 Key Achievements & Features

- **100% Mathematical Parity**: The C++ inference engine has been scientifically validated to produce the exact same raw logits and probabilities as the PyTorch golden standard. All Kaldi-specific quirks (e.g., asymmetric padding, energy floor `log(max(sum, eps))`, and continuous mel-bank mapping) are natively replicated.
- **State-of-the-Art Performance**: Built on [GGML](https://github.com/ggerganov/ggml), the engine utilizes zero-allocation graph reuse during inference, keeping memory footprints incredibly small and execution blazingly fast.
- **Native Dependency-Free Audio Processing**: We implemented a custom $O(N \log N)$ Radix-2 Cooley-Tukey FFT and a Torchaudio-aligned Fbank extractor natively in C++.
- **Memory-Safe Architecture**: The entire library is wrapped in modern C++ RAII semantics (`std::unique_ptr`, `fireredVADContext` destructors), strictly guaranteeing 0 memory leaks.

## 🚀 Pre-converted GGUF Models

We host all pre-converted, quantized models on HuggingFace:
👉 **[Strg-Alt-Entf-0x00/FireRedVAD-GGUF](https://huggingface.co/Strg-Alt-Entf-0x00/FireRedVAD-GGUF)**

*Note: Please ensure you are using models uploaded **after August 30, 2026**, as earlier versions contained a transposition error in the FSMN lookahead filters.*

| Model Type | Target Use-Case | Latency / Context | Lookback / Lookahead |
|---|---|---|---|
| **Standard VAD** (`vad`) | Offline batch processing, highest accuracy | High (Bidirectional) | 20 frames / 20 frames |
| **Streaming VAD** (`stream-vad`) | Real-time live audio, low latency | Low (Causal Only) | 40 frames / 0 frames |
| **AED** (`aed`) | Speech/Music/Singing classification | High (Bidirectional) | 20 frames / 20 frames |

### Available Quantizations

| Type | Precision | Memory Savings | Speedup (vs FP32) | Recommendation |
|---|---|---|---|---|
| **FP32** | 32-bit Float | 1x (Base: ~2.3MB) | 1x | **Recommended for Desktop / Server** (Baseline reference quality, 100% precision) |
| **INT16** | 16-bit Integer | 2x | ~2x | High precision systems |
| **INT8-CH** | 8-bit Per-Channel | 4x | ~4x | **Recommended for Edge / Mobile / IoT** (Preserves DFSMN channel variance perfectly) |
| **INT8** | 8-bit Per-Tensor | 4x | ~4x | Maximum compression, slight accuracy drop |

## 📊 Scientific Benchmarks

*Hardware: Standard Windows Desktop CPU (x86_64, Single Threaded). Audio length: 18.696s.*
*Date: 2026-08-30*

| Model (FP32) | Inference Time | Real-Time Factor (RTF) | Processing Speed | PyTorch Parity |
|---|---|---|---|---|
| **Standard VAD** | 0.597s | 0.0319 | **31.3x realtime** | EXACT MATCH |
| **Streaming VAD** | 0.349s | 0.0187 | **53.6x realtime** | EXACT MATCH |
| **AED (3-class)** | 0.589s | 0.0315 | **31.7x realtime** | EXACT MATCH |

*Note: The C++ execution speeds above include the entire pipeline (WAV parsing, Fbank extraction, CMVN normalization, and GGML inference).*

## 💻 C++ Integration (CMake)

The library is designed for seamless CMake integration.

```cmake
# Add as a subdirectory or find_package
add_subdirectory(lib/firered-vad)
target_link_libraries(your_app FireRed::vad)
```

```cpp
#include <firered-vad/firered_vad.h>
#include <vector>
#include <iostream>

int main() {
    // 1. Initialize the engine (RAII memory safe)
    firered::fireredVAD detector("models/firered-stream-vad-int8-ch.gguf");
    
    // 2. Load 16kHz Float32 PCM Audio
    std::vector<float> audio_samples = /* ... */;
    
    // 3. Run Inference
    float speech_prob = detector.detect(audio_samples, 16000);
    
    std::cout << "Speech probability: " << speech_prob << std::endl;
    return 0;
}
```

## 🛠️ Tooling & Scripts

The repository includes a suite of professional Python scripts:

- `tools/models_converter/models_converter.py`: Converts original PyTorch `.pth.tar` checkpoints to `gguf`. Handles FSMN filter time-reversal mathematically required for GGML cross-correlation.
- `tools/download_pth_models.bat`: Fetches the original PyTorch weights.
- `tools/convert_all_models.bat`: Batch converts all architectures into all quantizations.
- `tools/golden_test/test_weight_fidelity.py`: Validates GGUF weight integrity against original PyTorch tensors.

## 🤝 Acknowledgments

- **[FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)**: The original creators of the DFSMN architecture and PyTorch implementation.
- **[GGML](https://github.com/ggerganov/ggml)**: The incredible tensor library powering the C++ inference engine.

---
*Developed with 🩵 to push the boundaries of embedded voice AI.*
