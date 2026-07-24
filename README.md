# FireRed-VAD - State-of-the-Art Voice Activity Detection

[![Version](https://img.shields.io/badge/version-0.6.0-blue.svg)](https://github.com/Strg-Alt-Entf-0x00/firered-vad)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache%202.0-green.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-20-orange.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

High-accuracy, ML-based Voice Activity Detection with **3 operating modes** using GGUF/GGML backend.

## Features

- **State-of-the-art accuracy:** 97.57% F1 Score (FLEURS-VAD-102 benchmark)
- **Three operating modes:**
  - **Standard VAD:** Highest accuracy with lookback+lookahead context (offline processing)
  - **Streaming VAD:** Real-time processing with no lookahead (live audio)
  - **AED (Audio Event Detection):** Classify speech/music/singing
- **Fast inference:** Optimized GGML compute graphs
- **Minimal dependencies:** Only GGML (no ONNX Runtime)
- **Cross-platform:** Windows, Linux, macOS
- **GPU acceleration:** Optional CUDA support
- **Easy integration:** C API + C++ RAII wrapper
- **Multilingual:** 100+ languages supported

## 🚀 Quick Start (30 seconds)

```cpp
#include "firered-vad/firered_vad.h"

// Load model (auto-detects mode from filename)
auto* vad = firered_vad_init("models/firered-vad.gguf", nullptr);

// Detect speech
float prob = firered_vad_detect(vad, audio_buffer, sample_count, 16000);

if (prob > 0.5f) {
    printf("Speech detected! (%.1f%%)\n", prob * 100);
}

// Cleanup
firered_vad_free(vad);
```

## 📦 Three Models, Three Modes

| Model | Mode | Size | Use Case | Accuracy |
|-------|------|------|----------|----------|
| `firered-vad.gguf` | **Standard** | 2.4 MB | Offline, highest accuracy | **97.57% F1** |
| `firered-stream-vad.gguf` | **Streaming** | 2.3 MB | Real-time, low latency | ~96% F1 |
| `firered-aed-vad.gguf` | **AED** | 2.4 MB | Speech/Music/Singing classification | - |

**Download:** [HuggingFace - FireRed-VAD GGUF](https://huggingface.co/cstr/firered-vad-GGUF)

Mode is auto-detected from filename, or can be manually set in config.

## Requirements

- **CMake**: 3.20 or higher
- **Compiler**: C++20 support required
  - MSVC 2019 16.11+ (Visual Studio 2019)
  - MSVC 2022 (recommended)
  - GCC 10+
  - Clang 12+
- **Platform**: Windows, Linux, macOS
- **Optional**: CUDA Toolkit for GPU acceleration

## Building

### Windows
```cmd
cd D:\Third-party-cpp\firered-vad-0.6.0
build.bat
```

### Linux/macOS
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install . --prefix ../install
```

Build without CUDA:
```bash
cmake -DFireRed_VAD_USE_CUDA=OFF ..
```

## Operating Modes

### 1. 🎯 Standard VAD (Highest Accuracy)

**Model:** `firered-vad.gguf` (2.4 MB)  
**Use case:** Offline processing, ASR preprocessing, maximum accuracy  
**Features:** Lookback + lookahead context for best accuracy

```cpp
auto* vad = firered_vad_init("models/firered-vad.gguf", nullptr);

// Single probability for entire audio
float prob = firered_vad_detect(vad, audio, n_samples, 16000);

// Frame-level probabilities for segmentation
std::vector<float> frame_probs;
firered_vad_detect_frames(vad, audio, n_samples, 16000, frame_probs);

// Automatic segmentation
auto segments = firered_vad_segment(frame_probs, 0.5f, 0.3f, 10, 5);
for (auto [start, end] : segments) {
    printf("Speech: frame %d to %d\n", start, end);
}

firered_vad_free(vad);
```

### 2. ⚡ Streaming VAD (Real-Time)

**Model:** `firered-stream-vad.gguf` (2.3 MB)  
**Use case:** Live audio, microphone input, real-time applications  
**Features:** No lookahead, minimal latency, state management

```cpp
auto* vad = firered_vad_init("models/firered-stream-vad.gguf", nullptr);

// Process audio chunks in real-time
while (recording) {
    float chunk[480];  // 30ms at 16kHz
    capture_audio(chunk, 480);
    
    float prob = firered_vad_detect(vad, chunk, 480, 16000);
    
    if (prob > 0.5f) {
        // Speech detected, start recording/transcribing
    }
}

// Reset state for new audio stream
firered_vad_reset(vad);

firered_vad_free(vad);
```

### 3. 🎵 Audio Event Detection (AED)

**Model:** `firered-aed-vad.gguf` (2.4 MB)  
**Use case:** Audio classification, content filtering, music detection  
**Features:** 3-way classification (speech/music/singing)

```cpp
auto* vad = firered_vad_init("models/firered-aed-vad.gguf", nullptr);

// Detect audio event type
fireredAEDResult result = firered_aed_detect(vad, audio, n_samples, 16000);

printf("Speech:  %.1f%%\n", result.speech_prob * 100);
printf("Music:   %.1f%%\n", result.music_prob * 100);
printf("Singing: %.1f%%\n", result.singing_prob * 100);
printf("Type: %s\n", result.dominant_event());

// Frame-level AED results
std::vector<fireredAEDResult> aed_frames;
firered_aed_detect_frames(vad, audio, n_samples, 16000, aed_frames);

firered_vad_free(vad);
```

## 🔧 C++ Wrapper (RAII)

```cpp
#include "firered-vad/firered_vad.h"

// Standard mode - automatic resource management
firered::fireredVAD vad("models/firered-vad.gguf");

// Simple usage
if (vad.has_speech(audio)) {
    auto probs = vad.detect_frames(audio);
    auto segments = vad.segment(audio);
}

// AED mode
firered::fireredVAD aed("models/firered-aed-vad.gguf");
auto result = aed.detect_aed(audio);

if (result.music_prob > 0.7f) {
    printf("Music detected!\n");
}

// Check current mode
if (aed.mode() == firered::fireredVADMode::AED) {
    std::cout << "AED mode active\n";
}
```

## API Reference

### Initialization

```cpp
fireredVADContext* firered_vad_init(const char* model_path, const fireredVADConfig* config);
```

Mode is auto-detected from filename:
- `*-stream-*` → Streaming mode
- `*-aed-*` → AED mode  
- Otherwise → Standard mode

### Configuration

```cpp
struct fireredVADConfig {
    int sample_rate_hz = 16000;       // Input sample rate (16kHz required)
    int frame_size_ms = 30;           // Frame size (30ms for FireRed)
    int hop_size_ms = 10;             // Hop size
    float threshold = 0.5f;           // Speech probability threshold
    int n_threads = 4;                // CPU threads
    bool use_gpu = false;             // Use CUDA if available
    fireredVADMode mode = Standard;   // Operating mode (auto-detected)
};
```

### Standard/Streaming Mode Functions

```cpp
// Single probability for entire audio
float firered_vad_detect(fireredVADContext* ctx, const float* audio, 
                         int n_samples, int sample_rate_hz);

// Frame-level probabilities
bool firered_vad_detect_frames(fireredVADContext* ctx, const float* audio,
                               int n_samples, int sample_rate_hz,
                               std::vector<float>& frame_probabilities);

// Automatic speech/silence segmentation
std::vector<std::pair<int, int>> firered_vad_segment(
    const std::vector<float>& frame_probs,
    float onset_threshold = 0.5f,
    float offset_threshold = 0.3f,
    int min_speech_frames = 10,
    int min_silence_frames = 5
);

// Reset RNN state (for new audio stream)
void firered_vad_reset(fireredVADContext* ctx);
```

### AED Mode Functions

```cpp
// Audio event detection result
struct fireredAEDResult {
    float speech_prob;    // Probability of speech (0.0 - 1.0)
    float music_prob;     // Probability of music (0.0 - 1.0)
    float singing_prob;   // Probability of singing (0.0 - 1.0)
    
    const char* dominant_event() const;  // "speech", "music", or "singing"
};

// Single AED result for entire audio
fireredAEDResult firered_aed_detect(fireredVADContext* ctx, const float* audio,
                                    int n_samples, int sample_rate_hz);

// Frame-level AED results
bool firered_aed_detect_frames(fireredVADContext* ctx, const float* audio,
                               int n_samples, int sample_rate_hz,
                               std::vector<fireredAEDResult>& aed_results);
```

### Utility Functions

```cpp
// Get current operating mode
fireredVADMode firered_vad_get_mode(fireredVADContext* ctx);

// Get model metadata
const char* firered_vad_get_metadata(fireredVADContext* ctx, const char* key);

// Get last error message
const char* firered_vad_get_error();

// Cleanup
void firered_vad_free(fireredVADContext* ctx);
```

## CMake Integration

```cmake
# In your project's CMakeLists.txt
set(THIRD_PARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third-party-cpp")
add_subdirectory("${THIRD_PARTY_ROOT}/firered-vad-0.6.0" firered-vad)

# Link to your target
add_executable(your_app main.cpp)
target_link_libraries(your_app PRIVATE firered-vad)
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `FireRed_VAD_BUILD_TESTS` | `OFF` | Build test programs |
| `FireRed_VAD_BUILD_EXAMPLES` | `OFF` | Build example programs |
| `FireRed_VAD_USE_CUDA` | `ON` | Enable CUDA acceleration |

## Common Use Cases

### 1. ASR Preprocessing (Standard Mode)

```cpp
// Remove silence before ASR
auto* vad = firered_vad_init("models/firered-vad.gguf", nullptr);
auto* asr = whisper_init("models/whisper.gguf");

std::vector<float> frame_probs;
firered_vad_detect_frames(vad, audio, n_samples, 16000, frame_probs);

auto segments = firered_vad_segment(frame_probs, 0.5f, 0.3f, 10, 5);

for (auto [start, end] : segments) {
    auto speech_audio = extract_frames(audio, start, end);
    auto text = whisper_transcribe(asr, speech_audio.data(), speech_audio.size());
    printf("%s\n", text.c_str());
}
```

### 2. Voice Assistant Wake (Streaming Mode)

```cpp
auto* stream = firered_vad_init("models/firered-stream-vad.gguf", nullptr);

while (true) {
    auto chunk = capture_microphone(30); // 30ms
    if (firered_vad_detect(stream, chunk, 480, 16000) > 0.5f) {
        start_listening();
    }
}
```

### 3. Skip Music Before ASR (AED Mode)

```cpp
auto* aed = firered_vad_init("models/firered-aed-vad.gguf", nullptr);
auto result = firered_aed_detect(aed, audio, n, sr);

if (result.music_prob < 0.3f && result.speech_prob > 0.5f) {
    run_asr(audio);  // Only transcribe speech
}
```

## Performance

### Latency (CPU)
- **Standard mode:** ~50ms per 30ms frame
- **Streaming mode:** ~30ms per 30ms frame (optimized)
- **AED mode:** ~50ms per 30ms frame

### Memory Usage
- **Model:** 2.3-2.4 MB
- **RNN state:** ~128 floats (512 bytes)
- **Compute buffer:** ~16 MB (temporary)

### GPU Acceleration

```cpp
fireredVADConfig config;
config.use_gpu = true;  // Enable CUDA

auto* vad = firered_vad_init("models/firered-vad.gguf", &config);
// 5-10x faster on GPU
```

### Mode Comparison

| Feature | Standard | Streaming | AED |
|---------|----------|-----------|-----|
| **Lookahead** | Yes (5-10 frames) | No | Yes |
| **Lookback** | Yes (5-10 frames) | Limited | Yes |
| **Latency** | Medium | Low | Medium |
| **Accuracy** | Highest (97.57%) | High (~96%) | - |
| **Use case** | Offline | Live | Classification |
| **Output** | 1 probability | 1 probability | 3 probabilities |

## Best Practices

### Audio Format
- **Sample rate:** 16 kHz (required)
- **Format:** 32-bit float PCM, normalized [-1.0, 1.0]
- **Frame size:** 30ms (480 samples at 16kHz)

### Threshold Tuning
```cpp
vad.set_threshold(0.5f);  // Default - balanced
vad.set_threshold(0.3f);  // More sensitive (more false positives)
vad.set_threshold(0.7f);  // Less sensitive (may miss speech)
```

### Performance Tips

1. **GPU Acceleration:** Set `config.use_gpu = true` for 5-10x speedup
2. **Batch Processing:** Use `detect_frames()` instead of individual frame calls
3. **Reset State:** Call `firered_vad_reset()` between independent audio streams
4. **Choose Right Mode:** 
   - Standard: Maximum accuracy, offline
   - Streaming: Minimal latency, live audio
   - AED: Classification, content filtering

## Why FireRed-VAD?

### vs. Silero-VAD
- ✅ Higher accuracy (97.57% vs ~95%)
- ✅ More languages (100+ vs multilingual)
- ✅ Better architecture (DFSMN vs LSTM)
- ✅ More features (AED mode)

### vs. WebRTC VAD
- ✅ ML-based, not energy-based
- ✅ Much higher accuracy (97% vs 80%)
- ✅ Language-agnostic
- ❌ Higher latency
- ❌ Larger size (2.4 MB vs 100 KB)

### vs. ONNX-based VAD
- ✅ Unified backend with GGML-based ASR
- ✅ Shared memory pool
- ✅ Single CUDA context
- ✅ Fewer dependencies (no ONNX Runtime)
- ✅ Easier deployment

## Troubleshooting

### Model Not Found
- Download from [HuggingFace](https://huggingface.co/cstr/firered-vad-GGUF)
- Place at `models/firered-vad.gguf`
- Check path is correct

### Low Accuracy
- Verify audio sample rate (16 kHz required)
- Check audio is normalized properly [-1.0, 1.0]
- Adjust threshold value
- Ensure correct mode for use case

### CUDA Errors
- Ensure CUDA Toolkit is installed
- Verify GPU is CUDA-capable
- Build with `-DFireRed_VAD_USE_CUDA=OFF` for CPU-only

### Error Handling
```cpp
auto* vad = firered_vad_init("models/firered-vad.gguf", nullptr);
if (!vad) {
    fprintf(stderr, "Failed to load: %s\n", firered_vad_get_error());
    return 1;
}
```

## Technical Details

### Architecture
- **Model:** DFSMN (Deep Feedforward Sequential Memory Network)
- **Input:** 16kHz mono PCM float32
- **Frame size:** 30ms (480 samples at 16kHz)
- **Output:** Probability [0.0, 1.0] or 3-class AED result

### GGML Backend
- Quantized neural network inference
- CPU and GPU acceleration
- Optimized for real-time processing
- Unified backend with GGML-based ASR models

## License

- **Library code:** MIT License
- **FireRed-VAD models:** Apache 2.0 License

**Free for non-commercial use. Commercial use allowed under Apache 2.0 terms.**

## Credits & References

- **FireRed-VAD:** FireRedTeam - https://huggingface.co/FireRedTeam/FireRedVAD
- **GGUF Conversion:** cstr - https://huggingface.co/cstr/firered-vad-GGUF
- **GGML/GGUF:** ggml-org - https://github.com/ggerganov/ggml
- **Paper:** [FireRed-VAD: Training Voice Activity Detection with Semi-supervised Learning](https://arxiv.org/abs/2410.09363)
- **Benchmark:** FLEURS-VAD-102 - https://huggingface.co/datasets/google/fleurs

---

**Version:** 0.6.0  
**Status:** ✅ Production Ready  
**Last Updated:** 2026-07-13
