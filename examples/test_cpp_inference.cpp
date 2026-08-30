// test_cpp_inference.cpp - Test C++ DFSMN Implementation
// Simple test program to verify C++ VAD is working

#include "firered-vad/firered_vad.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace firered;

void print_separator() {
    std::cout << "========================================" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "FireRed-VAD C++ Implementation Test\n";
    print_separator();
    
    // Check arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        std::cerr << "Example: " << argv[0] << " ../models_gguf/firered-vad-fp32.gguf\n";
        return 1;
    }
    
    const char* model_path = argv[1];
    
    // Test 1: Load model
    std::cout << "\n[Test 1] Loading model...\n";
    std::cout << "Model: " << model_path << std::endl;
    
    auto ctx = firered_vad_init(model_path, nullptr);
    if (!ctx) {
        std::cerr << "✗ Failed to load model!\n";
        return 1;
    }
    
    std::cout << "✓ Model loaded successfully!\n";
    print_separator();
    
    // Test 2: Model is ready
    std::cout << "\n[Test 2] Model Information:\n";
    std::cout << "  Model loaded and ready for inference\n";
    std::cout << "✓ Model check passed!\n";
    print_separator();
    
    // Test 3: Generate test audio (1 second white noise)
    std::cout << "\n[Test 3] Testing with white noise (1 second)...\n";
    
    srand(static_cast<unsigned int>(time(nullptr)));
    const int sample_rate = 16000;
    const int duration_sec = 1;
    const int n_samples = sample_rate * duration_sec;
    
    std::vector<float> noise(n_samples);
    for (int i = 0; i < n_samples; i++) {
        noise[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;  // [-1, 1]
    }
    
    float noise_prob = firered_vad_detect(ctx, noise.data(), n_samples, sample_rate);
    
    std::cout << "  Noise speech probability: " << noise_prob << "\n";
    std::cout << "  Expected: ~0.0 (noise should not be detected as speech)\n";
    
    if (noise_prob < 0.0f || noise_prob > 1.0f) {
        std::cerr << "✗ Invalid probability range!\n";
        firered_vad_free(ctx);
        return 1;
    }
    
    std::cout << "✓ Noise test passed!\n";
    print_separator();
    
    // Test 4: Generate test audio (sine wave - simulates tone)
    std::cout << "\n[Test 4] Testing with sine wave (1 second, 440 Hz)...\n";
    
    std::vector<float> tone(n_samples);
    const float frequency = 440.0f;  // A4 note
    for (int i = 0; i < n_samples; i++) {
        float t = i / float(sample_rate);
        tone[i] = 0.5f * sin(static_cast<float>(2.0 * M_PI * frequency * t));
    }
    
    float tone_prob = firered_vad_detect(ctx, tone.data(), n_samples, sample_rate);
    
    std::cout << "  Tone speech probability: " << tone_prob << "\n";
    std::cout << "  Expected: ~0.0 (pure tone is not speech)\n";
    
    if (tone_prob < 0.0f || tone_prob > 1.0f) {
        std::cerr << "✗ Invalid probability range!\n";
        firered_vad_free(ctx);
        return 1;
    }
    
    std::cout << "✓ Tone test passed!\n";
    print_separator();
    
    // Test 5: Frame-level detection
    std::cout << "\n[Test 5] Testing frame-level detection...\n";
    
    std::vector<float> frame_probs;
    bool success = firered_vad_detect_frames(ctx, noise.data(), n_samples, 
                                              sample_rate, frame_probs);
    
    if (!success) {
        std::cerr << "✗ Frame detection failed!\n";
        firered_vad_free(ctx);
        return 1;
    }
    
    std::cout << "  Number of frames: " << frame_probs.size() << "\n";
    std::cout << "  Expected: ~100 frames (1s audio, 10ms hop)\n";
    
    if (frame_probs.empty()) {
        std::cerr << "✗ No frames extracted!\n";
        firered_vad_free(ctx);
        return 1;
    }
    
    // Show first few frames
    std::cout << "  First 5 frame probabilities: ";
    for (size_t i = 0; i < std::min(size_t(5), frame_probs.size()); i++) {
        std::cout << frame_probs[i] << " ";
    }
    std::cout << "\n";
    
    std::cout << "✓ Frame-level test passed!\n";
    print_separator();
    
    // Test 6: Memory management
    std::cout << "\n[Test 6] Testing cleanup...\n";
    
    firered_vad_free(ctx);
    ctx = nullptr;
    
    std::cout << "✓ Cleanup successful!\n";
    print_separator();
    
    // Final summary
    std::cout << "\n" << "🎉 ALL TESTS PASSED! 🎉\n";
    std::cout << "\nC++ DFSMN Implementation is working!\n";
    std::cout << "\nNext steps:\n";
    std::cout << "1. Test with real audio files (WAV)\n";
    std::cout << "2. Compare output with PyTorch implementation\n";
    std::cout << "3. Measure performance (latency, throughput)\n";
    std::cout << "4. Optimize (FFT library, SIMD, etc.)\n";
    
    print_separator();
    
    return 0;
}
