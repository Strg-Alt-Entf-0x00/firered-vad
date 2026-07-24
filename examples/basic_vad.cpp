// ==============================================================================
// basic_vad.cpp - Simple Voice Activity Detection Example
// ==============================================================================

#include <firered-vad/vad.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "FireRed-VAD Example\n";
    std::cout << "===================\n\n";
    
    // Initialize VAD
    firered::VAD vad("models/firered-vad.gguf");
    
    std::cout << "VAD initialized successfully\n\n";
    
    // Generate test audio (sine wave = speech-like signal)
    const int sample_rate = 16000;
    const int frame_size = 512;
    std::vector<float> audio_frame(frame_size);
    
    // Test with speech-like signal
    std::cout << "Testing with sine wave (simulated speech):\n";
    for (int i = 0; i < frame_size; ++i) {
        audio_frame[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * i / sample_rate);
    }
    
    bool is_speech = vad.detect(audio_frame.data(), audio_frame.size());
    std::cout << "  Result: " << (is_speech ? "SPEECH DETECTED" : "No speech") << "\n\n";
    
    // Test with silence
    std::cout << "Testing with silence:\n";
    std::fill(audio_frame.begin(), audio_frame.end(), 0.0f);
    
    is_speech = vad.detect(audio_frame.data(), audio_frame.size());
    std::cout << "  Result: " << (is_speech ? "SPEECH DETECTED" : "No speech") << "\n\n";
    
    std::cout << "Example completed\n";
    
    return 0;
}
