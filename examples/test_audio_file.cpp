// test_audio_file.cpp - Test C++ VAD with real audio files
// Professional WAV file testing for FireRed-VAD
// Supports 16-bit PCM WAV format (mono/stereo)

#include "firered-vad/firered_vad.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdint>

using namespace firered;

// Professional WAV file reader that handles various WAV formats
// Supports PCM format with proper chunk parsing
bool read_wav_file(const char* path, std::vector<float>& audio, int& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    
    // Read RIFF header
    char riff[4];
    file.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) {
        std::cerr << "Not a valid RIFF file" << std::endl;
        return false;
    }
    
    uint32_t file_size;
    file.read(reinterpret_cast<char*>(&file_size), 4);
    
    char wave[4];
    file.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) {
        std::cerr << "Not a WAVE file" << std::endl;
        return false;
    }
    
    // Parse chunks to find fmt and data
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;
    std::streampos data_position = 0;
    
    while (file.good() && !file.eof()) {
        char chunk_id[4];
        file.read(chunk_id, 4);
        if (file.gcount() != 4) break;
        
        uint32_t chunk_size;
        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            // Read format chunk
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            
            uint32_t byte_rate;
            file.read(reinterpret_cast<char*>(&byte_rate), 4);
            
            uint16_t block_align;
            file.read(reinterpret_cast<char*>(&block_align), 2);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            
            // Skip any extra format bytes
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        }
        else if (std::memcmp(chunk_id, "data", 4) == 0) {
            // Found data chunk
            data_size = chunk_size;
            data_position = file.tellg();
            break;  // We have what we need
        }
        else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }
    
    // Validate format
    if (audio_format != 1) {
        std::cerr << "Only PCM audio format is supported (format=" << audio_format << ")" << std::endl;
        return false;
    }
    
    if (bits_per_sample != 16) {
        std::cerr << "Only 16-bit audio is supported (bits=" << bits_per_sample << ")" << std::endl;
        return false;
    }
    
    if (data_size == 0) {
        std::cerr << "No data chunk found" << std::endl;
        return false;
    }
    
    // Read audio data
    file.seekg(data_position);
    int n_samples = data_size / 2;  // 16-bit = 2 bytes per sample
    
    std::vector<int16_t> raw_audio(n_samples);
    file.read(reinterpret_cast<char*>(raw_audio.data()), data_size);
    
    // Convert to float [-1, 1]
    audio.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        audio[i] = raw_audio[i] / 32768.0f;
    }
    
    // Convert stereo to mono if needed
    if (num_channels == 2) {
        std::vector<float> mono;
        mono.reserve(n_samples / 2);
        for (size_t i = 0; i < audio.size(); i += 2) {
            mono.push_back(audio[i]);
        }
        audio = std::move(mono);
    }
    else if (num_channels != 1) {
        std::cerr << "Only mono and stereo are supported (channels=" << num_channels << ")" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> <audio.wav>\n";
        std::cerr << "Example: " << argv[0] << " ../models_gguf/firered-vad-fp32.gguf ../example_wave/speech-mic-test.wav\n";
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* audio_path = argv[2];
    
    std::cout << "FireRed-VAD Real Audio Test\n";
    std::cout << "========================================\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Audio: " << audio_path << "\n\n";
    
    // Load model
    std::cout << "Loading model..." << std::endl;
    auto ctx = firered_vad_init(model_path, nullptr);
    if (!ctx) {
        std::cerr << "Failed to load model!\n";
        return 1;
    }
    std::cout << "✓ Model loaded\n\n";
    
    // Load audio file
    std::cout << "Loading audio file..." << std::endl;
    std::vector<float> audio;
    int sample_rate;
    
    if (!read_wav_file(audio_path, audio, sample_rate)) {
        firered_vad_free(ctx);
        return 1;
    }
    
    std::cout << "✓ Audio loaded\n";
    std::cout << "  Samples: " << audio.size() << "\n";
    std::cout << "  Sample rate: " << sample_rate << " Hz\n";
    std::cout << "  Duration: " << (audio.size() / float(sample_rate)) << " seconds\n\n";
    
    // Run VAD detection (average probability)
    std::cout << "Running VAD detection (average)..." << std::endl;
    float avg_prob = firered_vad_detect(ctx, audio.data(), audio.size(), sample_rate);
    
    std::cout << "✓ Detection complete\n";
    std::cout << "  Average speech probability: " << avg_prob << "\n\n";
    
    // Run frame-level detection
    std::cout << "Running frame-level detection..." << std::endl;
    std::vector<float> frame_probs;
    bool success = firered_vad_detect_frames(ctx, audio.data(), audio.size(), 
                                              sample_rate, frame_probs);
    
    if (!success) {
        std::cerr << "Frame detection failed!\n";
        firered_vad_free(ctx);
        return 1;
    }
    
    std::cout << "✓ Frame detection complete\n";
    std::cout << "  Frames: " << frame_probs.size() << "\n";
    
    // Calculate statistics
    float min_prob = 1.0f, max_prob = 0.0f, sum_prob = 0.0f;
    int speech_frames = 0;
    
    for (float prob : frame_probs) {
        if (prob < min_prob) min_prob = prob;
        if (prob > max_prob) max_prob = prob;
        sum_prob += prob;
        if (prob > 0.5f) speech_frames++;
    }
    
    float mean_prob = sum_prob / frame_probs.size();
    
    std::cout << "  Min probability: " << min_prob << "\n";
    std::cout << "  Max probability: " << max_prob << "\n";
    std::cout << "  Mean probability: " << mean_prob << "\n";
    std::cout << "  Speech frames (>0.5): " << speech_frames << " / " << frame_probs.size() 
              << " (" << (100.0f * speech_frames / frame_probs.size()) << "%)\n\n";
    
    // Show first 10 frame probabilities
    std::cout << "First 10 frame probabilities:\n";
    for (size_t i = 0; i < std::min(size_t(10), frame_probs.size()); i++) {
        std::cout << "  Frame " << i << ": " << frame_probs[i] << "\n";
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Test complete!\n";
    
    firered_vad_free(ctx);
    return 0;
}
