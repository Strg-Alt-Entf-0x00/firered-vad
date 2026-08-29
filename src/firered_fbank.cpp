// firered_fbank.cpp - Fbank Feature Extraction Implementation
// Minimal but complete Mel-filterbank implementation

#include "firered_fbank.h"
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace firered {

FbankExtractor::FbankExtractor(const FbankConfig& config) 
    : config_(config) {
    InitMelBanks();
    InitWindow();
}

FbankExtractor::~FbankExtractor() {
}

float FbankExtractor::HzToMel(float hz) {
    // Convert Hz to Mel scale: mel = 2595 * log10(1 + hz/700)
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float FbankExtractor::MelToHz(float mel) {
    // Convert Mel scale to Hz: hz = 700 * (10^(mel/2595) - 1)
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void FbankExtractor::InitMelBanks() {
    // Initialize mel filterbanks (triangular filters)
    
    int fft_bins = config_.num_fft_bins / 2 + 1;  // Only positive frequencies
    mel_banks_.resize(config_.num_mel_bins);
    
    // Convert frequency range to mel scale
    float mel_low = HzToMel(config_.low_freq);
    float mel_high = HzToMel(config_.high_freq);
    float mel_step = (mel_high - mel_low) / (config_.num_mel_bins + 1);
    
    // Create mel filterbank center frequencies
    std::vector<float> mel_centers(config_.num_mel_bins + 2);
    for (int i = 0; i < config_.num_mel_bins + 2; i++) {
        mel_centers[i] = mel_low + i * mel_step;
    }
    
    // Convert mel centers back to Hz
    std::vector<float> hz_centers(config_.num_mel_bins + 2);
    for (int i = 0; i < config_.num_mel_bins + 2; i++) {
        hz_centers[i] = MelToHz(mel_centers[i]);
    }
    
    // Convert Hz to FFT bin indices
    std::vector<int> bin_centers(config_.num_mel_bins + 2);
    for (int i = 0; i < config_.num_mel_bins + 2; i++) {
        bin_centers[i] = static_cast<int>(
            (config_.num_fft_bins + 1) * hz_centers[i] / config_.sample_rate
        );
    }
    
    // Create triangular filters
    for (int i = 0; i < config_.num_mel_bins; i++) {
        mel_banks_[i].resize(fft_bins, 0.0f);
        
        int left = bin_centers[i];
        int center = bin_centers[i + 1];
        int right = bin_centers[i + 2];
        
        // Rising slope
        for (int j = left; j < center; j++) {
            if (center > left) {
                mel_banks_[i][j] = static_cast<float>(j - left) / (center - left);
            }
        }
        
        // Falling slope
        for (int j = center; j < right; j++) {
            if (right > center) {
                mel_banks_[i][j] = static_cast<float>(right - j) / (right - center);
            }
        }
    }
}

void FbankExtractor::InitWindow() {
    // Initialize Hamming window
    int frame_length = (config_.frame_length_ms * config_.sample_rate) / 1000;
    window_.resize(frame_length);
    
    for (int i = 0; i < frame_length; i++) {
        // Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
        window_[i] = 0.54f - 0.46f * std::cos(
            2.0f * M_PI * i / (frame_length - 1)
        );
    }
}

void FbankExtractor::ApplyWindow(float* frame, int frame_length) {
    for (int i = 0; i < frame_length; i++) {
        frame[i] *= window_[i];
    }
}

void FbankExtractor::ComputeFFT(
    const float* frame, 
    int frame_length,
    float* power_spectrum, 
    int fft_bins
) {
    // Simple DFT implementation (for production, use FFT library like FFTW or kiss_fft)
    // Computing only positive frequencies (DC to Nyquist)
    
    int num_bins = fft_bins / 2 + 1;
    
    for (int k = 0; k < num_bins; k++) {
        float real = 0.0f;
        float imag = 0.0f;
        
        for (int n = 0; n < frame_length; n++) {
            float angle = -2.0f * M_PI * k * n / fft_bins;
            real += frame[n] * std::cos(angle);
            imag += frame[n] * std::sin(angle);
        }
        
        // Power spectrum: |X[k]|^2
        power_spectrum[k] = real * real + imag * imag;
    }
}

void FbankExtractor::ApplyMelBanks(
    const float* power_spectrum,
    float* mel_features
) {
    // Apply mel filterbanks to power spectrum
    for (int i = 0; i < config_.num_mel_bins; i++) {
        float sum = 0.0f;
        
        for (size_t j = 0; j < mel_banks_[i].size(); j++) {
            sum += power_spectrum[j] * mel_banks_[i][j];
        }
        
        // Apply log (add small epsilon to avoid log(0))
        mel_features[i] = std::log(sum + 1e-10f);
    }
}

int FbankExtractor::GetNumFrames(int num_samples) const {
    int frame_length = (config_.frame_length_ms * config_.sample_rate) / 1000;
    int frame_shift = (config_.frame_shift_ms * config_.sample_rate) / 1000;
    
    if (num_samples < frame_length) {
        return 0;
    }
    
    return 1 + (num_samples - frame_length) / frame_shift;
}

std::vector<std::vector<float>> FbankExtractor::ExtractFeatures(
    const float* samples,
    int num_samples
) {
    int frame_length = (config_.frame_length_ms * config_.sample_rate) / 1000;
    int frame_shift = (config_.frame_shift_ms * config_.sample_rate) / 1000;
    int num_frames = GetNumFrames(num_samples);
    
    std::vector<std::vector<float>> features;
    if (num_frames == 0) {
        return features;
    }
    
    features.resize(num_frames);
    
    // Temporary buffers
    std::vector<float> frame_buffer(frame_length);
    std::vector<float> power_spectrum(config_.num_fft_bins / 2 + 1);
    
    for (int f = 0; f < num_frames; f++) {
        int start_sample = f * frame_shift;
        
        // Copy frame
        for (int i = 0; i < frame_length; i++) {
            if (start_sample + i < num_samples) {
                frame_buffer[i] = samples[start_sample + i];
            } else {
                frame_buffer[i] = 0.0f;  // Zero padding
            }
        }
        
        // Apply dithering (optional)
        if (config_.dither > 0.0f) {
            for (int i = 0; i < frame_length; i++) {
                // Simple dithering: add small random noise
                frame_buffer[i] += config_.dither * ((rand() / float(RAND_MAX)) - 0.5f);
            }
        }
        
        // Apply window
        ApplyWindow(frame_buffer.data(), frame_length);
        
        // Compute power spectrum via FFT
        ComputeFFT(frame_buffer.data(), frame_length, 
                   power_spectrum.data(), config_.num_fft_bins);
        
        // Apply mel filterbanks
        features[f].resize(config_.num_mel_bins);
        ApplyMelBanks(power_spectrum.data(), features[f].data());
    }
    
    return features;
}

} // namespace firered
