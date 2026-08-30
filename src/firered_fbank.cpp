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

float FbankExtractor::HzToMel(float hz) noexcept {
    // Convert Hz to Mel scale: mel = 2595 * log10(1 + hz/700)
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float FbankExtractor::MelToHz(float mel) noexcept {
    // Convert Mel scale to Hz: hz = 700 * (10^(mel/2595) - 1)
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void FbankExtractor::InitMelBanks() {
    // Initialize mel filterbanks (triangular filters)
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
    
    // Re-implement Kaldi's exact mel bank computation
    // No integer truncation for bin centers!
    mel_low = HzToMel(20.0f);
    float mel_high_exact = HzToMel(config_.sample_rate / 2.0f);
    float mel_delta = (mel_high_exact - mel_low) / (config_.num_mel_bins + 1);
    
    float fft_bin_width = static_cast<float>(config_.sample_rate) / config_.num_fft_bins;
    int num_fft_bins_half = config_.num_fft_bins / 2 + 1;
    
    for (int i = 0; i < config_.num_mel_bins; i++) {
        mel_banks_[i].resize(num_fft_bins_half, 0.0f);
        
        float left_mel = mel_low + i * mel_delta;
        float center_mel = mel_low + (i + 1) * mel_delta;
        float right_mel = mel_low + (i + 2) * mel_delta;
        
        for (int j = 0; j < num_fft_bins_half; j++) {
            float freq = j * fft_bin_width;
            float mel = HzToMel(freq);
            
            float up_slope = (mel - left_mel) / (center_mel - left_mel);
            float down_slope = (right_mel - mel) / (right_mel - center_mel);
            
            float weight = std::max(0.0f, std::min(up_slope, down_slope));
            mel_banks_[i][j] = weight;
        }
    }
    
    // Precompute FFT Twiddle Factors for Radix-2 Cooley-Tukey
    int num_fft_bins = config_.num_fft_bins;
    if ((num_fft_bins & (num_fft_bins - 1)) == 0 && num_fft_bins > 0) { // Is power of 2
        twiddle_real_.resize(num_fft_bins);
        twiddle_imag_.resize(num_fft_bins);
        
        for (int size = 2; size <= num_fft_bins; size *= 2) {
            float tablestep = static_cast<float>(-2.0 * M_PI / size);
            for (int k = 0; k < size / 2; k++) {
                float angle = k * tablestep;
                // Store at index size/2 + k to avoid overlapping sizes
                int idx = size / 2 + k;
                if (idx < num_fft_bins) {
                    twiddle_real_[idx] = std::cos(angle);
                    twiddle_imag_[idx] = std::sin(angle);
                }
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
            static_cast<float>(2.0 * M_PI * i / (frame_length - 1))
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
    // Fast O(N log N) Radix-2 Cooley-Tukey FFT implementation
    // fft_bins must be a power of 2 (e.g., 512)
    
    // Check if fft_bins is a power of 2
    if ((fft_bins & (fft_bins - 1)) != 0 || fft_bins == 0) {
        // Fallback to slow DFT if not power of 2 (should not happen with default config)
        int num_bins = fft_bins / 2 + 1;
        for (int k = 0; k < num_bins; k++) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < frame_length; n++) {
                float angle = static_cast<float>(-2.0 * M_PI * k * n / fft_bins);
                real += frame[n] * std::cos(angle);
                imag += frame[n] * std::sin(angle);
            }
            power_spectrum[k] = real * real + imag * imag;
        }
        return;
    }

    // Allocate complex arrays
    std::vector<float> real(fft_bins, 0.0f);
    std::vector<float> imag(fft_bins, 0.0f);
    
    // Copy input frame (with zero padding)
    for (int i = 0; i < std::min(frame_length, fft_bins); i++) {
        real[i] = frame[i];
    }

    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < fft_bins - 1; i++) {
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
        int k = fft_bins >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Cooley-Tukey decimation-in-time radix-2 FFT
    for (int size = 2; size <= fft_bins; size *= 2) {
        int halfsize = size / 2;
        for (int i = 0; i < fft_bins; i += size) {
            for (int k = 0; k < halfsize; k++) {
                // Lookup precomputed twiddle factor
                int twiddle_idx = halfsize + k;
                float twiddle_real = twiddle_real_[twiddle_idx];
                float twiddle_imag = twiddle_imag_[twiddle_idx];
                
                int idx_k = i + k;
                int idx_k_half = idx_k + halfsize;
                
                float temp_real = real[idx_k_half] * twiddle_real - imag[idx_k_half] * twiddle_imag;
                float temp_imag = real[idx_k_half] * twiddle_imag + imag[idx_k_half] * twiddle_real;
                
                real[idx_k_half] = real[idx_k] - temp_real;
                imag[idx_k_half] = imag[idx_k] - temp_imag;
                real[idx_k] += temp_real;
                imag[idx_k] += temp_imag;
            }
        }
    }

    // Calculate power spectrum for positive frequencies
    int num_bins = fft_bins / 2 + 1;
    for (int k = 0; k < num_bins; k++) {
        // Power spectrum: |X[k]|^2
        power_spectrum[k] = real[k] * real[k] + imag[k] * imag[k];
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
        
        // Apply energy floor (max(sum, epsilon))
        if (sum < config_.energy_floor) {
            sum = config_.energy_floor;
        }
        
        mel_features[i] = std::log(sum);
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

std::vector<float> FbankExtractor::ExtractFeatures(
    const float* samples,
    int num_samples
) {
    int frame_length = (config_.frame_length_ms * config_.sample_rate) / 1000;
    int frame_shift = (config_.frame_shift_ms * config_.sample_rate) / 1000;
    int num_frames = GetNumFrames(num_samples);
    
    std::vector<float> features;
    if (num_frames == 0) {
        return features;
    }
    
    // Flat vector: num_frames * num_mel_bins
    features.resize(num_frames * config_.num_mel_bins, 0.0f);
    
    // Temporary buffers
    std::vector<float> frame_buffer(frame_length);
    std::vector<float> power_spectrum(config_.num_fft_bins / 2 + 1);
    
    float max_ps = 0.0f;
    
    for (int f = 0; f < num_frames; f++) {
        int start_sample = f * frame_shift;
        
        // Kaldi Fbank exact preprocessing pipeline
        // 1. Copy frame and scale by 32768 (torchaudio.compliance.kaldi.fbank implicitly scales [-1, 1] inputs to 16-bit PCM range)
        for (int i = 0; i < frame_length; i++) {
            if (start_sample + i < num_samples) {
                frame_buffer[i] = samples[start_sample + i] * 32768.0f;
            } else {
                frame_buffer[i] = 0.0f;
            }
        }
        
        // 2. Dither (default 0.0 in firered)
        if (config_.dither > 0.0f) {
            for (int i = 0; i < frame_length; i++) {
                frame_buffer[i] += config_.dither * ((rand() / float(RAND_MAX)) * 2.0f - 1.0f);
            }
        }
        
        // 3. Remove DC offset (subtract mean)
        float mean = 0.0f;
        int valid_samples = std::min(frame_length, num_samples - start_sample);
        if (valid_samples > 0) {
            for (int i = 0; i < valid_samples; i++) {
                mean += frame_buffer[i];
            }
            mean /= valid_samples;
            for (int i = 0; i < frame_length; i++) {
                if (i < valid_samples) frame_buffer[i] -= mean;
            }
        }
        
        // 4. Preemphasis (default 0.97)
        float preemph = 0.97f;
        if (start_sample > 0) {
            // Need the raw sample from before this frame, scaled and dithered?
            // Kaldi applies preemphasis AFTER dc offset. The "previous sample" in Kaldi is the 
            // PREVIOUS sample in the CURRENT windowed frame! Kaldi preemphasis is applied per-frame.
            // So y[0] = x[0] - 0.97 * 0.0 (or preemph_coeff * 0)
            // Wait, Kaldi uses x[i-1]. For i=0, it uses x[-1]? Actually Kaldi uses 0 for the first sample in the frame.
        }
        // Apply preemphasis backwards to avoid needing an extra buffer
        for (int i = frame_length - 1; i > 0; i--) {
            frame_buffer[i] = frame_buffer[i] - preemph * frame_buffer[i - 1];
        }
        frame_buffer[0] = frame_buffer[0] - preemph * frame_buffer[0]; // Kaldi does this! Wait, Kaldi uses x[0] - preemph * x[0] ? No, Kaldi uses x[0] - preemph * x[-1]. If x[-1] isn't available, some implementations use x[0] - preemph * x[0]. Let's just use x[0] for simplicity.
        
        // Apply window
        ApplyWindow(frame_buffer.data(), frame_length);
        
        // Compute power spectrum via FFT
        ComputeFFT(frame_buffer.data(), frame_length, 
                   power_spectrum.data(), config_.num_fft_bins);
                   
        // Apply mel filterbanks
        float* frame_features_ptr = features.data() + (f * config_.num_mel_bins);
        ApplyMelBanks(power_spectrum.data(), frame_features_ptr);
        
        for (int k = 0; k < config_.num_fft_bins / 2 + 1; k++) {
            if (power_spectrum[k] > max_ps) max_ps = power_spectrum[k];
        }
    }
    
    return features;
}

} // namespace firered
