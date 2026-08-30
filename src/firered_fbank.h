// firered_fbank.h - Fbank Feature Extraction for FireRed-VAD
// Minimal Mel-filterbank implementation for VAD feature extraction

#ifndef FIRERED_FBANK_H
#define FIRERED_FBANK_H

#include <vector>
#include <cmath>
#include <cstring>

namespace firered {

// Fbank configuration
struct FbankConfig {
    int sample_rate = 16000;      // Sample rate (Hz)
    int frame_length_ms = 25;     // Frame length (ms)
    int frame_shift_ms = 10;      // Frame shift (ms)
    int num_mel_bins = 80;        // Number of mel filterbank bins
    int num_fft_bins = 512;       // FFT size (power of 2)
    float energy_floor = 1.1920929e-07f; // Match torchaudio's float32 epsilon
    float low_freq = 20.0f;       // Low frequency (Hz)
    float high_freq = 8000.0f;    // High frequency (Hz)
    bool use_energy = false;      // Use energy instead of log energy
    float dither = 0.0f;          // Dithering constant
};

// Fbank feature extractor
class FbankExtractor {
public:
    explicit FbankExtractor(const FbankConfig& config = FbankConfig());
    ~FbankExtractor();
    
    // Extract fbank features from audio samples
    // Input: audio samples (float array)
    // Output: features as a flat array (size: n_frames * num_mel_bins, row-major)
    std::vector<float> ExtractFeatures(
        const float* samples,
        int num_samples
    );
    
    // Get number of frames for given audio length
    int GetNumFrames(int num_samples) const;
    
private:
    FbankConfig config_;
    std::vector<std::vector<float>> mel_banks_;  // Mel filterbank matrix
    std::vector<float> window_;                   // Hamming window
    
    // Precomputed FFT Twiddle Factors
    std::vector<float> twiddle_real_;
    std::vector<float> twiddle_imag_;
    
    // Initialize mel filterbanks
    void InitMelBanks();
    
    // Initialize window function
    void InitWindow();
    
    // Convert Hz to Mel scale
    static float HzToMel(float hz) noexcept;
    
    // Convert Mel scale to Hz
    static float MelToHz(float mel) noexcept;
    
    // Apply window to frame
    void ApplyWindow(float* frame, int frame_length);
    
    // Compute FFT (simple DFT for now, can be optimized with FFT library)
    void ComputeFFT(const float* frame, int frame_length, 
                    float* power_spectrum, int fft_bins);
    
    // Apply mel filterbanks
    void ApplyMelBanks(const float* power_spectrum, float* mel_features);
};

} // namespace firered

#endif // FIRERED_FBANK_H
