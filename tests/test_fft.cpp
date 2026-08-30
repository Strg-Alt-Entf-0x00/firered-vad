#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void slow_dft(const std::vector<float>& frame, int fft_bins, std::vector<float>& power_spectrum) {
    int num_bins = fft_bins / 2 + 1;
    for (int k = 0; k < num_bins; k++) {
        float real = 0.0f, imag = 0.0f;
        for (int n = 0; n < frame.size(); n++) {
            float angle = -2.0f * M_PI * k * n / fft_bins;
            real += frame[n] * std::cos(angle);
            imag += frame[n] * std::sin(angle);
        }
        power_spectrum[k] = real * real + imag * imag;
    }
}

void fast_fft(const std::vector<float>& frame, int fft_bins, std::vector<float>& power_spectrum) {
    std::vector<float> twiddle_real(fft_bins);
    std::vector<float> twiddle_imag(fft_bins);
    
    for (int size = 2; size <= fft_bins; size *= 2) {
        float tablestep = -2.0f * M_PI / size;
        for (int k = 0; k < size / 2; k++) {
            float angle = k * tablestep;
            int idx = size / 2 + k;
            if (idx < fft_bins) {
                twiddle_real[idx] = std::cos(angle);
                twiddle_imag[idx] = std::sin(angle);
            }
        }
    }
    
    std::vector<float> real(fft_bins, 0.0f);
    std::vector<float> imag(fft_bins, 0.0f);
    
    for (size_t i = 0; i < frame.size() && i < fft_bins; i++) {
        real[i] = frame[i];
    }
    
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
    
    for (int size = 2; size <= fft_bins; size *= 2) {
        int halfsize = size / 2;
        for (int i = 0; i < fft_bins; i += size) {
            for (int k = 0; k < halfsize; k++) {
                int twiddle_idx = halfsize + k;
                float twiddle_r = twiddle_real[twiddle_idx];
                float twiddle_i = twiddle_imag[twiddle_idx];
                
                int idx_k = i + k;
                int idx_k_half = idx_k + halfsize;
                
                float temp_real = real[idx_k_half] * twiddle_r - imag[idx_k_half] * twiddle_i;
                float temp_imag = real[idx_k_half] * twiddle_i + imag[idx_k_half] * twiddle_r;
                
                real[idx_k_half] = real[idx_k] - temp_real;
                imag[idx_k_half] = imag[idx_k] - temp_imag;
                real[idx_k] += temp_real;
                imag[idx_k] += temp_imag;
            }
        }
    }
    
    int num_bins = fft_bins / 2 + 1;
    for (int k = 0; k < num_bins; k++) {
        power_spectrum[k] = real[k] * real[k] + imag[k] * imag[k];
    }
}

int main() {
    int fft_bins = 512;
    std::vector<float> frame(400); // 25ms at 16kHz
    for (int i = 0; i < 400; i++) {
        frame[i] = std::sin(2.0f * M_PI * i * 440.0f / 16000.0f) * 32768.0f; // 440Hz sine wave
    }
    
    std::vector<float> ps_slow(512);
    std::vector<float> ps_fast(512);
    
    slow_dft(frame, fft_bins, ps_slow);
    fast_fft(frame, fft_bins, ps_fast);
    
    float max_diff = 0.0f;
    for (int i = 0; i < fft_bins / 2 + 1; i++) {
        float diff = std::abs(ps_slow[i] - ps_fast[i]);
        if (diff > max_diff) max_diff = diff;
        if (i < 10) {
            std::cout << "Bin " << i << ": Slow=" << ps_slow[i] << ", Fast=" << ps_fast[i] << "\n";
        }
    }
    std::cout << "Max diff: " << max_diff << "\n";
    return 0;
}
