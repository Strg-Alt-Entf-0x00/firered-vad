#include "firered-vad/firered_vad.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }
    
    // Load model
    auto ctx = firered::firered_vad_init(argv[1], nullptr);
    if (!ctx) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    // Generate 1 second white noise
    srand(time(nullptr));
    const int n_samples = 16000;
    std::vector<float> noise(n_samples);
    for (int i = 0; i < n_samples; i++) {
        noise[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }
    
    // Run inference
    float prob = firered::firered_vad_detect(ctx, noise.data(), n_samples, 16000);
    
    std::cout << "Noise probability: " << prob << "\n";
    std::cout << "Expected: ~0.0009-0.0014 (PyTorch reference)\n";
    
    firered::firered_vad_free(ctx);
    return 0;
}
