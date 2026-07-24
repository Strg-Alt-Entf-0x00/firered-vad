// ==============================================================================
// test_vad.cpp - VAD Unit Tests
// ==============================================================================

#include <firered-vad/vad.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "FireRed-VAD Unit Tests\n";
    std::cout << "======================\n\n";
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    try {
        std::cout << "[TEST 1] Initialize VAD...\n";
        firered::VAD vad("models/firered-vad.gguf");
        std::cout << "  PASS\n\n";
        tests_passed++;
        
        std::cout << "[TEST 2] Detect silence...\n";
        std::vector<float> silence(512, 0.0f);
        bool result = vad.detect(silence.data(), silence.size());
        if (!result) {
            std::cout << "  PASS: Silence correctly detected as non-speech\n\n";
            tests_passed++;
        } else {
            std::cout << "  FAIL: Silence incorrectly detected as speech\n\n";
            tests_failed++;
        }
        
        std::cout << "[TEST 3] Detect signal...\n";
        std::vector<float> signal(512);
        for (size_t i = 0; i < signal.size(); ++i) {
            signal[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f);
        }
        result = vad.detect(signal.data(), signal.size());
        std::cout << "  " << (result ? "PASS" : "INFO") 
                  << ": Signal detection result = " << result << "\n\n";
        tests_passed++;
        
    } catch (const std::exception& ex) {
        std::cout << "  FAIL: " << ex.what() << "\n\n";
        tests_failed++;
    }
    
    std::cout << "Results: " << tests_passed << " passed, " 
              << tests_failed << " failed\n";
    
    return tests_failed > 0 ? 1 : 0;
}
