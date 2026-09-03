// ============================================================================
// FireRed-VAD: Live Microphone Test (WASAPI)
// Tests the GGUF model in real-time streaming mode using the default Windows mic.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <comdef.h>
#pragma comment(lib, "ole32.lib")

#include "firered-vad/firered_vad.h"

namespace {

struct ComGuard {
    HRESULT hr;
    ComGuard() : hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComGuard() { if (SUCCEEDED(hr)) CoUninitialize(); }
};

std::vector<float> resample16k(const float* src, int n, int srcRate) {
    if (srcRate == 16000) return std::vector<float>(src, src + n);
    int n_out = static_cast<int>(static_cast<double>(n) * 16000.0 / srcRate);
    std::vector<float> out(n_out);
    for (int i = 0; i < n_out; ++i) {
        double pos = static_cast<double>(i) * srcRate / 16000.0;
        int i0 = static_cast<int>(pos);
        int i1 = (i0 + 1 < n ? i0 + 1 : n - 1);
        double frac = pos - i0;
        out[i] = static_cast<float>(src[i0] * (1.0 - frac) + src[i1] * frac);
    }
    return out;
}

void printBar(float prob, float rms) {
    if (std::isnan(prob)) prob = 0.0f;
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

    // Noise Gate
    if (rms < 0.002f) {
        prob = 0.0f;
    }

    const int W = 40;
    int filled = static_cast<int>(prob * W);
    if (filled < 0) filled = 0;
    if (filled > W) filled = W;

    std::string bar(filled, '#');
    bar += std::string(W - filled, '.');

    const char* label = (prob >= 0.5f) ? "SPEECH" : (prob >= 0.3f) ? "maybe " : "silent";
    const char* col   = (prob >= 0.5f) ? "\033[92m" : (prob >= 0.3f) ? "\033[93m" : "\033[91m";
    const char* rst   = "\033[0m";

    std::printf("\r  %s[%s]%s  %.2f  [%s%s%s]  Vol: %.4f   ",
        col, bar.c_str(), rst, prob, col, label, rst, rms);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: live_mic_test <path_to_gguf_model>\n";
        return 1;
    }
    std::string model_path = argv[1];

    firered::fireredVADContext* ctx = firered::firered_vad_init(model_path.c_str(), nullptr);
    if (!ctx) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }

    ComGuard com;
    if (FAILED(com.hr)) {
        std::cerr << "CoInitialize failed\n";
        return 1;
    }

    IMMDeviceEnumerator* enumerator = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                __uuidof(IMMDeviceEnumerator), (void**)&enumerator))) {
        std::cerr << "Failed to create enumerator\n";
        return 1;
    }

    IMMDevice* device = nullptr;
    if (FAILED(enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &device))) {
        std::cerr << "Failed to get default mic\n";
        return 1;
    }
    enumerator->Release();

    IAudioClient* client = nullptr;
    if (FAILED(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&client))) {
        std::cerr << "Failed to activate audio client\n";
        return 1;
    }
    device->Release();

    WAVEFORMATEX* fmt = nullptr;
    client->GetMixFormat(&fmt);

    REFERENCE_TIME bufDur = 2000000; // 200ms
    client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, bufDur, 0, fmt, nullptr);

    IAudioCaptureClient* capture = nullptr;
    client->GetService(__uuidof(IAudioCaptureClient), (void**)&capture);
    client->Start();

    int sampleRate = static_cast<int>(fmt->nSamplesPerSec);
    int channels   = static_cast<int>(fmt->nChannels);
    bool isFloat   = (fmt->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) ||
                     (fmt->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
                      reinterpret_cast<WAVEFORMATEXTENSIBLE*>(fmt)->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT);

    std::cout << "\n  ============================================\n";
    std::cout << "  FireRed GGUF - LIVE MICROPHONE TEST (15 sec)\n";
    std::cout << "  ============================================\n";
    std::cout << "  Speak into your microphone now!\n\n";

    auto t_start = std::chrono::steady_clock::now();
    std::vector<float> accum;

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - t_start;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= 15)
            break;

        BYTE* data = nullptr;
        UINT32 frames = 0;
        DWORD flags = 0;
        HRESULT hr = capture->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
        if (hr == AUDCLNT_S_BUFFER_EMPTY || frames == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
            for (UINT32 i = 0; i < frames; ++i) {
                float sample = 0.0f;
                if (isFloat) {
                    const float* f = reinterpret_cast<const float*>(data);
                    for (int c = 0; c < channels; ++c) sample += f[i * channels + c];
                } else {
                    const int16_t* s = reinterpret_cast<const int16_t*>(data);
                    for (int c = 0; c < channels; ++c) sample += s[i * channels + c] / 32768.0f;
                }
                accum.push_back(sample / channels);
            }
        }
        capture->ReleaseBuffer(frames);

        if (static_cast<int>(accum.size()) >= sampleRate * 200 / 1000) {
            int native_chunk = sampleRate * 200 / 1000;
            std::vector<float> mono16k = resample16k(accum.data(), native_chunk, sampleRate);
            accum.erase(accum.begin(), accum.begin() + native_chunk);

            float sum_sq = 0.0f;
            for (float s : mono16k) sum_sq += s * s;
            float rms = std::sqrt(sum_sq / mono16k.size());

            float prob = firered::firered_vad_detect(ctx, mono16k.data(), static_cast<int>(mono16k.size()), 16000);
            printBar(prob, rms);
        }
    }

    std::cout << "\n\n  Done!\n\n";

    client->Stop();
    capture->Release();
    client->Release();
    CoTaskMemFree(fmt);
    firered::firered_vad_free(ctx);

    return 0;
}
#else
int main() {
    std::cout << "Live Mic Test is Windows-only (WASAPI) in this example.\n";
    return 0;
}
#endif
