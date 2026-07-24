// firered_vad.h - firered Voice Activity Detection
// State-of-the-Art ML-based VAD for ASR Preprocessing
// 
// Architecture: RNN-based (LSTM)
// Input: 16kHz mono audio (float32)
// Output: Speech probability per 30ms frame (0.0 = silence, 1.0 = speech)
//
// License: MIT (firered Models)
// Reference: https://github.com/snakers4/firered-models

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace firered {

/// @brief firered VAD context - opaque handle
struct fireredVADContext;

/// @brief FireRed-VAD operating mode
enum class fireredVADMode {
    Standard,   ///< Standard VAD with lookback+lookahead (highest accuracy)
    Streaming,  ///< Real-time streaming VAD (no lookahead, for live audio)
    AED         ///< Audio Event Detection (speech/music/singing classification)
};

/// @brief Audio Event Detection result (for AED mode)
struct fireredAEDResult {
    float speech_prob;    ///< Probability of speech (0.0 - 1.0)
    float music_prob;     ///< Probability of music (0.0 - 1.0)
    float singing_prob;   ///< Probability of singing (0.0 - 1.0)
    
    /// Get the dominant audio event type
    const char* dominant_event() const {
        if (speech_prob > music_prob && speech_prob > singing_prob) return "speech";
        if (music_prob > singing_prob) return "music";
        return "singing";
    }
};

/// @brief Configuration for firered VAD
struct fireredVADConfig {
    int sample_rate_hz = 16000;       ///< Input sample rate (16kHz required)
    int frame_size_ms = 30;           ///< Frame size in milliseconds (firered: 30ms, NOT 20ms)
    int hop_size_ms = 10;             ///< Hop size in milliseconds
    float threshold = 0.5f;           ///< Speech probability threshold
    int n_threads = 4;                ///< Number of threads for inference
    bool use_gpu = false;             ///< Use CUDA if available
    fireredVADMode mode = fireredVADMode::Standard;  ///< Operating mode (auto-detected from model if not set)
};

/// @brief Initialize firered VAD from GGUF model
/// 
/// Supported models:
/// - firered-vad.gguf: Standard VAD (lookback+lookahead, highest accuracy)
/// - firered-stream-vad.gguf: Streaming VAD (no lookahead, for live audio)
/// - firered-aed-vad.gguf: Audio Event Detection (speech/music/singing)
/// 
/// Mode is auto-detected from filename unless explicitly set in config.
/// 
/// @param model_path Path to GGUF model file
/// @param config Configuration (optional, uses defaults if nullptr)
/// @return Context handle or nullptr on failure
fireredVADContext* firered_vad_init(const char* model_path, const fireredVADConfig* config = nullptr);

/// @brief Detect speech in audio buffer (single probability)
/// @param ctx firered context
/// @param audio_samples PCM float32 mono audio
/// @param n_samples Number of samples
/// @param sample_rate_hz Sample rate (will resample if != 16kHz)
/// @return Speech probability (0.0 = silence, 1.0 = speech), or -1.0f on error
float firered_vad_detect(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz
);

/// @brief Detect speech frame-by-frame (for streaming/segmentation)
/// @param ctx firered context
/// @param audio_samples PCM float32 mono audio
/// @param n_samples Number of samples
/// @param sample_rate_hz Sample rate (will resample if != 16kHz)
/// @param frame_probabilities Output vector - one probability per frame
/// @return true on success, false on error
/// @note For AED mode, this returns speech probability only. Use firered_aed_detect_frames() for full AED results.
bool firered_vad_detect_frames(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz,
    std::vector<float>& frame_probabilities
);

/// @brief Detect speech with configurable frame stride (for alignment with ASR)
/// 
/// FireRed natively uses 30ms frames. This function allows custom stride
/// (e.g., 10ms) for perfect alignment with ASR models like Parakeet.
/// Uses sliding window with interpolation for intermediate frames.
/// 
/// @param ctx firered context
/// @param audio_samples PCM float32 mono audio
/// @param n_samples Number of samples
/// @param sample_rate_hz Sample rate (will resample if != 16kHz)
/// @param hop_size_ms Hop size in milliseconds (e.g., 10ms for Parakeet alignment)
/// @param frame_probabilities Output vector - one probability per hop
/// @return true on success, false on error
/// 
/// @note Quality: ~95-98% of native 30ms quality with 10ms stride
bool firered_vad_detect_frames_aligned(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz,
    int hop_size_ms,
    std::vector<float>& frame_probabilities
);

/// @brief Detect audio events frame-by-frame (AED mode only)
/// 
/// Returns speech/music/singing probabilities for each frame.
/// Only works with firered-aed-vad.gguf model.
/// 
/// @param ctx firered context (must be initialized with AED model)
/// @param audio_samples PCM float32 mono audio
/// @param n_samples Number of samples
/// @param sample_rate_hz Sample rate (will resample if != 16kHz)
/// @param aed_results Output vector - one AED result per frame
/// @return true on success, false on error (including if not in AED mode)
bool firered_aed_detect_frames(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz,
    std::vector<fireredAEDResult>& aed_results
);

/// @brief Detect audio events in entire buffer (AED mode only)
/// 
/// Returns average speech/music/singing probabilities across all frames.
/// Only works with firered-aed-vad.gguf model.
/// 
/// @param ctx firered context (must be initialized with AED model)
/// @param audio_samples PCM float32 mono audio
/// @param n_samples Number of samples
/// @param sample_rate_hz Sample rate (will resample if != 16kHz)
/// @return AED result with averaged probabilities, or {-1,-1,-1} on error
fireredAEDResult firered_aed_detect(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz
);

/// @brief Reset RNN state (for new audio stream)
/// 
/// Call this when starting a new audio stream to clear RNN memory.
/// Important for accurate VAD when processing multiple independent audio files.
/// 
/// @param ctx firered context
void firered_vad_reset(fireredVADContext* ctx);

/// @brief Apply speech/silence segmentation with hysteresis
/// @param frame_probs Frame-level probabilities from detect_frames()
/// @param onset_threshold Probability threshold to start speech segment (e.g., 0.5)
/// @param offset_threshold Probability threshold to end speech segment (e.g., 0.3)
/// @param min_speech_frames Minimum frames for valid speech segment (e.g., 10 = 300ms)
/// @param min_silence_frames Minimum frames for valid silence gap (e.g., 5 = 150ms)
/// @return Vector of speech segments as [start_frame, end_frame] pairs
std::vector<std::pair<int, int>> firered_vad_segment(
    const std::vector<float>& frame_probs,
    float onset_threshold = 0.5f,
    float offset_threshold = 0.3f,
    int min_speech_frames = 10,
    int min_silence_frames = 5
);

/// @brief Get model metadata (version, language, training info)
/// @param ctx firered context
/// @param key Metadata key (e.g., "model.version", "model.language")
/// @return Metadata value or nullptr if not found
const char* firered_vad_get_metadata(fireredVADContext* ctx, const char* key);

/// @brief Get current operating mode
/// @param ctx firered context
/// @return Current VAD mode
fireredVADMode firered_vad_get_mode(fireredVADContext* ctx);

/// @brief Free firered context and resources
/// @param ctx firered context to free
void firered_vad_free(fireredVADContext* ctx);

/// @brief Get last error message
/// @return Error string or nullptr if no error
const char* firered_vad_get_error();

// ============================================================================
// C++ Convenience Wrapper (RAII)
// ============================================================================

/// @brief RAII wrapper for firered VAD context
class fireredVAD {
public:
    /// @brief Constructor - load model from path
    explicit fireredVAD(const std::string& model_path, const fireredVADConfig& config = fireredVADConfig());

    /// @brief Destructor - free resources
    ~fireredVAD();

    // Non-copyable, movable
    fireredVAD(const fireredVAD&) = delete;
    fireredVAD& operator=(const fireredVAD&) = delete;
    fireredVAD(fireredVAD&& other) noexcept;
    fireredVAD& operator=(fireredVAD&& other) noexcept;

    /// @brief Detect speech probability
    float detect(const std::vector<float>& audio, int sample_rate_hz = 16000);

    /// @brief Detect frame-by-frame probabilities (native 30ms frames)
    std::vector<float> detect_frames(const std::vector<float>& audio, int sample_rate_hz = 16000);

    /// @brief Detect frame-by-frame with custom hop size for ASR alignment
    /// @param hop_size_ms Hop size in milliseconds (e.g., 10ms for Parakeet)
    /// @note Perfect for aligning with ASR models that use 10ms mel hop
    std::vector<float> detect_frames_aligned(const std::vector<float>& audio, int hop_size_ms, int sample_rate_hz = 16000);

    /// @brief Segment audio into speech/silence regions
    std::vector<std::pair<int, int>> segment(
        const std::vector<float>& audio,
        int sample_rate_hz = 16000,
        float onset_threshold = 0.5f,
        float offset_threshold = 0.3f
    );

    /// @brief Check if audio contains speech (simple threshold check)
    bool has_speech(const std::vector<float>& audio, int sample_rate_hz = 16000, float threshold = 0.5f);

    /// @brief Detect audio events (AED mode only)
    fireredAEDResult detect_aed(const std::vector<float>& audio, int sample_rate_hz = 16000);

    /// @brief Detect audio events frame-by-frame (AED mode only)
    std::vector<fireredAEDResult> detect_aed_frames(const std::vector<float>& audio, int sample_rate_hz = 16000);

    /// @brief Reset RNN state for new audio stream
    void reset();

    /// @brief Get current operating mode
    fireredVADMode mode() const;

    /// @brief Get configuration
    const fireredVADConfig& config() const;

    /// @brief Check if context is valid
    explicit operator bool() const;

private:
    fireredVADContext* ctx_;
    fireredVADConfig config_;
};

} // namespace firered

