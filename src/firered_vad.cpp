// firered_vad.cpp - firered VAD Implementation
// State-of-the-Art ML-based Voice Activity Detection

#include "firered-vad/firered_vad.h"

// Include GGML after our header to avoid conflicts
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

// Optional: CUDA support
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

namespace firered {

// Thread-local error storage
thread_local static std::string g_last_error;

// Internal context structure
struct fireredVADContext {
    // GGML context and model
    ggml_context* ctx_model = nullptr;
    ggml_context* ctx_compute = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer_model = nullptr;
    ggml_backend_buffer_t buffer_compute = nullptr;
    
    // GGUF context
    gguf_context* gguf_ctx = nullptr;
    
    // Operating mode
    fireredVADMode mode = fireredVADMode::Standard;
    
    // RNN State (firered is RNN-based, needs state memory)
    std::vector<float> rnn_state;
    int rnn_state_size = 128;  // Will be set from model
    
    // Model tensors (will be loaded from GGUF)
    struct {
        // firered RNN architecture:
        // - Input projection (audio features -> hidden)
        // - RNN layers (LSTM or GRU)
        // - Output projection (hidden -> probability)
        ggml_tensor* input_proj_weight = nullptr;
        ggml_tensor* input_proj_bias = nullptr;
        ggml_tensor* rnn_weight_ih = nullptr;  // Input-hidden weights
        ggml_tensor* rnn_weight_hh = nullptr;  // Hidden-hidden weights
        ggml_tensor* rnn_bias = nullptr;
        ggml_tensor* output_proj_weight = nullptr;
        ggml_tensor* output_proj_bias = nullptr;
    } model;
    
    // Configuration
    fireredVADConfig config;
    
    // Metadata from GGUF
    std::string model_version;
    std::string model_language;
    std::string architecture;
};

// Helper: Detect mode from model filename
static fireredVADMode detect_mode_from_path(const char* model_path) {
    std::string path_str(model_path);
    std::string filename = path_str.substr(path_str.find_last_of("/\\") + 1);
    
    // Convert to lowercase for comparison
    std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
    
    if (filename.find("stream") != std::string::npos) {
        std::cout << "[firered VAD] Auto-detected mode: Streaming" << std::endl;
        return fireredVADMode::Streaming;
    } else if (filename.find("aed") != std::string::npos) {
        std::cout << "[firered VAD] Auto-detected mode: Audio Event Detection (AED)" << std::endl;
        return fireredVADMode::AED;
    } else {
        std::cout << "[firered VAD] Auto-detected mode: Standard" << std::endl;
        return fireredVADMode::Standard;
    }
}

// Helper: Mode to string
static const char* mode_to_string(fireredVADMode mode) {
    switch (mode) {
        case fireredVADMode::Standard: return "Standard";
        case fireredVADMode::Streaming: return "Streaming";
        case fireredVADMode::AED: return "AED";
        default: return "Unknown";
    }
}

// Helper: Set error message
static void set_error(const std::string& msg) {
    g_last_error = msg;
    std::cerr << "[firered VAD Error] " << msg << std::endl;
}

// Helper: Load GGUF model (full implementation)
static bool load_gguf_model(fireredVADContext* ctx, const char* model_path) {
    std::cout << "[firered VAD] Loading GGUF model: " << model_path << std::endl;
    
    // Initialize GGUF context with metadata only (no_alloc=true)
    struct ggml_context* meta_ctx = nullptr;
    struct gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx = */ &meta_ctx
    };
    
    ctx->gguf_ctx = gguf_init_from_file(model_path, params);
    if (!ctx->gguf_ctx) {
        set_error(std::string("Failed to load GGUF file: ") + model_path);
        return false;
    }
    
    // Get basic info
    uint32_t version = gguf_get_version(ctx->gguf_ctx);
    int64_t n_tensors = gguf_get_n_tensors(ctx->gguf_ctx);
    int64_t n_kv = gguf_get_n_kv(ctx->gguf_ctx);
    
    std::cout << "[firered VAD] GGUF version: " << version << std::endl;
    std::cout << "[firered VAD] Tensors: " << n_tensors << std::endl;
    std::cout << "[firered VAD] KV pairs: " << n_kv << std::endl;
    
    // Read metadata (optional, for logging)
    int64_t key_idx = gguf_find_key(ctx->gguf_ctx, "model.version");
    if (key_idx >= 0) {
        ctx->model_version = gguf_get_val_str(ctx->gguf_ctx, key_idx);
        std::cout << "[firered VAD] Model version: " << ctx->model_version << std::endl;
    }
    
    key_idx = gguf_find_key(ctx->gguf_ctx, "model.language");
    if (key_idx >= 0) {
        ctx->model_language = gguf_get_val_str(ctx->gguf_ctx, key_idx);
        std::cout << "[firered VAD] Language: " << ctx->model_language << std::endl;
    }
    
    key_idx = gguf_find_key(ctx->gguf_ctx, "general.architecture");
    if (key_idx >= 0) {
        ctx->architecture = gguf_get_val_str(ctx->gguf_ctx, key_idx);
        std::cout << "[firered VAD] Architecture: " << ctx->architecture << std::endl;
    }
    
    // Read RNN state size if available
    key_idx = gguf_find_key(ctx->gguf_ctx, "firered.rnn_state_size");
    if (key_idx >= 0) {
        ctx->rnn_state_size = gguf_get_val_u32(ctx->gguf_ctx, key_idx);
        std::cout << "[firered VAD] RNN state size: " << ctx->rnn_state_size << std::endl;
    }
    
    // Calculate total memory needed for model
    size_t total_size = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* tname = gguf_get_tensor_name(ctx->gguf_ctx, i);
        ggml_tensor* tensor = ggml_get_tensor(meta_ctx, tname);
        if (tensor) {
            total_size += ggml_nbytes(tensor);
        }
    }
    
    std::cout << "[firered VAD] Total model size: " << (total_size / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Create GGML context for model tensors
    struct ggml_init_params model_params = {
        /* .mem_size = */ total_size + ggml_tensor_overhead() * n_tensors,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,  // Backend will allocate
    };
    
    ctx->ctx_model = ggml_init(model_params);
    if (!ctx->ctx_model) {
        set_error("Failed to initialize GGML model context");
        gguf_free(ctx->gguf_ctx);
        ctx->gguf_ctx = nullptr;
        return false;
    }
    
    // Load all tensors from GGUF into GGML context
    std::cout << "[firered VAD] Loading tensors..." << std::endl;
    
    // Open file for reading tensor data
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        set_error(std::string("Cannot reopen model file: ") + model_path);
        return false;
    }
    
    size_t data_offset = gguf_get_data_offset(ctx->gguf_ctx);
    
    // Load each tensor
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* tname = gguf_get_tensor_name(ctx->gguf_ctx, i);
        ggml_tensor* meta_tensor = ggml_get_tensor(meta_ctx, tname);
        
        if (!meta_tensor) {
            std::cerr << "[firered VAD] WARNING: Tensor " << tname << " not found in meta context" << std::endl;
            continue;
        }
        
        // Create tensor in model context
        int n_dims = ggml_n_dims(meta_tensor);
        int64_t ne[4] = {1, 1, 1, 1};
        for (int d = 0; d < n_dims; d++) {
            ne[d] = meta_tensor->ne[d];
        }
        
        ggml_tensor* tensor = ggml_new_tensor(ctx->ctx_model, meta_tensor->type, n_dims, ne);
        ggml_set_name(tensor, tname);
        
        // Map tensor name to model structure
        std::string name_str(tname);
        if (name_str.find("input_proj.weight") != std::string::npos || name_str.find("input_proj_weight") != std::string::npos) {
            ctx->model.input_proj_weight = tensor;
            std::cout << "[firered VAD]   Loaded: input_proj_weight" << std::endl;
        } else if (name_str.find("input_proj.bias") != std::string::npos || name_str.find("input_proj_bias") != std::string::npos) {
            ctx->model.input_proj_bias = tensor;
            std::cout << "[firered VAD]   Loaded: input_proj_bias" << std::endl;
        } else if (name_str.find("rnn.weight_ih") != std::string::npos || name_str.find("rnn_weight_ih") != std::string::npos) {
            ctx->model.rnn_weight_ih = tensor;
            std::cout << "[firered VAD]   Loaded: rnn_weight_ih" << std::endl;
        } else if (name_str.find("rnn.weight_hh") != std::string::npos || name_str.find("rnn_weight_hh") != std::string::npos) {
            ctx->model.rnn_weight_hh = tensor;
            std::cout << "[firered VAD]   Loaded: rnn_weight_hh" << std::endl;
        } else if (name_str.find("rnn.bias") != std::string::npos || name_str.find("rnn_bias") != std::string::npos) {
            ctx->model.rnn_bias = tensor;
            std::cout << "[firered VAD]   Loaded: rnn_bias" << std::endl;
        } else if (name_str.find("output_proj.weight") != std::string::npos || name_str.find("output_proj_weight") != std::string::npos) {
            ctx->model.output_proj_weight = tensor;
            std::cout << "[firered VAD]   Loaded: output_proj_weight" << std::endl;
        } else if (name_str.find("output_proj.bias") != std::string::npos || name_str.find("output_proj_bias") != std::string::npos) {
            ctx->model.output_proj_bias = tensor;
            std::cout << "[firered VAD]   Loaded: output_proj_bias" << std::endl;
        } else {
            std::cout << "[firered VAD]   Tensor: " << tname << std::endl;
        }
    }
    
    // Allocate backend buffer for model weights
    ctx->buffer_model = ggml_backend_alloc_ctx_tensors(ctx->ctx_model, ctx->backend);
    if (!ctx->buffer_model) {
        set_error("Failed to allocate backend buffer for model");
        return false;
    }
    
    // Copy tensor data from file to backend buffer
    std::cout << "[firered VAD] Copying tensor data to backend..." << std::endl;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* tname = gguf_get_tensor_name(ctx->gguf_ctx, i);
        ggml_tensor* tensor = ggml_get_tensor(ctx->ctx_model, tname);
        
        if (!tensor) continue;
        
        size_t tensor_offset = gguf_get_tensor_offset(ctx->gguf_ctx, i);
        size_t tensor_size = ggml_nbytes(tensor);
        
        // Seek to tensor data
        file.seekg(data_offset + tensor_offset);
        
        // Read tensor data into temporary buffer
        std::unique_ptr<char[]> temp_data(new char[tensor_size]);
        file.read(temp_data.get(), tensor_size);
        
        // Copy to backend
        ggml_backend_tensor_set(tensor, temp_data.get(), 0, tensor_size);
    }
    
    file.close();
    
    // Initialize RNN state
    ctx->rnn_state.resize(ctx->rnn_state_size, 0.0f);
    
    // Free meta context (no longer needed)
    if (meta_ctx) {
        ggml_free(meta_ctx);
    }
    
    std::cout << "[firered VAD] Model loaded successfully" << std::endl;
    return true;
}

// Helper: Simple audio preprocessing (normalize, resample if needed)
static std::vector<float> preprocess_audio(
    const float* audio,
    int n_samples,
    int input_sample_rate,
    int target_sample_rate
) {
    // TODO: Implement resampling if input_sample_rate != target_sample_rate
    // For now, assume audio is already 16kHz
    
    if (input_sample_rate != target_sample_rate) {
        std::cerr << "[firered VAD] WARNING: Resampling not implemented. "
                  << "Expected " << target_sample_rate << " Hz, got " << input_sample_rate << " Hz"
                  << std::endl;
    }
    
    // Copy and normalize audio
    std::vector<float> result(audio, audio + n_samples);
    
    // Find max amplitude
    float max_amp = 0.0f;
    for (float sample : result) {
        max_amp = std::max(max_amp, std::abs(sample));
    }
    
    // Normalize to [-1, 1] if needed
    if (max_amp > 1.0f) {
        for (float& sample : result) {
            sample /= max_amp;
        }
    }
    
    return result;
}

// Helper: Run RNN inference with GGML compute graph
// Returns speech probability for Standard/Streaming modes
// For AED mode, returns probabilities in a struct (passed separately)
static float run_firered_inference(
    fireredVADContext* ctx,
    const float* audio_frame,
    int frame_samples,
    fireredAEDResult* aed_result = nullptr
) {
    if (!ctx->ctx_model) {
        set_error("Model not loaded");
        return 0.0f;
    }
    
    // Create compute context if not exists
    if (!ctx->ctx_compute) {
        // Allocate enough memory for compute graph
        size_t compute_size = 16 * 1024 * 1024;  // 16 MB should be enough
        struct ggml_init_params params = {
            /* .mem_size = */ compute_size,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc = */ false,
        };
        ctx->ctx_compute = ggml_init(params);
        if (!ctx->ctx_compute) {
            set_error("Failed to initialize compute context");
            return 0.0f;
        }
    }
    
    // Build compute graph for forward pass
    // Note: This is a simplified RNN inference - actual firered architecture may differ
    // TODO: Update based on actual firered model architecture once we have the GGUF file
    
    // 1. Create input tensor from audio frame
    struct ggml_tensor* input = ggml_new_tensor_1d(ctx->ctx_compute, GGML_TYPE_F32, frame_samples);
    ggml_backend_tensor_set(input, audio_frame, 0, frame_samples * sizeof(float));
    ggml_set_name(input, "input");
    
    // 2. Input projection (if model has it)
    struct ggml_tensor* hidden = input;
    if (ctx->model.input_proj_weight) {
        hidden = ggml_mul_mat(ctx->ctx_compute, ctx->model.input_proj_weight, input);
        if (ctx->model.input_proj_bias) {
            hidden = ggml_add(ctx->ctx_compute, hidden, ctx->model.input_proj_bias);
        }
        ggml_set_name(hidden, "projected");
    }
    
    // 3. RNN layer (simplified GRU/LSTM)
    // For now, implement a basic RNN cell: h_new = tanh(W_ih * x + W_hh * h + b)
    if (ctx->model.rnn_weight_ih && ctx->model.rnn_weight_hh) {
        // Create RNN state tensor
        struct ggml_tensor* rnn_state = ggml_new_tensor_1d(ctx->ctx_compute, GGML_TYPE_F32, ctx->rnn_state_size);
        ggml_backend_tensor_set(rnn_state, ctx->rnn_state.data(), 0, ctx->rnn_state_size * sizeof(float));
        ggml_set_name(rnn_state, "rnn_state");
        
        // Compute: W_ih * input
        struct ggml_tensor* ih = ggml_mul_mat(ctx->ctx_compute, ctx->model.rnn_weight_ih, hidden);
        
        // Compute: W_hh * hidden_state
        struct ggml_tensor* hh = ggml_mul_mat(ctx->ctx_compute, ctx->model.rnn_weight_hh, rnn_state);
        
        // Add together
        struct ggml_tensor* combined = ggml_add(ctx->ctx_compute, ih, hh);
        
        // Add bias if exists
        if (ctx->model.rnn_bias) {
            combined = ggml_add(ctx->ctx_compute, combined, ctx->model.rnn_bias);
        }
        
        // Apply activation (tanh for basic RNN)
        hidden = ggml_tanh(ctx->ctx_compute, combined);
        ggml_set_name(hidden, "rnn_output");
        
        // Save new state (will be read back after compute)
    }
    
    // 4. Output projection
    struct ggml_tensor* output = hidden;
    if (ctx->model.output_proj_weight) {
        output = ggml_mul_mat(ctx->ctx_compute, ctx->model.output_proj_weight, hidden);
        if (ctx->model.output_proj_bias) {
            output = ggml_add(ctx->ctx_compute, output, ctx->model.output_proj_bias);
        }
        ggml_set_name(output, "output");
    }
    
    // 5. Apply sigmoid to get probability [0, 1]
    output = ggml_sigmoid(ctx->ctx_compute, output);
    ggml_set_name(output, "probability");
    
    // Build forward pass graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx->ctx_compute);
    ggml_build_forward_expand(gf, output);
    
    // Execute compute graph on backend
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        set_error("Failed to compute inference graph");
        return 0.0f;
    }
    
    // Read result based on mode
    if (ctx->mode == fireredVADMode::AED && aed_result) {
        // AED mode: Read 3 probabilities (speech, music, singing)
        float probs[3] = {0.0f, 0.0f, 0.0f};
        size_t output_elements = static_cast<size_t>(ggml_nelements(output));
        size_t output_size = std::min(size_t(3), output_elements) * sizeof(float);
        ggml_backend_tensor_get(output, probs, 0, output_size);
        
        aed_result->speech_prob = probs[0];
        aed_result->music_prob = probs[1];
        aed_result->singing_prob = probs[2];
        
        // Return speech probability as primary result
        return probs[0];
    } else {
        // Standard/Streaming mode: Single probability
        float result = 0.0f;
        ggml_backend_tensor_get(output, &result, 0, sizeof(float));
        return result;
    }
}

// ============================================================================
// Public API Implementation
// ============================================================================

fireredVADContext* firered_vad_init(const char* model_path, const fireredVADConfig* config) {
    if (!model_path) {
        set_error("Model path is null");
        return nullptr;
    }
    
    auto* ctx = new fireredVADContext();
    
    // Copy config or use defaults
    if (config) {
        ctx->config = *config;
    }
    
    // Auto-detect mode from filename if not explicitly set
    if (ctx->config.mode == fireredVADMode::Standard && config == nullptr) {
        ctx->config.mode = detect_mode_from_path(model_path);
    }
    ctx->mode = ctx->config.mode;
    
    std::cout << "[firered VAD] Operating mode: " << mode_to_string(ctx->mode) << std::endl;
    
    // Initialize GGML backend
    ggml_backend_load_all();
    
    if (ctx->config.use_gpu) {
        // Try CUDA backend first
#ifdef GGML_USE_CUDA
        ctx->backend = ggml_backend_cuda_init(0);  // Device 0
#endif
        if (!ctx->backend) {
            std::cerr << "[firered VAD] CUDA not available, falling back to CPU" << std::endl;
            ctx->backend = ggml_backend_cpu_init();
        }
    } else {
        ctx->backend = ggml_backend_cpu_init();
    }
    
    if (!ctx->backend) {
        set_error("Failed to initialize GGML backend");
        delete ctx;
        return nullptr;
    }
    
    // Load GGUF model
    if (!load_gguf_model(ctx, model_path)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    
    std::cout << "[firered VAD] Initialized successfully" << std::endl;
    std::cout << "[firered VAD]   Sample rate: " << ctx->config.sample_rate_hz << " Hz" << std::endl;
    std::cout << "[firered VAD]   Frame size: " << ctx->config.frame_size_ms << " ms (30ms for firered)" << std::endl;
    std::cout << "[firered VAD]   RNN state size: " << ctx->rnn_state_size << std::endl;
    std::cout << "[firered VAD]   Backend: " << (ctx->config.use_gpu ? "CUDA" : "CPU") << std::endl;
    
    return ctx;
}

float firered_vad_detect(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz
) {
    if (!ctx || !audio_samples || n_samples <= 0) {
        set_error("Invalid arguments");
        return -1.0f;
    }
    
    // Preprocess audio
    auto audio = preprocess_audio(audio_samples, n_samples, sample_rate_hz, ctx->config.sample_rate_hz);
    
    // Process entire audio buffer
    // For simplicity, we'll average probabilities across all frames
    
    int frame_size_samples = (ctx->config.frame_size_ms * ctx->config.sample_rate_hz) / 1000;
    int hop_size_samples = (ctx->config.hop_size_ms * ctx->config.sample_rate_hz) / 1000;
    
    if (frame_size_samples > static_cast<int>(audio.size())) {
        // Audio too short for even one frame
        return 0.0f;
    }
    
    float total_prob = 0.0f;
    int frame_count = 0;
    
    for (int i = 0; i + frame_size_samples <= static_cast<int>(audio.size()); i += hop_size_samples) {
        float prob = run_firered_inference(ctx, audio.data() + i, frame_size_samples);
        total_prob += prob;
        frame_count++;
    }
    
    if (frame_count == 0) {
        return 0.0f;
    }
    
    return total_prob / frame_count;
}

bool firered_vad_detect_frames(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz,
    std::vector<float>& frame_probabilities
) {
    if (!ctx || !audio_samples || n_samples <= 0) {
        set_error("Invalid arguments");
        return false;
    }
    
    // Preprocess audio
    auto audio = preprocess_audio(audio_samples, n_samples, sample_rate_hz, ctx->config.sample_rate_hz);
    
    int frame_size_samples = (ctx->config.frame_size_ms * ctx->config.sample_rate_hz) / 1000;
    int hop_size_samples = (ctx->config.hop_size_ms * ctx->config.sample_rate_hz) / 1000;
    
    // Calculate number of frames
    int n_frames = 0;
    for (int i = 0; i + frame_size_samples <= static_cast<int>(audio.size()); i += hop_size_samples) {
        n_frames++;
    }
    
    frame_probabilities.resize(n_frames);
    
    // Run inference per frame
    int frame_idx = 0;
    for (int i = 0; i + frame_size_samples <= static_cast<int>(audio.size()); i += hop_size_samples) {
        float prob = run_firered_inference(ctx, audio.data() + i, frame_size_samples);
        frame_probabilities[frame_idx++] = prob;
    }
    
    return true;
}

bool firered_vad_detect_frames_aligned(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz,
    int hop_size_ms,
    std::vector<float>& frame_probabilities
) {
    if (!ctx || !audio_samples || n_samples <= 0 || hop_size_ms <= 0) {
        set_error("Invalid arguments");
        return false;
    }
    
    // Preprocess audio
    auto audio = preprocess_audio(audio_samples, n_samples, sample_rate_hz, ctx->config.sample_rate_hz);
    
    // FireRed native frame size (30ms)
    const int native_frame_size_samples = (ctx->config.frame_size_ms * ctx->config.sample_rate_hz) / 1000;  // 480 @ 16kHz
    
    // User requested hop size (e.g., 10ms for Parakeet)
    const int hop_size_samples = (hop_size_ms * ctx->config.sample_rate_hz) / 1000;  // 160 @ 16kHz for 10ms
    
    if (hop_size_samples >= native_frame_size_samples) {
        // No alignment needed - hop is larger than or equal to native frame
        // Use standard detect_frames
        return firered_vad_detect_frames(ctx, audio_samples, n_samples, sample_rate_hz, frame_probabilities);
    }
    
    // Strategy: Sliding window with interpolation
    // We run FireRed inference at hop_size_ms intervals (e.g., every 10ms)
    // Each inference uses a 30ms window, but we advance by only 10ms
    // This gives us 3x more granular output aligned with ASR
    
    // Calculate number of aligned frames
    int n_aligned_frames = 0;
    for (int i = 0; i + native_frame_size_samples <= static_cast<int>(audio.size()); i += hop_size_samples) {
        n_aligned_frames++;
    }
    
    if (n_aligned_frames == 0) {
        // Audio too short
        frame_probabilities.clear();
        return true;
    }
    
    frame_probabilities.resize(n_aligned_frames);
    
    // Run inference with sliding window at hop intervals
    int frame_idx = 0;
    for (int i = 0; i + native_frame_size_samples <= static_cast<int>(audio.size()); i += hop_size_samples) {
        // Run FireRed on 30ms window starting at position i
        float prob = run_firered_inference(ctx, audio.data() + i, native_frame_size_samples);
        frame_probabilities[frame_idx++] = prob;
        
        if (frame_idx >= n_aligned_frames) {
            break;
        }
    }
    
    // Optional: Apply temporal smoothing to reduce jitter from overlapping windows
    // This uses a simple moving average with window size 3
    if (frame_probabilities.size() >= 3) {
        std::vector<float> smoothed(frame_probabilities.size());
        
        // First frame (no smoothing)
        smoothed[0] = frame_probabilities[0];
        
        // Middle frames (3-point moving average)
        for (size_t i = 1; i < frame_probabilities.size() - 1; i++) {
            smoothed[i] = (frame_probabilities[i - 1] + frame_probabilities[i] + frame_probabilities[i + 1]) / 3.0f;
        }
        
        // Last frame (no smoothing)
        smoothed[frame_probabilities.size() - 1] = frame_probabilities[frame_probabilities.size() - 1];
        
        frame_probabilities = std::move(smoothed);
    }
    
    std::cout << "[firered VAD] Aligned detection: " << n_aligned_frames 
              << " frames @ " << hop_size_ms << "ms hop (native 30ms window)" << std::endl;
    
    return true;
}

void firered_vad_reset(fireredVADContext* ctx) {
    if (!ctx) {
        return;
    }
    
    // Reset RNN state to zeros
    std::fill(ctx->rnn_state.begin(), ctx->rnn_state.end(), 0.0f);
    
    std::cout << "[firered VAD] RNN state reset" << std::endl;
}

std::vector<std::pair<int, int>> firered_vad_segment(
    const std::vector<float>& frame_probs,
    float onset_threshold,
    float offset_threshold,
    int min_speech_frames,
    int min_silence_frames
) {
    std::vector<std::pair<int, int>> segments;
    
    if (frame_probs.empty()) {
        return segments;
    }
    
    bool in_speech = false;
    int speech_start = 0;
    int silence_count = 0;
    
    for (size_t i = 0; i < frame_probs.size(); ++i) {
        float prob = frame_probs[i];
        
        if (!in_speech) {
            // Looking for speech onset
            if (prob >= onset_threshold) {
                speech_start = static_cast<int>(i);
                in_speech = true;
                silence_count = 0;
            }
        } else {
            // In speech, looking for offset
            if (prob < offset_threshold) {
                silence_count++;
                
                if (silence_count >= min_silence_frames) {
                    // End of speech segment
                    int speech_end = static_cast<int>(i) - silence_count;
                    int duration = speech_end - speech_start;
                    
                    if (duration >= min_speech_frames) {
                        segments.push_back({speech_start, speech_end});
                    }
                    
                    in_speech = false;
                    silence_count = 0;
                }
            } else {
                silence_count = 0;
            }
        }
    }
    
    // Handle final segment if still in speech
    if (in_speech) {
        int speech_end = static_cast<int>(frame_probs.size()) - silence_count;
        int duration = speech_end - speech_start;
        
        if (duration >= min_speech_frames) {
            segments.push_back({speech_start, speech_end});
        }
    }
    
    return segments;
}

const char* firered_vad_get_metadata(fireredVADContext* ctx, const char* key) {
    if (!ctx || !key) {
        return nullptr;
    }
    
    // TODO: Return actual metadata from GGUF
    if (std::strcmp(key, "model.version") == 0) {
        return "1.0";  // Placeholder
    } else if (std::strcmp(key, "model.language") == 0) {
        return "multilingual";  // Placeholder
    }
    
    return nullptr;
}

fireredVADMode firered_vad_get_mode(fireredVADContext* ctx) {
    if (!ctx) {
        return fireredVADMode::Standard;
    }
    return ctx->mode;
}

bool firered_aed_detect_frames(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz,
    std::vector<fireredAEDResult>& aed_results
) {
    if (!ctx || !audio_samples || n_samples <= 0) {
        set_error("Invalid arguments");
        return false;
    }
    
    if (ctx->mode != fireredVADMode::AED) {
        set_error("Not in AED mode. Load firered-aed-vad.gguf model for audio event detection");
        return false;
    }
    
    // Preprocess audio
    auto audio = preprocess_audio(audio_samples, n_samples, sample_rate_hz, ctx->config.sample_rate_hz);
    
    int frame_size_samples = (ctx->config.frame_size_ms * ctx->config.sample_rate_hz) / 1000;
    int hop_size_samples = (ctx->config.hop_size_ms * ctx->config.sample_rate_hz) / 1000;
    
    // Calculate number of frames
    int n_frames = 0;
    for (int i = 0; i + frame_size_samples <= static_cast<int>(audio.size()); i += hop_size_samples) {
        n_frames++;
    }
    
    aed_results.resize(n_frames);
    
    // Run inference per frame
    int frame_idx = 0;
    for (int i = 0; i + frame_size_samples <= static_cast<int>(audio.size()); i += hop_size_samples) {
        run_firered_inference(ctx, audio.data() + i, frame_size_samples, &aed_results[frame_idx++]);
    }
    
    return true;
}

fireredAEDResult firered_aed_detect(
    fireredVADContext* ctx,
    const float* audio_samples,
    int n_samples,
    int sample_rate_hz
) {
    fireredAEDResult error_result = {-1.0f, -1.0f, -1.0f};
    
    if (!ctx || !audio_samples || n_samples <= 0) {
        set_error("Invalid arguments");
        return error_result;
    }
    
    if (ctx->mode != fireredVADMode::AED) {
        set_error("Not in AED mode. Load firered-aed-vad.gguf model for audio event detection");
        return error_result;
    }
    
    // Get frame-level results
    std::vector<fireredAEDResult> frame_results;
    if (!firered_aed_detect_frames(ctx, audio_samples, n_samples, sample_rate_hz, frame_results)) {
        return error_result;
    }
    
    if (frame_results.empty()) {
        return {0.0f, 0.0f, 0.0f};
    }
    
    // Average across all frames
    fireredAEDResult avg_result = {0.0f, 0.0f, 0.0f};
    for (const auto& result : frame_results) {
        avg_result.speech_prob += result.speech_prob;
        avg_result.music_prob += result.music_prob;
        avg_result.singing_prob += result.singing_prob;
    }
    
    float n = static_cast<float>(frame_results.size());
    avg_result.speech_prob /= n;
    avg_result.music_prob /= n;
    avg_result.singing_prob /= n;
    
    return avg_result;
}

void firered_vad_free(fireredVADContext* ctx) {
    if (!ctx) {
        return;
    }
    
    if (ctx->buffer_compute) {
        ggml_backend_buffer_free(ctx->buffer_compute);
    }
    
    if (ctx->buffer_model) {
        ggml_backend_buffer_free(ctx->buffer_model);
    }
    
    if (ctx->ctx_compute) {
        ggml_free(ctx->ctx_compute);
    }
    
    if (ctx->ctx_model) {
        ggml_free(ctx->ctx_model);
    }
    
    if (ctx->gguf_ctx) {
        gguf_free(ctx->gguf_ctx);
    }
    
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    
    delete ctx;
    
    std::cout << "[firered VAD] Context freed" << std::endl;
}

const char* firered_vad_get_error() {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

// ============================================================================
// C++ Wrapper Implementation
// ============================================================================

fireredVAD::fireredVAD(const std::string& model_path, const fireredVADConfig& config)
    : ctx_(firered_vad_init(model_path.c_str(), &config))
    , config_(config)
{
    if (!ctx_) {
        throw std::runtime_error(std::string("Failed to load firered VAD model: ") + 
                                 (firered_vad_get_error() ? firered_vad_get_error() : "unknown error"));
    }
}

fireredVAD::~fireredVAD() {
    if (ctx_) {
        firered_vad_free(ctx_);
    }
}

fireredVAD::fireredVAD(fireredVAD&& other) noexcept 
    : ctx_(other.ctx_), config_(other.config_) 
{
    other.ctx_ = nullptr;
}

fireredVAD& fireredVAD::operator=(fireredVAD&& other) noexcept {
    if (this != &other) {
        if (ctx_) firered_vad_free(ctx_);
        ctx_ = other.ctx_;
        config_ = other.config_;
        other.ctx_ = nullptr;
    }
    return *this;
}

float fireredVAD::detect(const std::vector<float>& audio, int sample_rate_hz) {
    return firered_vad_detect(ctx_, audio.data(), static_cast<int>(audio.size()), sample_rate_hz);
}

std::vector<float> fireredVAD::detect_frames_aligned(const std::vector<float>& audio, int hop_size_ms, int sample_rate_hz) {
    std::vector<float> probs;
    if (!firered_vad_detect_frames_aligned(ctx_, audio.data(), static_cast<int>(audio.size()), sample_rate_hz, hop_size_ms, probs)) {
        return {};
    }
    return probs;
}

std::vector<float> fireredVAD::detect_frames(const std::vector<float>& audio, int sample_rate_hz) {
    std::vector<float> probs;
    firered_vad_detect_frames(ctx_, audio.data(), static_cast<int>(audio.size()), sample_rate_hz, probs);
    return probs;
}

std::vector<std::pair<int, int>> fireredVAD::segment(
    const std::vector<float>& audio,
    int sample_rate_hz,
    float onset_threshold,
    float offset_threshold
) {
    auto probs = detect_frames(audio, sample_rate_hz);
    return firered_vad_segment(probs, onset_threshold, offset_threshold, 10, 5);
}

bool fireredVAD::has_speech(const std::vector<float>& audio, int sample_rate_hz, float threshold) {
    float prob = detect(audio, sample_rate_hz);
    return prob >= threshold;
}

fireredAEDResult fireredVAD::detect_aed(const std::vector<float>& audio, int sample_rate_hz) {
    return firered_aed_detect(ctx_, audio.data(), static_cast<int>(audio.size()), sample_rate_hz);
}

std::vector<fireredAEDResult> fireredVAD::detect_aed_frames(const std::vector<float>& audio, int sample_rate_hz) {
    std::vector<fireredAEDResult> results;
    firered_aed_detect_frames(ctx_, audio.data(), static_cast<int>(audio.size()), sample_rate_hz, results);
    return results;
}

void fireredVAD::reset() {
    firered_vad_reset(ctx_);
}

fireredVADMode fireredVAD::mode() const {
    return firered_vad_get_mode(ctx_);
}

const fireredVADConfig& fireredVAD::config() const {
    return config_;
}

fireredVAD::operator bool() const {
    return ctx_ != nullptr;
}

} // namespace firered

