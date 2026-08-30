// firered_vad.cpp - firered VAD Implementation
// State-of-the-Art ML-based Voice Activity Detection

#include "firered-vad/firered_vad.h"
#include "firered_fbank.h"

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
    
    // Model dimensions (will be detected from tensors)
    int feature_dim = 80;      // Input features (Fbank)
    int hidden_dim = 128;      // Hidden state dimension
    int n_classes = 1;         // Output classes (1 for VAD, 3 for AED)
    
    // FSMN configuration
    int lookback_order = 20;   // Lookback filter size (past context)
    int lookahead_order = 20;  // Lookahead filter size (future context, 0 for streaming)
    
    // Feature extraction
    firered::FbankExtractor* fbank_extractor = nullptr;
    
    // Model tensors (will be loaded from GGUF)
    struct {
        // FireRed-VAD uses DFSMN (Deep Feed-Forward Sequential Memory Network)
        // NOT RNN! Architecture:
        // 1. FC1: (80 → 256)
        // 2. FC2: (256 → 128)
        // 3. FSMN1: Lookback/lookahead filters
        // 4. 7× FSMN blocks: Each with FC1(128→256), FC2(256→128), FSMN, residual
        // 5. DNN: (128 → 128)
        // 6. Output: (128 → n_classes)
        
        // Initial layers
        ggml_tensor* fc1_weight = nullptr;  // dfsmn.fc1.0.weight
        ggml_tensor* fc1_bias = nullptr;    // dfsmn.fc1.0.bias
        ggml_tensor* fc2_weight = nullptr;  // dfsmn.fc2.0.weight
        ggml_tensor* fc2_bias = nullptr;    // dfsmn.fc2.0.bias
        
        // FSMN1 filters
        ggml_tensor* fsmn1_lookback = nullptr;   // dfsmn.fsmn1.lookback_filter.weight
        ggml_tensor* fsmn1_lookahead = nullptr;  // dfsmn.fsmn1.lookahead_filter.weight (may not exist for streaming)
        
        // FSMN blocks (0-6)
        struct FSMNBlock {
            ggml_tensor* fc1_weight = nullptr;
            ggml_tensor* fc1_bias = nullptr;
            ggml_tensor* fc2_weight = nullptr;
            ggml_tensor* lookback_filter = nullptr;
            ggml_tensor* lookahead_filter = nullptr;  // May be nullptr for streaming
        } fsmn_blocks[7];
        
        // Final layers
        ggml_tensor* dnn_weight = nullptr;  // dfsmn.dnns.0.weight
        ggml_tensor* dnn_bias = nullptr;    // dfsmn.dnns.0.bias
        ggml_tensor* out_weight = nullptr;  // out.weight
        ggml_tensor* out_bias = nullptr;    // out.bias
        
        // CMVN statistics (for feature normalization)
        ggml_tensor* cmvn_mean = nullptr;      // Optional: from metadata
        ggml_tensor* cmvn_variance = nullptr;  // Optional: from metadata
    } model;
    
    // Configuration
    fireredVADConfig config;
    
    // Metadata from GGUF
    std::string model_version;
    std::string model_language;
    std::string architecture;
    
    // Global CMVN statistics (loaded from metadata)
    std::vector<float> cmvn_mean;
    std::vector<float> cmvn_std;
    
    // RAII destructor for safe memory management
    ~fireredVADContext() {
        if (fbank_extractor) {
            delete fbank_extractor;
            fbank_extractor = nullptr;
        }
        
        if (buffer_compute) {
            ggml_backend_buffer_free(buffer_compute);
            buffer_compute = nullptr;
        }
        
        if (buffer_model) {
            ggml_backend_buffer_free(buffer_model);
            buffer_model = nullptr;
        }
        
        if (ctx_compute) {
            ggml_free(ctx_compute);
            ctx_compute = nullptr;
        }
        
        if (ctx_model) {
            ggml_free(ctx_model);
            ctx_model = nullptr;
        }
        
        if (gguf_ctx) {
            gguf_free(gguf_ctx);
            gguf_ctx = nullptr;
        }
        
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        
        if (allocr) {
            ggml_gallocr_free(allocr);
            allocr = nullptr;
        }
    }
    
    // Delete copy semantics
    fireredVADContext(const fireredVADContext&) = delete;
    fireredVADContext& operator=(const fireredVADContext&) = delete;
    
    // Allow default construction
    fireredVADContext() = default;
    
    // Graph allocator for zero-allocation inference
    struct ggml_gallocr* allocr = nullptr;
};

// Helper: Detect mode from model filename
static fireredVADMode detect_mode_from_path(const char* model_path) {
    std::string path_str(model_path);
    std::string filename = path_str.substr(path_str.find_last_of("/\\") + 1);
    
    // Convert to lowercase for comparison
    std::transform(filename.begin(), filename.end(), filename.begin(), [](unsigned char c) -> char { return static_cast<char>(std::tolower(c)); });
    
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
    
    key_idx = gguf_find_key(ctx->gguf_ctx, "firered.cmvn_mean");
    if (key_idx >= 0) {
        int n_elements = static_cast<int>(gguf_get_arr_n(ctx->gguf_ctx, key_idx));
        const float* data = (const float*)gguf_get_arr_data(ctx->gguf_ctx, key_idx);
        ctx->cmvn_mean.assign(data, data + n_elements);
        std::cout << "[firered VAD] Loaded CMVN mean (" << n_elements << " dims)" << std::endl;
    }
    
    key_idx = gguf_find_key(ctx->gguf_ctx, "firered.cmvn_variance");
    if (key_idx >= 0) {
        int n_elements = static_cast<int>(gguf_get_arr_n(ctx->gguf_ctx, key_idx));
        const float* data = (const float*)gguf_get_arr_data(ctx->gguf_ctx, key_idx);
        ctx->cmvn_std.resize(n_elements);
        for (int i = 0; i < n_elements; i++) {
            ctx->cmvn_std[i] = std::sqrt(data[i] + 1e-8f); // Add epsilon, convert to std dev
        }
        std::cout << "[firered VAD] Loaded CMVN variance (" << n_elements << " dims)" << std::endl;
    }
    
    // DFSMN model - no RNN state needed (stateless)
    
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
        
        // Initial FC layers
        if (name_str == "dfsmn.fc1.0.weight") {
            ctx->model.fc1_weight = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.fc1.0.weight" << std::endl;
        } else if (name_str == "dfsmn.fc1.0.bias") {
            ctx->model.fc1_bias = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.fc1.0.bias" << std::endl;
        } else if (name_str == "dfsmn.fc2.0.weight") {
            ctx->model.fc2_weight = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.fc2.0.weight" << std::endl;
        } else if (name_str == "dfsmn.fc2.0.bias") {
            ctx->model.fc2_bias = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.fc2.0.bias" << std::endl;
        }
        // FSMN1 filters
        else if (name_str == "dfsmn.fsmn1.lookback_filter.weight") {
            ctx->model.fsmn1_lookback = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.fsmn1.lookback_filter.weight" << std::endl;
        } else if (name_str == "dfsmn.fsmn1.lookahead_filter.weight") {
            ctx->model.fsmn1_lookahead = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.fsmn1.lookahead_filter.weight" << std::endl;
        }
        // FSMN blocks (0-6)
        else if (name_str.find("dfsmn.fsmns.") != std::string::npos) {
            // Extract block index
            size_t pos = name_str.find("dfsmn.fsmns.");
            if (pos != std::string::npos) {
                int block_idx = name_str[pos + 12] - '0';  // Get digit after "dfsmn.fsmns."
                if (block_idx >= 0 && block_idx < 7) {
                    if (name_str.find(".fc1.0.weight") != std::string::npos) {
                        ctx->model.fsmn_blocks[block_idx].fc1_weight = tensor;
                        std::cout << "[firered VAD]   Loaded: dfsmn.fsmns." << block_idx << ".fc1.0.weight" << std::endl;
                    } else if (name_str.find(".fc1.0.bias") != std::string::npos) {
                        ctx->model.fsmn_blocks[block_idx].fc1_bias = tensor;
                        std::cout << "[firered VAD]   Loaded: dfsmn.fsmns." << block_idx << ".fc1.0.bias" << std::endl;
                    } else if (name_str.find(".fc2.weight") != std::string::npos) {
                        ctx->model.fsmn_blocks[block_idx].fc2_weight = tensor;
                        std::cout << "[firered VAD]   Loaded: dfsmn.fsmns." << block_idx << ".fc2.weight" << std::endl;
                    } else if (name_str.find(".fsmn.lookback_filter.weight") != std::string::npos) {
                        ctx->model.fsmn_blocks[block_idx].lookback_filter = tensor;
                        std::cout << "[firered VAD]   Loaded: dfsmn.fsmns." << block_idx << ".fsmn.lookback_filter.weight" << std::endl;
                    } else if (name_str.find(".fsmn.lookahead_filter.weight") != std::string::npos) {
                        ctx->model.fsmn_blocks[block_idx].lookahead_filter = tensor;
                        std::cout << "[firered VAD]   Loaded: dfsmn.fsmns." << block_idx << ".fsmn.lookahead_filter.weight" << std::endl;
                    }
                }
            }
        }
        // Final DNN and output layers
        else if (name_str == "dfsmn.dnns.0.weight") {
            ctx->model.dnn_weight = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.dnns.0.weight" << std::endl;
        } else if (name_str == "dfsmn.dnns.0.bias") {
            ctx->model.dnn_bias = tensor;
            std::cout << "[firered VAD]   Loaded: dfsmn.dnns.0.bias" << std::endl;
        } else if (name_str == "out.weight") {
            ctx->model.out_weight = tensor;
            std::cout << "[firered VAD]   Loaded: out.weight" << std::endl;
        } else if (name_str == "out.bias") {
            ctx->model.out_bias = tensor;
            std::cout << "[firered VAD]   Loaded: out.bias" << std::endl;
        } else {
            // Unknown tensor - just log it
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
    
    // Detect model dimensions from tensors
    if (ctx->model.fc1_weight) {
        ctx->feature_dim = static_cast<int>(ctx->model.fc1_weight->ne[0]);  // Input dimension (80)
        std::cout << "[firered VAD] Feature dimension: " << ctx->feature_dim << std::endl;
    }
    
    if (ctx->model.fc2_weight) {
        ctx->hidden_dim = static_cast<int>(ctx->model.fc2_weight->ne[1]);  // Output dimension of FC2 (128)
        std::cout << "[firered VAD] Hidden dimension: " << ctx->hidden_dim << std::endl;
    }
    
    if (ctx->model.out_weight) {
        ctx->n_classes = static_cast<int>(ctx->model.out_weight->ne[1]);  // Number of output classes (1)
        std::cout << "[firered VAD] Output classes: " << ctx->n_classes << std::endl;
    }
    
    if (ctx->model.fsmn1_lookback) {
        ctx->lookback_order = static_cast<int>(ctx->model.fsmn1_lookback->ne[0]);  // Filter size
        std::cout << "[firered VAD] Lookback order: " << ctx->lookback_order << std::endl;
    }
    
    if (ctx->model.fsmn1_lookahead) {
        ctx->lookahead_order = static_cast<int>(ctx->model.fsmn1_lookahead->ne[0]);  // Filter size
        std::cout << "[firered VAD] Lookahead order: " << ctx->lookahead_order << std::endl;
    } else {
        ctx->lookahead_order = 0;  // Streaming mode
        std::cout << "[firered VAD] No lookahead (streaming mode)" << std::endl;
    }
    
    // Free meta context (no longer needed)
    if (meta_ctx) {
        ggml_free(meta_ctx);
    }
    
    // Initialize graph allocator
    ctx->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    
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
    std::vector<float> result;
    
    if (input_sample_rate != target_sample_rate) {
        // Linear interpolation resampling
        double ratio = static_cast<double>(input_sample_rate) / target_sample_rate;
        int target_samples = static_cast<int>(n_samples / ratio);
        result.resize(target_samples);
        
        for (int i = 0; i < target_samples; i++) {
            double src_idx = i * ratio;
            int idx1 = static_cast<int>(src_idx);
            int idx2 = std::min(idx1 + 1, n_samples - 1);
            float frac = static_cast<float>(src_idx - idx1);
            
            result[i] = audio[idx1] * (1.0f - frac) + audio[idx2] * frac;
        }
    } else {
        // Copy audio directly
        result.assign(audio, audio + n_samples);
    }
    
    return result;
}

// Helper: Apply FSMN layer (Feed-Forward Sequential Memory Network)
// FSMN uses 1D grouped convolutions for temporal modeling
static struct ggml_tensor* apply_fsmn_layer(
    ggml_context* ctx_compute,
    struct ggml_tensor* input,  // Shape: (n_frames, hidden_dim)
    struct ggml_tensor* lookback_filter,   // Shape: (hidden_dim, 1, lookback_order)
    struct ggml_tensor* lookahead_filter   // Shape: (hidden_dim, 1, lookahead_order) or nullptr
) {
    int n_frames = static_cast<int>(input->ne[1]);
    int hidden_dim = static_cast<int>(input->ne[0]);
    
    // FSMN operation:
    // 1. Transpose input to (1, hidden_dim, n_frames) for conv1d
    // 2. Apply grouped conv1d with lookback filter (past context)
    // 3. Apply grouped conv1d with lookahead filter (future context, if exists)
    // 4. Add to input (residual connection)
    
    // Reshape for conv1d: (n_frames, hidden_dim) → (1, hidden_dim, n_frames)
    struct ggml_tensor* x_reshaped = ggml_cont(ctx_compute, ggml_permute(ctx_compute, input, 1, 0, 2, 3));
    x_reshaped = ggml_reshape_3d(ctx_compute, x_reshaped, n_frames, hidden_dim, 1);
    
    // Apply lookback convolution (grouped conv1d)
    int lookback_order = static_cast<int>(lookback_filter->ne[0]);
    
    // Pad left for lookback (past context) on ne[0]
    struct ggml_tensor* x_padded = ggml_pad_ext(ctx_compute, x_reshaped, lookback_order - 1, 0, 0, 0, 0, 0, 0, 0);
    
    // Grouped 1D convolution for lookback
    // GGML conv_1d_dw parameters: (ctx, kernel, input, stride, padding, dilation)
    struct ggml_tensor* lookback_out = ggml_conv_1d_dw(ctx_compute, lookback_filter, x_padded, 1, 0, 1);
    
    // Reshape back to (n_frames, hidden_dim)
    lookback_out = ggml_cont(ctx_compute, ggml_permute(ctx_compute, 
        ggml_reshape_2d(ctx_compute, lookback_out, n_frames, hidden_dim), 1, 0, 2, 3));
    
    // Apply lookahead convolution if exists (non-streaming mode)
    struct ggml_tensor* result = ggml_add(ctx_compute, input, lookback_out);
    
    if (lookahead_filter != nullptr) {
        int lookahead_order = static_cast<int>(lookahead_filter->ne[0]);
        
        // Pad right for lookahead (future context) on ne[0]
        struct ggml_tensor* x_padded_ahead = ggml_pad_ext(ctx_compute, x_reshaped, 0, lookahead_order - 1, 0, 0, 0, 0, 0, 0);
        
        // Grouped 1D convolution for lookahead (reversed filter)
        struct ggml_tensor* lookahead_out = ggml_conv_1d_dw(ctx_compute, lookahead_filter, x_padded_ahead, 1, 0, 1);
        
        // Reshape back
        lookahead_out = ggml_cont(ctx_compute, ggml_permute(ctx_compute,
            ggml_reshape_2d(ctx_compute, lookahead_out, n_frames, hidden_dim), 1, 0, 2, 3));
        
        // Add lookahead to result
        result = ggml_add(ctx_compute, result, lookahead_out);
    }
    
    return result;
}

// Helper: Apply one FSMN block
// Helper: Apply FSMN block (FC1 → FC2 → FSMN → Residual)
static struct ggml_tensor* apply_fsmn_block(
    ggml_context* ctx_compute,
    struct ggml_tensor* input,  // Shape: (n_frames, 128)
    const decltype(fireredVADContext::model)::FSMNBlock& block
) {
    // Save for residual connection
    struct ggml_tensor* residual = input;
    
    // FC1: (128 → 256) with ReLU
    struct ggml_tensor* block_x = input;
        
    block_x = ggml_mul_mat(ctx_compute, block.fc1_weight, block_x);
    block_x = ggml_add(ctx_compute, block_x, block.fc1_bias);
    block_x = ggml_relu(ctx_compute, block_x);
    
    block_x = ggml_mul_mat(ctx_compute, block.fc2_weight, block_x);
    
    block_x = apply_fsmn_layer(ctx_compute, block_x, block.lookback_filter, block.lookahead_filter);
    
    // Add residual connection
    struct ggml_tensor* x = ggml_add(ctx_compute, residual, block_x);
    
    return x;
}

static float run_firered_inference(
    fireredVADContext* ctx,
    const float* features,
    int n_frames,
    fireredAEDResult* aed_result = nullptr,
    float* out_probs = nullptr) {
    if (!ctx->ctx_model) {
        set_error("Model not loaded");
        return 0.0f;
    }
    
    // Allocate memory for compute graph metadata only (tensors allocated via gallocr on backend)
    size_t compute_size = ggml_tensor_overhead() * 2048;
    struct ggml_init_params params = {
        /* .mem_size = */ compute_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    struct ggml_context* ctx_compute = ggml_init(params);
    if (!ctx_compute) {
        set_error("Failed to initialize compute context");
        return 0.0f;
    }
    
    // ---- Build DFSMN forward pass compute graph ----
    
    // 1. Input tensor: (feature_dim, n_frames) in ggml column-major
    struct ggml_tensor* inp = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, ctx->feature_dim, n_frames);
    ggml_set_input(inp);
    ggml_set_name(inp, "input_features");
    
    struct ggml_tensor* x = inp;
    
    // 2. FC1: (80 → 256) with ReLU
    x = ggml_mul_mat(ctx_compute, ctx->model.fc1_weight, x);
    x = ggml_add(ctx_compute, x, ctx->model.fc1_bias);
    x = ggml_relu(ctx_compute, x);
    ggml_set_name(x, "fc1_out");
    
    // 3. FC2: (256 → 128) with ReLU
    x = ggml_mul_mat(ctx_compute, ctx->model.fc2_weight, x);
    x = ggml_add(ctx_compute, x, ctx->model.fc2_bias);
    x = ggml_relu(ctx_compute, x);
    ggml_set_name(x, "fc2_out");
    
    // 4. FSMN1: Initial memory layer
    x = apply_fsmn_layer(ctx_compute, x, ctx->model.fsmn1_lookback, ctx->model.fsmn1_lookahead);
    ggml_set_name(x, "fsmn1_out");
    
    // 5. FSMN blocks 0-6: Each with FC1, FC2, FSMN, residual
    for (int i = 0; i < 7; i++) {
        x = apply_fsmn_block(ctx_compute, x, ctx->model.fsmn_blocks[i]);
        ggml_set_name(x, ("fsmn_block_" + std::to_string(i)).c_str());
    }
    
    // 6. DNN layer: (128 → 128) with ReLU
    x = ggml_mul_mat(ctx_compute, ctx->model.dnn_weight, x);
    x = ggml_add(ctx_compute, x, ctx->model.dnn_bias);
    x = ggml_relu(ctx_compute, x);
    ggml_set_name(x, "dnn_out");
    
    // 7. Output layer: (128 → n_classes) with Sigmoid
    x = ggml_mul_mat(ctx_compute, ctx->model.out_weight, x);
    x = ggml_add(ctx_compute, x, ctx->model.out_bias);
    x = ggml_sigmoid(ctx_compute, x);
    ggml_set_name(x, "output");
    ggml_set_output(x);
    struct ggml_tensor* out = x;  // stable reference to output node
    
    // Build and execute the forward pass graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx_compute);
    ggml_build_forward_expand(gf, out);
    
    // Allocate tensors on the backend using gallocr
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        set_error("Failed to allocate graph buffers using ggml_gallocr");
        ggml_gallocr_free(allocr);
        ggml_free(ctx_compute);
        return 0.0f;
    }
    
    // Copy features and apply CMVN normalization directly into the allocated backend tensor
    if (!ctx->cmvn_mean.empty() && !ctx->cmvn_std.empty() && ctx->cmvn_mean.size() == (size_t)ctx->feature_dim) {
        std::vector<float> cmvn_buf(n_frames * ctx->feature_dim);
        for (int i = 0; i < n_frames; i++) {
            for (int j = 0; j < ctx->feature_dim; j++) {
                cmvn_buf[i * ctx->feature_dim + j] =
                    (features[i * ctx->feature_dim + j] - ctx->cmvn_mean[j]) / ctx->cmvn_std[j];
            }
        }
        ggml_backend_tensor_set(inp, cmvn_buf.data(), 0, cmvn_buf.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(inp, features, 0, n_frames * ctx->feature_dim * sizeof(float));
    }
    
    // Execute compute graph on backend
    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        set_error("Failed to compute DFSMN inference graph");
        ggml_gallocr_free(allocr);
        ggml_free(ctx_compute);
        return 0.0f;
    }
    
    // Read result based on mode
    // Output shape: (n_frames, n_classes)
    std::vector<float> output_data(n_frames * ctx->n_classes);
    ggml_backend_tensor_get(out, output_data.data(), 0, output_data.size() * sizeof(float));
    
    // Output mapping based on mode
    if (ctx->mode == fireredVADMode::AED && aed_result && ctx->n_classes == 3) {
        // AED mode: Average 3 probabilities (speech, music, singing) over all frames
        float speech_sum = 0.0f, music_sum = 0.0f, singing_sum = 0.0f;
        for (int i = 0; i < n_frames; i++) {
            speech_sum += output_data[i * 3 + 0];
            music_sum += output_data[i * 3 + 1];
            singing_sum += output_data[i * 3 + 2];
        }
        
        aed_result->speech_prob = speech_sum / n_frames;
        aed_result->music_prob = music_sum / n_frames;
        aed_result->singing_prob = singing_sum / n_frames;
        
        ggml_free(ctx_compute);
        return aed_result->speech_prob;
    } else {
        // Standard/Streaming mode
        float prob_sum = 0.0f;
        for (int i = 0; i < n_frames; i++) {
            prob_sum += output_data[i];
            if (out_probs) {
                out_probs[i] = output_data[i];
            }
        }
        
        ggml_free(ctx_compute);
        return prob_sum / n_frames;
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
    
    // Initialize Fbank feature extractor
    firered::FbankConfig fbank_config;
    fbank_config.sample_rate = ctx->config.sample_rate_hz;
    fbank_config.frame_length_ms = 25;  // Standard for speech
    fbank_config.frame_shift_ms = 10;   // Standard for speech
    fbank_config.num_mel_bins = ctx->feature_dim;  // 80 for FireRed-VAD
    fbank_config.num_fft_bins = 512;
    fbank_config.low_freq = 20.0f;
    fbank_config.high_freq = 8000.0f;
    
    ctx->fbank_extractor = new firered::FbankExtractor(fbank_config);
    
    std::cout << "[firered VAD] Initialized successfully" << std::endl;
    std::cout << "[firered VAD]   Sample rate: " << ctx->config.sample_rate_hz << " Hz" << std::endl;
    std::cout << "[firered VAD]   Frame size: " << ctx->config.frame_size_ms << " ms" << std::endl;
    std::cout << "[firered VAD]   Architecture: DFSMN (Deep Feed-Forward Sequential Memory Network)" << std::endl;
    std::cout << "[firered VAD]   Feature dim: " << ctx->feature_dim << std::endl;
    std::cout << "[firered VAD]   Hidden dim: " << ctx->hidden_dim << std::endl;
    std::cout << "[firered VAD]   Output classes: " << ctx->n_classes << std::endl;
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
    
    // Preprocess (resample if needed)
    auto processed_audio = preprocess_audio(audio_samples, n_samples, sample_rate_hz, ctx->config.sample_rate_hz);
    
    // Extract Fbank features
    auto features = ctx->fbank_extractor->ExtractFeatures(processed_audio.data(), static_cast<int>(processed_audio.size()));
    
    if (features.empty()) {
        std::cerr << "[firered VAD] WARNING: No features extracted (audio too short?)" << std::endl;
        return 0.0f;
    }
    
    int n_frames = static_cast<int>(features.size() / ctx->feature_dim);
    
    // Run DFSMN inference
    float prob = run_firered_inference(ctx, features.data(), n_frames);
    
    return prob;
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
    
    // Preprocess (resample if needed)
    auto processed_audio = preprocess_audio(audio_samples, n_samples, sample_rate_hz, ctx->config.sample_rate_hz);
    
    // Extract Fbank features
    auto features = ctx->fbank_extractor->ExtractFeatures(processed_audio.data(), static_cast<int>(processed_audio.size()));
    
    if (features.empty()) {
        frame_probabilities.clear();
        return true;
    }
    
    int n_frames = static_cast<int>(features.size() / ctx->feature_dim);
    
    // Resize frame probabilities to hold output
    frame_probabilities.resize(n_frames, 0.0f);
    
    // Run DFSMN inference (returns per-frame probabilities)
    run_firered_inference(ctx, features.data(), n_frames, nullptr, frame_probabilities.data());
    
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
    
    // DFSMN is stateless - no hidden state to reset
    // This function is kept for API compatibility but does nothing
    
    std::cout << "[firered VAD] Reset called (DFSMN is stateless, no operation needed)" << std::endl;
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
    
    // Read from GGUF KV store directly
    if (ctx->gguf_ctx) {
        int64_t idx = gguf_find_key(ctx->gguf_ctx, key);
        if (idx >= 0) {
            // Only return string-valued keys
            gguf_type ktype = gguf_get_kv_type(ctx->gguf_ctx, idx);
            if (ktype == GGUF_TYPE_STRING) {
                return gguf_get_val_str(ctx->gguf_ctx, idx);
            }
        }
    }
    
    // Fallback: pre-populated fields
    if (std::strcmp(key, "model.version") == 0 && !ctx->model_version.empty()) {
        return ctx->model_version.c_str();
    } else if (std::strcmp(key, "model.language") == 0 && !ctx->model_language.empty()) {
        return ctx->model_language.c_str();
    } else if (std::strcmp(key, "general.architecture") == 0 && !ctx->architecture.empty()) {
        return ctx->architecture.c_str();
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
    if (ctx) {
        delete ctx;
    }
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

