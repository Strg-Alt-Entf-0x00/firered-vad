// firered_postprocess.cpp - FireRed VAD Post-Processing Implementation
// Ports the Python VadPostprocessor to C++

#include "firered-vad/firered_vad.h"
#include <algorithm>
#include <deque>
#include <iostream>

namespace firered {

enum class VadState {
    SILENCE = 0,
    POSSIBLE_SPEECH = 1,
    SPEECH = 2,
    POSSIBLE_SILENCE = 3
};

// Helper: Simple moving average smoothing
static std::vector<float> smooth_prob_simple(const std::vector<float>& probs, int smooth_window_size) {
    if (smooth_window_size <= 1 || probs.empty()) {
        return probs;
    }
    
    std::vector<float> smoothed_probs(probs.size(), 0.0f);
    std::deque<float> window;
    float window_sum = 0.0f;
    
    for (size_t i = 0; i < probs.size(); ++i) {
        window.push_back(probs[i]);
        window_sum += probs[i];
        
        if (static_cast<int>(window.size()) > smooth_window_size) {
            window_sum -= window.front();
            window.pop_front();
        }
        
        smoothed_probs[i] = window_sum / static_cast<float>(window.size());
    }
    
    return smoothed_probs;
}

// Helper: Apply threshold
static std::vector<int> apply_threshold(const std::vector<float>& probs, float threshold) {
    std::vector<int> binary_preds(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        binary_preds[i] = (probs[i] >= threshold) ? 1 : 0;
    }
    return binary_preds;
}

// Helper: State machine for VAD transitions
static std::vector<int> smooth_preds_with_state_machine(
    const std::vector<int>& binary_preds, 
    int min_speech_frame, 
    int min_silence_frame) 
{
    if (min_speech_frame <= 0 && min_silence_frame <= 0) {
        return binary_preds;
    }
    
    std::vector<int> decisions(binary_preds.size(), 0);
    VadState state = VadState::SILENCE;
    int speech_start = -1;
    int silence_start = -1;
    
    for (size_t t = 0; t < binary_preds.size(); ++t) {
        bool is_speech = (binary_preds[t] == 1);
        
        // State transition
        if (state == VadState::SILENCE) {
            if (is_speech) {
                state = VadState::POSSIBLE_SPEECH;
                speech_start = static_cast<int>(t);
            }
        } else if (state == VadState::POSSIBLE_SPEECH) {
            if (is_speech) {
                if (speech_start != -1 && static_cast<int>(t) - speech_start >= min_speech_frame) {
                    state = VadState::SPEECH;
                    // Retrospectively mark possible speech frames as actual speech
                    for (int i = speech_start; i <= static_cast<int>(t); ++i) {
                        decisions[i] = 1;
                    }
                }
            } else {
                state = VadState::SILENCE;
                speech_start = -1;
            }
        } else if (state == VadState::SPEECH) {
            if (!is_speech) {
                state = VadState::POSSIBLE_SILENCE;
                silence_start = static_cast<int>(t);
            }
        } else if (state == VadState::POSSIBLE_SILENCE) {
            if (!is_speech) {
                if (silence_start != -1 && static_cast<int>(t) - silence_start >= min_silence_frame) {
                    state = VadState::SILENCE;
                    speech_start = -1;
                }
            } else {
                state = VadState::SPEECH;
                silence_start = -1;
            }
        }
        
        // Current frame's decision
        if (state == VadState::SPEECH || state == VadState::POSSIBLE_SILENCE) {
            decisions[t] = 1;
        } else if (state == VadState::SILENCE || state == VadState::POSSIBLE_SPEECH) {
            decisions[t] = 0;
        }
    }
    
    return decisions;
}

// Helper: Fix smoothing delay
static std::vector<int> fix_smooth_window_start(const std::vector<int>& decisions, int smooth_window_size) {
    std::vector<int> new_decisions = decisions;
    for (size_t t = 1; t < decisions.size(); ++t) {
        if (decisions[t - 1] == 0 && decisions[t] == 1) {
            int start = std::max(0, static_cast<int>(t) - smooth_window_size);
            for (int i = start; i < static_cast<int>(t); ++i) {
                new_decisions[i] = 1;
            }
        }
    }
    return new_decisions;
}

// Helper: Merge short silence gaps
static std::vector<int> merge_short_silence_segments(const std::vector<int>& decisions, int merge_silence_frame) {
    if (merge_silence_frame <= 0) {
        return decisions;
    }
    
    std::vector<int> new_decisions = decisions;
    int silence_start = -1;
    
    for (size_t t = 1; t < decisions.size(); ++t) {
        if (decisions[t - 1] == 1 && decisions[t] == 0 && silence_start == -1) {
            silence_start = static_cast<int>(t);
        } else if (decisions[t - 1] == 0 && decisions[t] == 1 && silence_start != -1) {
            int silence_frame = static_cast<int>(t) - silence_start;
            if (silence_frame < merge_silence_frame) {
                for (int i = silence_start; i < static_cast<int>(t); ++i) {
                    new_decisions[i] = 1;
                }
            }
            silence_start = -1;
        }
    }
    return new_decisions;
}

// Helper: Extend speech segments slightly
static std::vector<int> extend_speech_segments_simple(const std::vector<int>& decisions, int extend_speech_frame) {
    if (extend_speech_frame <= 0) {
        return decisions;
    }
    
    std::vector<int> new_decisions = decisions;
    for (size_t t = 0; t < decisions.size(); ++t) {
        if (decisions[t] == 1) {
            int start = std::max(0, static_cast<int>(t) - extend_speech_frame);
            int end = std::min(static_cast<int>(decisions.size()), static_cast<int>(t) + extend_speech_frame + 1);
            for (int i = start; i < end; ++i) {
                new_decisions[i] = 1;
            }
        }
    }
    return new_decisions;
}

// Convert frame decisions to time segments (assuming 10ms hop size)
static std::vector<fireredVADSegment> decision_to_segment(
    const std::vector<int>& decisions, 
    int min_speech_frame, 
    float frame_shift_s = 0.010f) 
{
    std::vector<fireredVADSegment> segments;
    int speech_start = -1;
    
    for (size_t t = 0; t < decisions.size(); ++t) {
        if (decisions[t] == 1 && speech_start == -1) {
            speech_start = static_cast<int>(t);
        } else if (decisions[t] == 0 && speech_start != -1) {
            if (static_cast<int>(t) - speech_start < min_speech_frame) {
                // Unexpected short speech segment (should have been filtered by state machine)
            }
            segments.push_back({speech_start * frame_shift_s, static_cast<int>(t) * frame_shift_s});
            speech_start = -1;
        }
    }
    
    if (speech_start != -1) {
        int t = static_cast<int>(decisions.size()) - 1;
        if (t - speech_start < min_speech_frame) {
            // Unexpected short speech segment
        }
        float end_time = decisions.size() * frame_shift_s + 0.025f; // Add frame length
        segments.push_back({speech_start * frame_shift_s, end_time});
    }
    
    return segments;
}

// Main post-processing pipeline
std::vector<fireredVADSegment> firered_vad_postprocess(
    const std::vector<float>& probs,
    const fireredVADPostProcessConfig& config
) {
    if (probs.empty()) {
        return {};
    }
    
    // 1. Smooth probabilities
    auto smoothed_probs = smooth_prob_simple(probs, config.smooth_window_size);
    
    // 2. Apply threshold to get binary predictions
    auto binary_preds = apply_threshold(smoothed_probs, config.prob_threshold);
    
    // 3. Smooth binary predictions with state machine
    auto decisions = smooth_preds_with_state_machine(binary_preds, config.min_speech_frame, config.min_silence_frame);
    
    // 4. Fix smooth window start
    auto fixed_decisions = fix_smooth_window_start(decisions, config.smooth_window_size);
    
    // 5. Merge short silence gaps
    auto smoothed_decisions = merge_short_silence_segments(fixed_decisions, config.merge_silence_frame);
    
    // 6. Extend speech segments
    auto extend_decisions = extend_speech_segments_simple(smoothed_decisions, config.extend_speech_frame);
    
    // 7. Convert frame decisions to timestamps
    // Note: Python splits long segments (>300s). We skip that here as it's rarely needed 
    // for standard downstream tasks, but can be added if requested.
    auto segments = decision_to_segment(extend_decisions, config.min_speech_frame);
    
    return segments;
}

} // namespace firered
