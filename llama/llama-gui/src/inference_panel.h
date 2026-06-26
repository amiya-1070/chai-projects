#pragma once
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <deque>
#include <chrono>
#include "config_panel.h"
#include "telemetry.h"

struct InferenceStats {
    float   tokens_per_sec     = 0.0f;
    int     total_tokens       = 0;
    float   elapsed_sec        = 0.0f;
    float   avg_latency_ms     = 0.0f;  // ms per token
    int     context_used       = 0;
    bool    running            = false;
};

struct ChatMessage {
    bool        is_user  = true;
    std::string text;
    InferenceStats stats; // only meaningful for assistant messages
};

class InferencePanel {
public:
    InferencePanel();
    ~InferencePanel();

    void set_telemetry(TelemetryCollector* t) { m_telemetry = t; }

    void render(const DashboardConfig& cfg);

private:
    void start_inference(const DashboardConfig& cfg,
                         const std::string& prompt);
    void stop_inference();
    void infer_thread_func(InferParams params, std::string prompt);

    void render_chat_history();
    void render_input_bar(const DashboardConfig& cfg);
    void render_stats_overlay();
    void render_telemetry_strip();

    // Parse llama-cli output:
    // llama-cli streams tokens directly to stdout, one at a time,
    // with timing info at the end in the format:
    // "llama_print_timings: eval time = ... ms / N tokens"
    void parse_output_line(const std::string& line);
    bool is_timing_line(const std::string& line);

    // State
    std::thread         m_thread;
    std::atomic<bool>   m_running{false};
    std::atomic<bool>   m_cancel{false};

    mutable std::mutex  m_mutex;

    std::vector<ChatMessage> m_messages;
    std::string              m_streaming_text; // current assistant response
    InferenceStats           m_live_stats;

    // Token timing ring buffer for live t/s calculation
    struct TokenStamp {
        std::chrono::steady_clock::time_point t;
    };
    std::deque<TokenStamp>   m_token_stamps;
    static constexpr int     TOKEN_WINDOW = 20; // rolling window size

    int                      m_total_tokens_session = 0;
    std::chrono::steady_clock::time_point m_infer_start;

    // Telemetry snapshots taken during inference
    std::vector<float>       m_infer_temp;
    std::vector<float>       m_infer_power;
    std::vector<float>       m_infer_freq;
    std::vector<float>       m_infer_tps;   // live tps over time
    std::vector<float>       m_infer_times; // timestamps

    TelemetryCollector*      m_telemetry = nullptr;

    // UI state
    char                     m_input_buf[2048] = "";
    bool                     m_scroll_to_bottom = false;
    bool                     m_focus_input      = true;
    int                      m_context_size     = 2048;
};