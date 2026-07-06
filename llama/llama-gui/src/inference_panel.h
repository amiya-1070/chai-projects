#pragma once
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <deque>
#include <chrono>
#include <unordered_map>
#include "config_panel.h"
#include "telemetry.h"

struct InferenceStats {
    float tokens_per_sec = 0.0f;
    int   total_tokens   = 0;
    float elapsed_sec    = 0.0f;
    bool  running        = false;
};

struct TokenProb {
    std::string token;
    float prob;
};

struct ChatMessage {
    bool        is_user = true;
    std::string text;
    InferenceStats stats;
};

struct KLResult {
    bool valid = false;

    float mean_forward_kl = 0.f;
    float mean_reverse_kl = 0.f;

    std::vector<float> forward_kl_per_position;
    std::vector<float> reverse_kl_per_position;

    std::vector<TokenProb> top_base;
    std::vector<TokenProb> top_finetuned;

    int n_tokens = 0;
    std::string error;
};

class InferencePanel {
public:
    InferencePanel();
    ~InferencePanel();

    void set_telemetry(TelemetryCollector* t) { m_telemetry = t; }
    void render(const DashboardConfig& cfg);

private:
    // ---- Base model (llama-cli pty) ----------------------------------------
    bool spawn_base(const DashboardConfig& cfg);
    void kill_base();
    bool base_alive();
    void send_to_base(const std::string& prompt);
    void base_reader_func();

    pid_t m_base_pid    = -1;
    int   m_base_master = -1;   // single pty fd for stdin+stdout
    std::thread       m_base_reader;
    std::atomic<bool> m_base_reader_running{false};
    std::string       m_base_read_buf;
    bool              m_base_ready      = false;
    bool              m_base_generating = false;

    // ---- KL helper (pipe process, owns finetuned generation + KL) ----------
    bool spawn_kl_helper(const DashboardConfig& cfg);
    void kill_kl_helper();
    bool kl_helper_alive();
    void send_to_kl_helper(const std::string& prompt);
    void kl_reader_func();

    pid_t m_kl_pid       = -1;
    int   m_kl_stdin_fd  = -1;
    int   m_kl_stdout_fd = -1;
    std::thread       m_kl_reader;
    std::atomic<bool> m_kl_reader_running{false};
    std::string       m_kl_read_buf;
    bool              m_kl_ready      = false;
    bool              m_kl_generating = false;  // finetuned generating
    bool              m_kl_computing  = false;  // KL forward passes running

    // ---- Shared state ------------------------------------------------------
    mutable std::mutex m_mutex;

    std::vector<ChatMessage> m_base_messages;
    std::vector<ChatMessage> m_ft_messages;
    std::string              m_base_streaming;
    std::string              m_ft_streaming;
    std::string              m_current_prompt;

    KLResult m_latest_kl;

    InferenceStats m_base_stats;
    InferenceStats m_ft_stats;
    int m_base_session_tokens = 0;
    int m_ft_session_tokens   = 0;

    // Token timing for live tps
    using clk = std::chrono::steady_clock;
    struct TokenStamp { clk::time_point t; };
    std::deque<TokenStamp> m_base_stamps;
    std::deque<TokenStamp> m_ft_stamps;
    static constexpr int TOKEN_WINDOW = 20;

    // Telemetry
    TelemetryCollector* m_telemetry = nullptr;
    std::vector<float>  m_infer_temp;
    std::vector<float>  m_infer_power;
    std::vector<float>  m_infer_freq;
    std::vector<float>  m_infer_tps_base;
    std::vector<float>  m_infer_tps_ft;
    std::vector<float>  m_infer_times;
    clk::time_point     m_session_start;

    bool m_processes_spawned = false;
    int  m_n_predict         = 512;

    // UI state
    char m_input_buf[2048] = "";
    bool m_scroll_base     = false;
    bool m_scroll_ft       = false;
    bool m_focus_input     = true;

    // UI sub-renderers
    void render_controls(const DashboardConfig& cfg);
    void render_chat_columns(const DashboardConfig& cfg);
    void render_input_bar(const DashboardConfig& cfg);
    void render_kl_panel();
    void render_telemetry_strip();
};