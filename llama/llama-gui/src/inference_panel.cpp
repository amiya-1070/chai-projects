#include "inference_panel.h"
#include "imgui.h"
#include "implot.h"
#include "process_utils.h"
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdio>
#include <regex>

using clock_t2 = std::chrono::steady_clock;

static double elapsed_sec(clock_t2::time_point start) {
    return std::chrono::duration<double>(
        clock_t2::now() - start).count();
}

// ---- InferencePanel --------------------------------------------------------

InferencePanel::InferencePanel() {}

InferencePanel::~InferencePanel() {
    stop_inference();
}

void InferencePanel::start_inference(const DashboardConfig& cfg,
                                     const std::string& prompt) {
    if (m_running) return;

    m_cancel = false;
    m_streaming_text.clear();
    m_token_stamps.clear();
    m_infer_temp.clear();
    m_infer_power.clear();
    m_infer_freq.clear();
    m_infer_tps.clear();
    m_infer_times.clear();
    m_live_stats = {};
    m_live_stats.running = true;
    m_infer_start = clock_t2::now();

    InferParams params  = cfg.to_infer_params();
    params.prompt       = prompt;
    params.n_predict    = cfg.n_predict;
    m_context_size      = cfg.n_predict;

    fprintf(stderr, "Starting inference thread, prompt: %s\n", prompt.c_str());
    m_running = true;
    m_thread  = std::thread(
        &InferencePanel::infer_thread_func, this, params, prompt);
}

void InferencePanel::stop_inference() {
    m_cancel  = true;
    m_running = false;
    if (m_thread.joinable()) m_thread.join();
}

void InferencePanel::infer_thread_func(InferParams params,std::string prompt) {

    fprintf(stderr, "Infer thread started\n");

    std::string cmd = build_infer_command(params);

    fprintf(stderr, "INFER CMD: %s\n", cmd.c_str());

    stream_command_chars(cmd, [this](const std::string& chunk) {
        parse_output_line(chunk);

        // Sample telemetry every ~10 tokens
        if (m_live_stats.total_tokens % 10 == 0 && m_telemetry) {
            auto s = m_telemetry->latest();
            float t_now = (float)elapsed_sec(m_infer_start);
            m_infer_temp.push_back(s.pkg_temp_c);
            m_infer_power.push_back(s.pkg_power_w);
            if (!s.cpu_freq_mhz.empty()) {
                float avg = std::accumulate(
                    s.cpu_freq_mhz.begin(),
                    s.cpu_freq_mhz.end(), 0.0f)
                    / s.cpu_freq_mhz.size();
                m_infer_freq.push_back(avg);
            }
            m_infer_tps.push_back(m_live_stats.tokens_per_sec);
            m_infer_times.push_back(t_now);
        }
    }, m_cancel);

    // Finalize
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        float elapsed = (float)elapsed_sec(m_infer_start);
        m_live_stats.elapsed_sec = elapsed;
        m_live_stats.running     = false;

        if (m_live_stats.total_tokens > 0)
            m_live_stats.avg_latency_ms =
                (elapsed / m_live_stats.total_tokens) * 1000.0f;

        // Finalize the streaming message
        ChatMessage msg;
        msg.is_user = false;
        msg.text    = m_streaming_text;
        msg.stats   = m_live_stats;
        m_messages.push_back(msg);
        m_streaming_text.clear();
        m_scroll_to_bottom = true;
        m_total_tokens_session += m_live_stats.total_tokens;
    }

    fprintf(stderr, "Infer thread finished, total tokens: %d\n",m_live_stats.total_tokens);
    m_running = false;
}

void InferencePanel::parse_output_line(const std::string& chunk) {
    // filter out llama diagnostic lines
    if (chunk.find("llama_") != std::string::npos) return;
    if (chunk.find("ggml_")  != std::string::npos) return;
    if (chunk.find("main:")  != std::string::npos) return;
    if (chunk.find("Log ")   != std::string::npos) return;

    std::lock_guard<std::mutex> lk(m_mutex);
    m_streaming_text += chunk;
    m_scroll_to_bottom = true;

    // approximate token count by spaces
    for (char c : chunk)
        if (c == ' ' || c == '\n')
            m_live_stats.total_tokens++;

    // rolling tps
    m_token_stamps.push_back({clock_t2::now()});
    if ((int)m_token_stamps.size() > TOKEN_WINDOW)
        m_token_stamps.pop_front();

    if (m_token_stamps.size() >= 2) {
        double window_sec = std::chrono::duration<double>(
            m_token_stamps.back().t -
            m_token_stamps.front().t).count();
        if (window_sec > 0.0)
            m_live_stats.tokens_per_sec =
                (float)(m_token_stamps.size() - 1) / window_sec;
    }
}

bool InferencePanel::is_timing_line(const std::string& line) {
    return line.find("llama_print_timings") != std::string::npos
        && line.find("eval time") != std::string::npos;
}

// ---- Render ----------------------------------------------------------------

void InferencePanel::render(const DashboardConfig& cfg) {
    // Layout: chat on left, telemetry strip on right
    float panel_w    = ImGui::GetContentRegionAvail().x;
    float chat_w     = panel_w * 0.65f;
    float telem_w    = panel_w * 0.33f;

    // Chat column
    ImGui::BeginChild("chat_col", ImVec2(chat_w, 0), false);
    render_stats_overlay();
    ImGui::Separator();
    render_chat_history();
    ImGui::Separator();
    render_input_bar(cfg);
    ImGui::EndChild();

    ImGui::SameLine();

    // Telemetry column
    ImGui::BeginChild("telem_col", ImVec2(telem_w, 0), true);
    render_telemetry_strip();
    ImGui::EndChild();
}

void InferencePanel::render_stats_overlay() {
    std::lock_guard<std::mutex> lk(m_mutex);

    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Inference Stats");
    ImGui::SameLine(0, 20);

    if (m_live_stats.running) {
        ImGui::TextColored({1.0f,0.8f,0.0f,1.0f}, "RUNNING");
        ImGui::SameLine();
        if (ImGui::SmallButton("Stop")) stop_inference();
    } else {
        ImGui::TextDisabled("idle");
    }

    ImGui::Text("%.2f t/s", m_live_stats.tokens_per_sec);
    ImGui::SameLine(100);
    ImGui::Text("%d tokens", m_live_stats.total_tokens);
    ImGui::SameLine(220);
    ImGui::Text("%.1f s elapsed", m_live_stats.elapsed_sec);

    // Context bar
    float ctx_pct = m_context_size > 0
        ? (float)m_live_stats.context_used / m_context_size
        : 0.0f;
    char ctx_label[32];
    std::snprintf(ctx_label, sizeof(ctx_label),
                  "ctx %d / %d",
                  m_live_stats.context_used, m_context_size);
    ImGui::ProgressBar(ctx_pct, ImVec2(-1, 0), ctx_label);

    ImGui::Text("Avg latency: %.1f ms/tok  |  Session total: %d tokens",
                m_live_stats.avg_latency_ms,
                m_total_tokens_session);
}

void InferencePanel::render_chat_history() {
    float available_h = ImGui::GetContentRegionAvail().y - 60.0f;
    ImGui::BeginChild("chat_history",
                      ImVec2(0, available_h), false);

    std::lock_guard<std::mutex> lk(m_mutex);

    for (auto& msg : m_messages) {
        if (msg.is_user) {
            ImGui::TextColored({0.5f,0.7f,1.0f,1.0f}, "You:");
            ImGui::SameLine();
            ImGui::TextWrapped("%s", msg.text.c_str());
        } else {
            ImGui::TextColored({0.3f,0.9f,0.5f,1.0f}, "Model:");
            ImGui::SameLine();
            ImGui::TextWrapped("%s", msg.text.c_str());
            ImGui::TextDisabled(
                "  %.2f t/s  |  %d tokens  |  %.1f s  |  %.1f ms/tok",
                msg.stats.tokens_per_sec,
                msg.stats.total_tokens,
                msg.stats.elapsed_sec,
                msg.stats.avg_latency_ms);
        }
        ImGui::Separator();
    }

    // Streaming response in progress
    if (m_live_stats.running && !m_streaming_text.empty()) {
        ImGui::TextColored({0.3f,0.9f,0.5f,1.0f}, "Model:");
        ImGui::SameLine();
        ImGui::TextWrapped("%s", m_streaming_text.c_str());
        // Blinking cursor
        float t = (float)ImGui::GetTime();
        if ((int)(t * 2) % 2 == 0)
            ImGui::TextUnformatted("▌");
    }

    if (m_scroll_to_bottom) {
        ImGui::SetScrollHereY(1.0f);
        m_scroll_to_bottom = false;
    }

    ImGui::EndChild();
}

void InferencePanel::render_input_bar(const DashboardConfig& cfg) {
    bool submit = false;

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80.0f);

    if (m_focus_input) {
        ImGui::SetKeyboardFocusHere();
        m_focus_input = false;
    }

    ImGuiInputTextFlags flags =
        ImGuiInputTextFlags_EnterReturnsTrue;

    if (m_running) flags |= ImGuiInputTextFlags_ReadOnly;

    if (ImGui::InputText("##input", m_input_buf,
                         sizeof(m_input_buf), flags))
        submit = true;

    ImGui::SameLine();

    if (m_running) {
        ImGui::BeginDisabled();
        ImGui::Button("Send");
        ImGui::EndDisabled();
    } else {
        if (ImGui::Button("Send")) submit = true;
    }

    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_messages.clear();
        m_total_tokens_session = 0;
    }

    if (submit && !m_running && m_input_buf[0] != '\0') {
        std::string prompt(m_input_buf);
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            ChatMessage user_msg;
            user_msg.is_user = true;
            user_msg.text    = prompt;
            m_messages.push_back(user_msg);
            m_scroll_to_bottom = true;
        }
        std::memset(m_input_buf, 0, sizeof(m_input_buf));
        m_focus_input = true;
        start_inference(cfg, prompt);
    }
}

void InferencePanel::render_telemetry_strip() {
    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Live Telemetry");

    std::lock_guard<std::mutex> lk(m_mutex);

    if (m_infer_times.empty()) {
        ImGui::TextDisabled("Run an inference to see telemetry.");
        return;
    }

    int n = (int)m_infer_times.size();
    float* xs  = m_infer_times.data();

    auto small_plot = [&](const char* label,
                          std::vector<float>& ys,
                          const char* y_label,
                          ImVec4 color) {
        if ((int)ys.size() < n) return;
        if (ImPlot::BeginPlot(label, ImVec2(-1, 120),
                              ImPlotFlags_NoLegend |
                              ImPlotFlags_NoMenus))
        {
            ImPlot::SetupAxes("s", y_label,
                ImPlotAxisFlags_AutoFit,
                ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine(label, xs, ys.data(), n);
            ImPlot::EndPlot();
        }
    };

    small_plot("t/s##infer",   m_infer_tps,
               "t/s",  {0.3f,0.9f,0.3f,1.0f});
    small_plot("Temp##infer",  m_infer_temp,
               "C",    {1.0f,0.4f,0.2f,1.0f});
    small_plot("Power##infer", m_infer_power,
               "W",    {0.9f,0.7f,0.1f,1.0f});
    small_plot("Freq##infer",  m_infer_freq,
               "MHz",  {0.4f,0.6f,1.0f,1.0f});
}