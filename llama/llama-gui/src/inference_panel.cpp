#include<iostream>
#include "inference_panel.h"
#include "imgui.h"
#include "implot.h"
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <pty.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <sstream>
#include "json.hpp"

using clk  = std::chrono::steady_clock;
using json = nlohmann::json;

static double elapsed_sec(clk::time_point start) {
    return std::chrono::duration<double>(clk::now() - start).count();
}

// ---- InferencePanel --------------------------------------------------------

InferencePanel::InferencePanel() {
    m_session_start = clk::now();
}

InferencePanel::~InferencePanel() {
    kill_base();
    kill_kl_helper();
}

// ---- Base model (pty) ------------------------------------------------------

bool InferencePanel::spawn_base(const DashboardConfig& cfg) {
    kill_base();

    int master_fd, slave_fd;
    if (openpty(&master_fd, &slave_fd, nullptr, nullptr, nullptr) < 0)
        return false;

    pid_t pid = fork();
    if (pid < 0) { close(master_fd); close(slave_fd); return false; }

    if (pid == 0) {
        close(master_fd);
        setsid();
        ioctl(slave_fd, TIOCSCTTY, 0);
        dup2(slave_fd, STDIN_FILENO);
        dup2(slave_fd, STDOUT_FILENO);
        dup2(slave_fd, STDERR_FILENO);
        close(slave_fd);

        std::string threads = std::to_string(cfg.n_threads);
        std::string npredict= std::to_string(cfg.n_predict);
        std::string kv      = DashboardConfig::KV_TYPES[cfg.kv_type_idx];

        std::vector<std::string> args_str = {
            cfg.llama_cli,
            "-m",  cfg.base_gguf_path,
            "-t",  threads,
            "-n",  npredict,
            "-fa", "on",
            "-ctk", kv,
            "-ctv", kv,
            "--log-disable",
        };
        std::vector<char*> argv;
        for (auto& s : args_str) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(1);
    }

    close(slave_fd);
    int flags = fcntl(master_fd, F_GETFL, 0);
    fcntl(master_fd, F_SETFL, flags | O_NONBLOCK);

    std::cerr << "Base PID = " << pid << '\n';
    sleep(1);
    std::cerr << "alive = "
          << (waitpid(pid, nullptr, WNOHANG) == 0)
          << '\n';
    
    m_base_pid    = pid;
    m_base_master = master_fd;
    m_base_ready  = false;
    m_base_reader_running = true;
    m_base_reader = std::thread(&InferencePanel::base_reader_func, this);
    return true;
}

void InferencePanel::kill_base() {
    m_base_reader_running = false;
    if (m_base_pid > 0) {
        kill(m_base_pid, SIGTERM);
        waitpid(m_base_pid, nullptr, 0);
        m_base_pid = -1;
    }
    if (m_base_master >= 0) { close(m_base_master); m_base_master = -1; }
    if (m_base_reader.joinable()) m_base_reader.join();
    m_base_ready = false;
    m_base_generating = false;
}

bool InferencePanel::base_alive() {
    if (m_base_pid <= 0) return false;
    return waitpid(m_base_pid, nullptr, WNOHANG) == 0;
}

void InferencePanel::send_to_base(const std::string& prompt) {
    if (m_base_master < 0) return;
    std::string line = prompt + "\n";
    write(m_base_master, line.c_str(), line.size());
}

void InferencePanel::base_reader_func() {
    char buf[256];
    while (m_base_reader_running.load()) {
        ssize_t n = read(m_base_master, buf, sizeof(buf) - 1);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(10));
                continue;
            }
            break;
        }
        if (n == 0) break;
        buf[n] = '\0';
        std::cerr << "[BASE RAW]\n" << buf << "\n";
        m_base_read_buf += std::string(buf, n);

        // Detect initial ready prompt "> "
        if (!m_base_ready) {
            if (m_base_read_buf.find("> ") != std::string::npos) {
                std::lock_guard<std::mutex> lk(m_mutex);
                m_base_ready = true;
                m_base_read_buf.clear();
            }
            continue;
        }

        // Detect end of response: "\n> "
        size_t end_pos = m_base_read_buf.find("\n> ");
        if (end_pos != std::string::npos) {
            std::string response = m_base_read_buf.substr(0, end_pos);
            m_base_read_buf = m_base_read_buf.substr(end_pos + 3);

            // Strip timing line
            size_t stats = response.rfind("[ Prompt:");
            if (stats != std::string::npos)
                response = response.substr(0, stats);
            while (!response.empty() &&
                   (response.back() == '\n' || response.back() == ' '))
                response.pop_back();

            {
                std::lock_guard<std::mutex> lk(m_mutex);
                ChatMessage msg;
                msg.is_user         = false;
                msg.text            = response;
                msg.stats           = m_base_stats;
                msg.stats.running   = false;
                m_base_messages.push_back(msg);
                m_base_streaming.clear();
                m_base_generating        = false;
                m_base_stats.running     = false;
                m_base_session_tokens   += msg.stats.total_tokens;
                m_scroll_base            = true;
            }
        } else {
            // Streaming token chunk
            std::lock_guard<std::mutex> lk(m_mutex);
            m_base_streaming += std::string(buf, n);
            m_scroll_base = true;

            // Update rolling tps
            m_base_stamps.push_back({clk::now()});
            if ((int)m_base_stamps.size() > TOKEN_WINDOW)
                m_base_stamps.pop_front();
            if (m_base_stamps.size() >= 2) {
                double w = std::chrono::duration<double>(
                    m_base_stamps.back().t -
                    m_base_stamps.front().t).count();
                if (w > 0.0)
                    m_base_stats.tokens_per_sec =
                        (float)(m_base_stamps.size() - 1) / w;
            }
            m_base_stats.total_tokens++;

            // Sample telemetry
            if (m_telemetry) {
                auto s = m_telemetry->latest();
                float t = (float)elapsed_sec(m_session_start);
                m_infer_temp.push_back(s.pkg_temp_c);
                m_infer_power.push_back(s.pkg_power_w);
                m_infer_tps_base.push_back(m_base_stats.tokens_per_sec);
                m_infer_times.push_back(t);
                if (!s.cpu_freq_mhz.empty()) {
                    float avg = std::accumulate(
                        s.cpu_freq_mhz.begin(),
                        s.cpu_freq_mhz.end(), 0.0f)
                        / s.cpu_freq_mhz.size();
                    m_infer_freq.push_back(avg);
                }
            }
        }
    }
    m_base_reader_running = false;
}

// ---- KL helper (pipe process) ----------------------------------------------

bool InferencePanel::spawn_kl_helper(const DashboardConfig& cfg) {
    kill_kl_helper();

    int stdin_pipe[2], stdout_pipe[2];
    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0) return false;

    pid_t pid = fork();
    if (pid < 0) return false;

    if (pid == 0) {
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        dup2(stdin_pipe[0],  STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        std::vector<std::string> args_str = {
            "python3",
            cfg.kl_helper_script,
            cfg.base_model_hf_id,
            cfg.finetuned_model_hf_id,
        };
        std::vector<char*> argv;
        for (auto& s : args_str) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(1);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    int flags = fcntl(stdout_pipe[0], F_GETFL, 0);
    fcntl(stdout_pipe[0], F_SETFL, flags | O_NONBLOCK);

    m_kl_pid       = pid;
    m_kl_stdin_fd  = stdin_pipe[1];
    m_kl_stdout_fd = stdout_pipe[0];
    m_kl_ready     = false;
    m_kl_reader_running = true;
    m_kl_reader = std::thread(&InferencePanel::kl_reader_func, this);
    return true;
}

void InferencePanel::kill_kl_helper() {
    m_kl_reader_running = false;
    if (m_kl_pid > 0) {
        kill(m_kl_pid, SIGTERM);
        waitpid(m_kl_pid, nullptr, 0);
        m_kl_pid = -1;
    }
    if (m_kl_stdin_fd  >= 0) { close(m_kl_stdin_fd);  m_kl_stdin_fd  = -1; }
    if (m_kl_stdout_fd >= 0) { close(m_kl_stdout_fd); m_kl_stdout_fd = -1; }
    if (m_kl_reader.joinable()) m_kl_reader.join();
    m_kl_ready     = false;
    m_kl_generating= false;
    m_kl_computing = false;
}

bool InferencePanel::kl_helper_alive() {
    if (m_kl_pid <= 0) return false;
    return waitpid(m_kl_pid, nullptr, WNOHANG) == 0;
}

void InferencePanel::send_to_kl_helper(const std::string& prompt) {
    if (m_kl_stdin_fd < 0) return;
    json req;
    req["prompt"] = prompt;
    std::string line = req.dump() + "\n";
    write(m_kl_stdin_fd, line.c_str(), line.size());
}

void InferencePanel::kl_reader_func() {
    char buf[65536];
    while (m_kl_reader_running.load()) {
        ssize_t n = read(m_kl_stdout_fd, buf, sizeof(buf) - 1);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(10));
                continue;
            }
            break;
        }
        if (n == 0) break;
        buf[n] = '\0';
        m_kl_read_buf += std::string(buf, n);

        size_t pos;
        while ((pos = m_kl_read_buf.find('\n')) != std::string::npos) {
            std::string line = m_kl_read_buf.substr(0, pos);
            m_kl_read_buf    = m_kl_read_buf.substr(pos + 1);
            if (line.empty()) continue;

            try {
                json msg = json::parse(line);
                std::string type = msg["type"].get<std::string>();

                if (type == "ready") {
                    std::lock_guard<std::mutex> lk(m_mutex);
                    m_kl_ready = true;
                    fprintf(stderr, "[kl_helper] ready\n");

                } else if (type == "token") {
                    std::string token = msg["token"].get<std::string>();
                    std::lock_guard<std::mutex> lk(m_mutex);
                    m_ft_streaming += token;
                    m_scroll_ft = true;

                    // Update rolling tps for finetuned column
                    m_ft_stamps.push_back({clk::now()});
                    if ((int)m_ft_stamps.size() > TOKEN_WINDOW)
                        m_ft_stamps.pop_front();
                    if (m_ft_stamps.size() >= 2) {
                        double w = std::chrono::duration<double>(
                            m_ft_stamps.back().t -
                            m_ft_stamps.front().t).count();
                        if (w > 0.0)
                            m_ft_stats.tokens_per_sec =
                                (float)(m_ft_stamps.size() - 1) / w;
                    }
                    m_ft_stats.total_tokens++;
                    m_infer_tps_ft.push_back(
                        m_ft_stats.tokens_per_sec);

                } else if (type == "kl") {
                    KLResult result;
                    result.valid    = true;
                    result.mean_forward_kl = msg["mean_forward_kl"].get<float>();
                    result.mean_reverse_kl = msg["mean_reverse_kl"].get<float>();
                    result.n_tokens = msg["n_tokens"].get<int>();

                    for (auto& v : msg["forward_kl_per_position"])
                        result.forward_kl_per_position.push_back(
                            v.get<float>());

                    for (auto& v : msg["reverse_kl_per_position"])
                        result.reverse_kl_per_position.push_back(
                            v.get<float>());

                    for (auto& p : msg["top_tokens_base"])
                        result.top_base.push_back({
                            p[0].get<std::string>(),
                            p[1].get<float>()});
                    for (auto& p : msg["top_tokens_finetuned"])
                        result.top_finetuned.push_back({
                            p[0].get<std::string>(),
                            p[1].get<float>()});

                    std::string ft_response =
                        msg["finetuned_response"].get<std::string>();

                    {
                        std::lock_guard<std::mutex> lk(m_mutex);
                        m_latest_kl = result;

                        ChatMessage ft_msg;
                        ft_msg.is_user  = false;
                        ft_msg.text     = ft_response;
                        ft_msg.stats    = m_ft_stats;
                        m_ft_messages.push_back(ft_msg);
                        m_ft_streaming.clear();
                        m_ft_session_tokens += m_ft_stats.total_tokens;
                        m_ft_stats        = {};
                        m_kl_generating   = false;
                        m_kl_computing    = false;
                        m_scroll_ft       = true;
                    }

                } else if (type == "error") {
                    std::string err = msg["error"].get<std::string>();
                    std::lock_guard<std::mutex> lk(m_mutex);
                    m_latest_kl.valid = false;
                    m_latest_kl.error = err;
                    m_kl_generating   = false;
                    m_kl_computing    = false;
                    fprintf(stderr, "[kl_helper] error: %s\n", err.c_str());
                }

            } catch (std::exception& e) {
                fprintf(stderr, "[kl parse] %s\n", e.what());
            }
        }
    }
    m_kl_reader_running = false;
}

// ---- Render ----------------------------------------------------------------

void InferencePanel::render(const DashboardConfig& cfg) {
    if (!m_processes_spawned ||
        (!base_alive() && !kl_helper_alive())) {

        if (ImGui::Button("Start Models")) {
            bool b = spawn_base(cfg);
            bool k = spawn_kl_helper(cfg);
            m_processes_spawned = b && k;
            m_n_predict = cfg.n_predict;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Models not running.");

        if (m_processes_spawned && !base_alive() && !kl_helper_alive()) {
            ImGui::TextColored({1.0f,0.3f,0.3f,1.0f},
                               "Processes died. Restart.");
            m_processes_spawned = false;
        }
        return;
    }

    // Loading screen
    if (!m_base_ready || !m_kl_ready) {
        ImGui::TextColored({1.0f,0.8f,0.0f,1.0f}, "Loading...");
        ImGui::Text("  Base model (llama-cli): %s",
                    m_base_ready ? "ready" : "loading...");
        ImGui::Text("  KL helper (transformers): %s",
                    m_kl_ready   ? "ready" : "loading (may take 1-2 min)...");
        ImGui::SameLine();
        if (ImGui::SmallButton("Kill")) {
            kill_base(); kill_kl_helper();
            m_processes_spawned = false;
        }
        return;
    }

    render_controls(cfg);
    ImGui::Separator();

    float avail_h = ImGui::GetContentRegionAvail().y;
    float chat_h  = avail_h * 0.63f;
    float kl_h    = avail_h * 0.20f;
    float telem_h = avail_h * 0.15f;

    ImGui::BeginChild("chat_area", ImVec2(0, chat_h), false);
    render_chat_columns(cfg);
    ImGui::EndChild();

    ImGui::Separator();
    ImGui::BeginChild("kl_area", ImVec2(0, kl_h), false);
    render_kl_panel();
    ImGui::EndChild();

    ImGui::Separator();
    ImGui::BeginChild("telem_area", ImVec2(0, telem_h), false);
    render_telemetry_strip();
    ImGui::EndChild();
}

void InferencePanel::render_controls(const DashboardConfig& cfg) {
    bool any_generating =
        m_base_generating || m_kl_generating || m_kl_computing;

    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Dual Inference");
    ImGui::SameLine(0, 20);

    if (m_base_generating)
        ImGui::TextColored({1.0f,0.8f,0.0f,1.0f}, "Base generating...");
    else if (m_kl_generating)
        ImGui::TextColored({0.3f,0.9f,0.5f,1.0f}, "Finetuned generating...");
    else if (m_kl_computing)
        ImGui::TextColored({0.5f,0.8f,1.0f,1.0f}, "Computing KL...");
    else
        ImGui::TextDisabled("ready");

    ImGui::SameLine();
    if (ImGui::SmallButton("Kill All")) {
        kill_base(); kill_kl_helper();
        m_processes_spawned = false;
    }

    ImGui::Text("Base: %.1f t/s  |  Finetuned: %.1f t/s  |  "
                "Session: %d / %d tokens",
                m_base_stats.tokens_per_sec,
                m_ft_stats.tokens_per_sec,
                m_base_session_tokens,
                m_ft_session_tokens);
}

void InferencePanel::render_chat_columns(const DashboardConfig& cfg) {
    float avail   = ImGui::GetContentRegionAvail().x;
    float col_w   = avail * 0.5f - 4.0f;
    float chat_h  = ImGui::GetContentRegionAvail().y - 50.0f;

    // ---- Base model column ----
    ImGui::BeginChild("base_col", ImVec2(col_w, 0), false);
    ImGui::TextColored({0.5f,0.7f,1.0f,1.0f},
                       "Base Model  (%.1f t/s)",
                       m_base_stats.tokens_per_sec);
    ImGui::BeginChild("base_chat", ImVec2(0, chat_h), true);
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        for (auto& msg : m_base_messages) {
            if (msg.is_user)
                ImGui::TextColored({0.7f,0.7f,0.7f,1.0f},
                                   "You: %s", msg.text.c_str());
            else {
                ImGui::TextColored({0.5f,0.7f,1.0f,1.0f}, "Base:");
                ImGui::SameLine();
                ImGui::TextWrapped("%s", msg.text.c_str());
                ImGui::TextDisabled("  %.1f t/s",
                                    msg.stats.tokens_per_sec);
            }
            ImGui::Separator();
        }
        if (m_base_generating && !m_base_streaming.empty()) {
            ImGui::TextColored({0.5f,0.7f,1.0f,1.0f}, "Base:");
            ImGui::SameLine();
            ImGui::TextWrapped("%s", m_base_streaming.c_str());
            if ((int)(ImGui::GetTime() * 2) % 2 == 0)
                ImGui::TextUnformatted("▌");
        }
        if (m_scroll_base) {
            ImGui::SetScrollHereY(1.0f);
            m_scroll_base = false;
        }
    }
    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::SameLine();

    // ---- Finetuned model column ----
    ImGui::BeginChild("ft_col", ImVec2(col_w, 0), false);
    ImGui::TextColored({0.3f,0.9f,0.5f,1.0f},
                       "Finetuned Model  (%.1f t/s)",
                       m_ft_stats.tokens_per_sec);
    ImGui::BeginChild("ft_chat", ImVec2(0, chat_h), true);
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        for (auto& msg : m_ft_messages) {
            if (msg.is_user)
                ImGui::TextColored({0.7f,0.7f,0.7f,1.0f},
                                   "You: %s", msg.text.c_str());
            else {
                ImGui::TextColored({0.3f,0.9f,0.5f,1.0f}, "Finetuned:");
                ImGui::SameLine();
                ImGui::TextWrapped("%s", msg.text.c_str());
                ImGui::TextDisabled("  %.1f t/s",
                                    msg.stats.tokens_per_sec);
            }
            ImGui::Separator();
        }
        if (m_kl_generating && !m_ft_streaming.empty()) {
            ImGui::TextColored({0.3f,0.9f,0.5f,1.0f}, "Finetuned:");
            ImGui::SameLine();
            ImGui::TextWrapped("%s", m_ft_streaming.c_str());
            if ((int)(ImGui::GetTime() * 2) % 2 == 0)
                ImGui::TextUnformatted("▌");
        }
        if (m_kl_computing) {
            ImGui::TextColored({0.5f,0.8f,1.0f,1.0f},
                               "Computing KL divergence...");
        }
        if (m_scroll_ft) {
            ImGui::SetScrollHereY(1.0f);
            m_scroll_ft = false;
        }
    }
    ImGui::EndChild();
    ImGui::EndChild();

    // Input bar spanning full width
    render_input_bar(cfg);
}

void InferencePanel::render_input_bar(const DashboardConfig& cfg) {
    bool busy = m_base_generating || m_kl_generating || m_kl_computing;
    bool submit = false;

    ImGui::SetNextItemWidth(
        ImGui::GetContentRegionAvail().x - 120.0f);

    if (m_focus_input) {
        ImGui::SetKeyboardFocusHere();
        m_focus_input = false;
    }

    ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue;
    if (busy) flags |= ImGuiInputTextFlags_ReadOnly;

    if (ImGui::InputText("##prompt", m_input_buf,
                         sizeof(m_input_buf), flags))
        submit = true;

    ImGui::SameLine();
    if (busy) {
        ImGui::BeginDisabled();
        ImGui::Button("Send");
        ImGui::EndDisabled();
    } else {
        if (ImGui::Button("Send")) submit = true;
    }

    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_base_messages.clear();
        m_ft_messages.clear();
        m_base_session_tokens = 0;
        m_ft_session_tokens   = 0;
        m_latest_kl           = {};
        m_infer_temp.clear();
        m_infer_power.clear();
        m_infer_freq.clear();
        m_infer_tps_base.clear();
        m_infer_tps_ft.clear();
        m_infer_times.clear();
    }

    if (submit && !busy && m_input_buf[0] != '\0') {
        std::string prompt(m_input_buf);
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            ChatMessage user_msg;
            user_msg.is_user = true;
            user_msg.text    = prompt;
            m_base_messages.push_back(user_msg);
            m_ft_messages.push_back(user_msg);
            m_current_prompt     = prompt;
            m_base_generating    = true;
            m_kl_generating      = true;
            m_base_stats         = {};
            m_ft_stats           = {};
            m_base_stats.running = true;
            m_ft_stats.running   = true;
            m_base_stamps.clear();
            m_ft_stamps.clear();
            m_scroll_base = true;
            m_scroll_ft   = true;
        }
        std::memset(m_input_buf, 0, sizeof(m_input_buf));
        m_focus_input = true;

        // Send simultaneously to both
        send_to_base(prompt);
        send_to_kl_helper(prompt);
    }
}

void InferencePanel::render_kl_panel() {
    
    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f},"Forward and Reverse KL Divergence");

    std::lock_guard<std::mutex> lk(m_mutex);

    if (m_kl_computing) {
        ImGui::TextColored({1.0f,0.8f,0.0f,1.0f},
                           "Running forward passes...");
        return;
    }
    if (!m_latest_kl.valid) {
        if (!m_latest_kl.error.empty())
            ImGui::TextColored({1.0f,0.3f,0.3f,1.0f},
                               "Error: %s", m_latest_kl.error.c_str());
        else
            ImGui::TextDisabled(
                "Send a prompt to compute KL divergence.");
        return;
    }


    ImGui::Text("Forward KL (Base -> Finetuned): %.4f",
                m_latest_kl.mean_forward_kl);

    ImGui::Text("Reverse KL (Finetuned -> Base): %.4f",
                m_latest_kl.mean_reverse_kl);

    ImGui::Text("Tokens: %d",
                m_latest_kl.n_tokens);

    float panel_w = ImGui::GetContentRegionAvail().x;
    float half_w  = panel_w * 0.5f - 4.0f;

    // Left: per-position KL trace
    if (!m_latest_kl.forward_kl_per_position.empty()) {

        int n = (int)m_latest_kl.forward_kl_per_position.size();

        std::vector<float> xs(n);
        std::iota(xs.begin(), xs.end(), 0.0f);

        if (ImPlot::BeginPlot(
                "KL per token##kltrace",
                ImVec2(half_w,160)))
        {
            ImPlot::SetupAxes(
                "Token position",
                "KL",
                ImPlotAxisFlags_AutoFit,
                ImPlotAxisFlags_AutoFit);

            ImPlot::PlotLine(
                "Forward",
                xs.data(),
                m_latest_kl.forward_kl_per_position.data(),
                n);

            ImPlot::PlotLine(
                "Reverse",
                xs.data(),
                m_latest_kl.reverse_kl_per_position.data(),
                n);

            ImPlot::EndPlot();
        }
    }

    ImGui::SameLine();

    // Right: top-K token probability comparison
    if (!m_latest_kl.top_base.empty()) {
        int n = std::min((int)m_latest_kl.top_base.size(), 12);

        std::vector<double>      base_p(n), ft_p(n);
        std::vector<std::string> label_strs(n);
        std::vector<const char*> labels(n);

        std::unordered_map<std::string,float> ft_map;
        for (auto& p : m_latest_kl.top_finetuned)
            ft_map[p.token] = p.prob;

        for (int i = 0; i < n; i++) {
            label_strs[i] = m_latest_kl.top_base[i].token;
            // sanitize for display
            for (char& c : label_strs[i])
                if (c < 32) c = '_';
            labels[i] = label_strs[i].c_str();
            base_p[i] = m_latest_kl.top_base[i].prob;
            auto it   = ft_map.find(m_latest_kl.top_base[i].token);
            ft_p[i]   = it != ft_map.end() ? it->second : 0.0f;
        }

        if (ImPlot::BeginPlot("Next-token distribution##dist",
                              ImVec2(half_w, 160)))
        {
            ImPlot::SetupAxes("token", "probability",
                ImPlotAxisFlags_AutoFit,
                ImPlotAxisFlags_AutoFit);
            
            ImPlot::SetupAxisTicks(ImAxis_X1, 0.0, n - 1.0, n, labels.data());
            ImPlot::PlotBars("Base",      base_p.data(), n, 0.35, -0.2);
            ImPlot::PlotBars("Finetuned", ft_p.data(),   n, 0.35,  0.2);
            ImPlot::EndPlot();
        }
    }
}

void InferencePanel::render_telemetry_strip() {
    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Inference Telemetry");

    std::lock_guard<std::mutex> lk(m_mutex);
    if (m_infer_times.empty()) {
        ImGui::TextDisabled("Send a prompt to see telemetry.");
        return;
    }

    
    auto small_plot = [&](const char* label,
                        std::vector<float>& ys,
                        const char* ylabel)
    {
        int n = std::min((int)m_infer_times.size(),
                        (int)ys.size());

        if (n < 2)
            return;

        if (ImPlot::BeginPlot(label, ImVec2(-1,120),
                            ImPlotFlags_NoLegend |
                            ImPlotFlags_NoMenus))
        {
            ImPlot::SetupAxes("s", ylabel,
                            ImPlotAxisFlags_AutoFit,
                            ImPlotAxisFlags_AutoFit);

            ImPlot::PlotLine(label,
                            m_infer_times.data(),
                            ys.data(),
                            n);

            ImPlot::EndPlot();
        }
    };

    float w4 = ImGui::GetContentRegionAvail().x;
    small_plot("Base t/s##bt",  m_infer_tps_base, "t/s");
    
    small_plot("FT t/s##ft",    m_infer_tps_ft,   "t/s");
    ImGui::SameLine();
    small_plot("Temp##tmp",     m_infer_temp,     "C");
    
    small_plot("Power##pwr",    m_infer_power,    "W");
    ImGui::SameLine();
}