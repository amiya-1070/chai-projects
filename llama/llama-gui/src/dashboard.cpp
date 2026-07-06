#include "dashboard.h"
#include "imgui.h"
#include "implot.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

Dashboard::Dashboard() {}
Dashboard::~Dashboard() { shutdown(); }

bool Dashboard::init() {
    // Start telemetry collection
    m_telemetry.start();

    // Pass telemetry pointer to panels
    m_bench.set_telemetry(&m_telemetry);
    m_infer.set_telemetry(&m_telemetry);

    // Open storage database
    auto& cfg = m_config.config();
    if (cfg.db_path[0] != '\0') {
        // Ensure directory exists
        fs::path db_dir = fs::path(cfg.db_path).parent_path();
        if (!db_dir.empty())
            fs::create_directories(db_dir);

        if (!m_storage.open(cfg.db_path)) {
            std::snprintf(m_status, sizeof(m_status),
                          "Warning: could not open DB at %s", cfg.db_path);
        } else {
            std::snprintf(m_status, sizeof(m_status),
                          "DB opened: %s", cfg.db_path);
        }
    }
    m_bench.set_storage(&m_storage);
    m_model_sweep.set_storage(&m_storage); 
    m_model_sweep.set_telemetry(&m_telemetry);

    return true;
}

void Dashboard::shutdown() {
    m_telemetry.stop();
    m_storage.close();
}

// ---- Main render -----------------------------------------------------------

void Dashboard::render() {
    // Re-open DB if config path changed
    if (m_config.changed()) {
        auto& cfg = m_config.config();
        if (!m_storage.is_open() && cfg.db_path[0] != '\0') {
            fs::create_directories(
                fs::path(cfg.db_path).parent_path());
            m_storage.open(cfg.db_path);
            m_bench.set_storage(&m_storage);
        }
    }

    render_menu_bar();
    render_telemetry_bar();
    render_main_tabs();
}

// ---- Menu bar --------------------------------------------------------------

void Dashboard::render_menu_bar() {
    if (!ImGui::BeginMainMenuBar()) return;

    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Config Panel",    nullptr, &m_show_config);
        ImGui::MenuItem("Telemetry Panel", nullptr, &m_show_telemetry);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Presets")) {
        auto& cfg = m_config.config();

        if (ImGui::MenuItem("P-cores only (taskset 0-3, t=4)")) {
            cfg.n_threads   = 4;
            cfg.use_taskset = true;
            std::strncpy(cfg.cpu_mask, "0-3",
                         sizeof(cfg.cpu_mask) - 1);
            std::snprintf(m_status, sizeof(m_status),
                          "Preset: P-cores only");
        }
        if (ImGui::MenuItem("E-cores only (taskset 4-11, t=8)")) {
            cfg.n_threads   = 8;
            cfg.use_taskset = true;
            std::strncpy(cfg.cpu_mask, "4-11",
                         sizeof(cfg.cpu_mask) - 1);
            std::snprintf(m_status, sizeof(m_status),
                          "Preset: E-cores only");
        }
        if (ImGui::MenuItem("All cores t=8 (optimal)")) {
            cfg.n_threads   = 8;
            cfg.use_taskset = false;
            std::snprintf(m_status, sizeof(m_status),
                          "Preset: All cores t=8");
        }
        if (ImGui::MenuItem("All cores t=10")) {
            cfg.n_threads   = 10;
            cfg.use_taskset = false;
            std::snprintf(m_status, sizeof(m_status),
                          "Preset: All cores t=10");
        }
        if (ImGui::MenuItem("Default (t=2, no flags)")) {
            cfg.n_threads   = 2;
            cfg.use_taskset = false;
            cfg.flash_attn  = false;
            cfg.mmap_off    = false;
            cfg.kv_type_idx = 0; // f16
            std::snprintf(m_status, sizeof(m_status),
                          "Preset: Default baseline");
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("About")) {
            std::snprintf(m_status, sizeof(m_status),
                "Llama.cpp Benchmark Dashboard — "
                "built on Dear ImGui + ImPlot");
        }
        ImGui::EndMenu();
    }

    // Right-aligned status
    float status_w = ImGui::CalcTextSize(m_status).x + 20.0f;
    ImGui::SetCursorPosX(
        ImGui::GetContentRegionMax().x - status_w);
    ImGui::TextDisabled("%s", m_status);

    ImGui::EndMainMenuBar();
}

// ---- Telemetry bar ---------------------------------------------------------

void Dashboard::render_telemetry_bar() {
    auto snap = m_telemetry.latest();

    // Compute average freq across active CPUs
    float avg_freq = 0.0f;
    if (!snap.cpu_freq_mhz.empty()) {
        avg_freq = std::accumulate(
            snap.cpu_freq_mhz.begin(),
            snap.cpu_freq_mhz.end(), 0.0f)
            / snap.cpu_freq_mhz.size();
    }

    // One-line telemetry bar just below menu bar
    ImGuiWindowFlags bar_flags =
        ImGuiWindowFlags_NoDecoration     |
        ImGuiWindowFlags_NoInputs         |
        ImGuiWindowFlags_NoMove           |
        ImGuiWindowFlags_NoSavedSettings  |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNav;

    float menu_h    = ImGui::GetFrameHeight();
    float bar_h     = 28.0f;
    float disp_w    = ImGui::GetIO().DisplaySize.x;

    ImGui::SetNextWindowPos({0, menu_h});
    ImGui::SetNextWindowSize({disp_w, bar_h});
    ImGui::SetNextWindowBgAlpha(0.85f);

    ImGui::Begin("##telem_bar", nullptr, bar_flags);

    // Temp
    ImVec4 temp_col = snap.pkg_temp_c > 85.0f
        ? ImVec4(1.0f, 0.2f, 0.2f, 1.0f)
        : snap.pkg_temp_c > 70.0f
        ? ImVec4(1.0f, 0.7f, 0.1f, 1.0f)
        : ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
    ImGui::TextColored(temp_col, "Temp: %.0f C", snap.pkg_temp_c);
    ImGui::SameLine(120);

    // Power
    ImVec4 pwr_col = snap.pkg_power_w > 20.0f
        ? ImVec4(1.0f, 0.5f, 0.1f, 1.0f)
        : ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
    ImGui::TextColored(pwr_col,  "Power: %.1f W", snap.pkg_power_w);
    ImGui::SameLine(240);

    // Freq
    ImGui::Text("Avg freq: %.0f MHz", avg_freq);
    ImGui::SameLine(380);

    // RAM
    ImVec4 ram_col = (snap.ram_used_gib / snap.ram_total_gib) > 0.85f
        ? ImVec4(1.0f, 0.3f, 0.3f, 1.0f)
        : ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
    ImGui::TextColored(ram_col, "RAM: %.1f / %.1f GiB",
                       snap.ram_used_gib, snap.ram_total_gib);
    ImGui::SameLine(560);

    // Per-CPU freq mini strip
    ImGui::TextDisabled("CPUs:");
    ImGui::SameLine();
    for (int i = 0; i < (int)snap.cpu_freq_mhz.size(); i++) {
        float f    = snap.cpu_freq_mhz[i];
        float norm = f / 4700.0f; // normalize to P-core max
        ImVec4 col = norm > 0.85f
            ? ImVec4(0.3f, 0.9f, 0.3f, 1.0f)
            : norm > 0.5f
            ? ImVec4(0.9f, 0.8f, 0.2f, 1.0f)
            : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
        ImGui::TextColored(col, "%d", (int)(f / 100.0f)); // hundreds of MHz
        ImGui::SameLine(0, 2);
    }

    ImGui::End();
}

// ---- Main tab layout -------------------------------------------------------

void Dashboard::render_main_tabs() {
    float menu_h  = ImGui::GetFrameHeight();
    float bar_h   = 38.0f;
    float top_off = menu_h + bar_h;
    float disp_w  = ImGui::GetIO().DisplaySize.x;
    float disp_h  = ImGui::GetIO().DisplaySize.y;

    // Config sidebar (left)
    if (m_show_config) {
        ImGui::SetNextWindowPos({0, top_off});
        ImGui::SetNextWindowSize({m_config_width, disp_h - top_off});
        ImGui::SetNextWindowBgAlpha(0.95f);

        ImGuiWindowFlags cfg_flags =
            ImGuiWindowFlags_NoMove         |
            ImGuiWindowFlags_NoResize       |
            ImGuiWindowFlags_NoCollapse     |
            ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::Begin("Configuration", &m_show_config, cfg_flags);
        m_config.render();
        ImGui::End();
    }

    // Main content area (right of config)
    float content_x = m_show_config ? m_config_width : 0.0f;
    float content_w = disp_w - content_x;

    ImGui::SetNextWindowPos({content_x, top_off});
    ImGui::SetNextWindowSize({content_w, disp_h - top_off});
    ImGui::SetNextWindowBgAlpha(0.95f);

    ImGuiWindowFlags main_flags =
        ImGuiWindowFlags_NoMove         |
        ImGuiWindowFlags_NoResize       |
        ImGuiWindowFlags_NoCollapse     |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin("##main", nullptr, main_flags);

    if (ImGui::BeginTabBar("main_tabs")) {

        if (ImGui::BeginTabItem("Benchmark")) {
            m_bench.render(m_config.config());
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Model Size Sweep")) {
            m_model_sweep.render(m_config.config());
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Inference")) {
            m_infer.render(m_config.config());
            ImGui::EndTabItem();
        }

        if (m_show_telemetry) {
            if (ImGui::BeginTabItem("Telemetry")) {
                render_telemetry_window();
                ImGui::EndTabItem();
            }
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

// ---- Telemetry window ------------------------------------------------------

void Dashboard::render_telemetry_window() {
    auto hist = m_telemetry.history();

    std::vector<float> ts, temp, power, ram, freq, ipc, cache_miss;
    hist.get_arrays(ts, temp, power, ram, freq, ipc, cache_miss);

    if (ts.empty()) {
        ImGui::TextDisabled("Collecting data...");
        return;
    }

    int n = (int)ts.size();
    float* t = ts.data();

    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f},
                       "System Telemetry (last %.0f s)",
                       ts.back() - ts.front());

    auto telem_plot = [&](const char* title,
                          std::vector<float>& ys,
                          const char* ylabel,
                          ImVec4 color,
                          float ymin, float ymax) {
        ImPlot::PushStyleColor(ImPlotCol_PlotBg, {0.08f, 0.08f, 0.10f, 1.0f});
        if (ImPlot::BeginPlot(title, ImVec2(-1, 140),
                              ImPlotFlags_NoLegend |
                              ImPlotFlags_NoMenus))
        {
            ImPlot::SetupAxes("s", ylabel,
                ImPlotAxisFlags_AutoFit,
                ImPlotAxisFlags_None);
            ImPlot::SetupAxisLimits(ImAxis_Y1,
                ymin, ymax, ImGuiCond_Always);
            ImPlot::PlotLine(title, t, ys.data(), n);
            
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleColor(1);
    };

    telem_plot("Package Temperature##t", temp,
               "C",   {1.0f, 0.4f, 0.2f, 1.0f}, 0.0f, 100.0f);
    telem_plot("Package Power##p",       power,
               "W",   {0.9f, 0.7f, 0.1f, 1.0f}, 0.0f, 60.0f);
    telem_plot("Avg CPU Frequency##f",   freq,
               "MHz", {0.4f, 0.6f, 1.0f, 1.0f}, 0.0f, 5000.0f);
    telem_plot("RAM Used##r",            ram,
               "GiB", {0.6f, 0.4f, 0.9f, 1.0f}, 0.0f,
               m_telemetry.latest().ram_total_gib * 1.05f);

    // Per-CPU frequency heatmap
    ImGui::Separator();
    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Per-CPU Frequency");

    auto snap = m_telemetry.latest();
    for (int i = 0; i < (int)snap.cpu_freq_mhz.size(); i++) {
        float f    = snap.cpu_freq_mhz[i];
        float norm = f / 4700.0f;
        ImVec4 col = norm > 0.85f
            ? ImVec4(0.2f, 0.9f, 0.2f, 1.0f)
            : norm > 0.6f
            ? ImVec4(0.9f, 0.8f, 0.1f, 1.0f)
            : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);

        char bar_label[32];
        std::snprintf(bar_label, sizeof(bar_label),
                      "cpu%d\n%.0fMHz", i, f);

        ImGui::BeginGroup();
        ImGui::TextColored(col, "cpu%d", i);

        // Vertical-ish progress bar
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
        char id[16];
        std::snprintf(id, sizeof(id), "##cpu%d", i);
        ImGui::ProgressBar(norm, ImVec2(40.0f, 80.0f), "");
        ImGui::PopStyleColor();

        ImGui::TextColored(col, "%.0f", f);
        ImGui::EndGroup();

        if (i < (int)snap.cpu_freq_mhz.size() - 1)
            ImGui::SameLine(0, 6);
    }
}