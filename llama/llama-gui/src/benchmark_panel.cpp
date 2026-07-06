//benchmark_panel.cpp
#include "benchmark_panel.h"
#include "imgui.h"
#include "implot.h"
#include "process_utils.h"
#include <sstream>
#include <regex>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>

// ---- helpers ---------------------------------------------------------------

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size()
        && s.compare(0, prefix.size(), prefix) == 0;
}


// ---- BenchmarkPanel --------------------------------------------------------

BenchmarkPanel::BenchmarkPanel() {}

BenchmarkPanel::~BenchmarkPanel() {
    stop_run();
    m_sweep_cancel = true;
    if (m_sweep_thread.joinable()) m_sweep_thread.join();
}

void BenchmarkPanel::start_run(const DashboardConfig& cfg) {
    if (m_running) return;
    m_cancel = false;
    m_current = BenchRun();
    m_current.params = cfg.to_bench_params();
    m_temp_samples.clear();
    m_power_samples.clear();
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_log.clear();
    }
    m_running = true;
    m_thread = std::thread(&BenchmarkPanel::bench_thread_func,
                           this, cfg.to_bench_params());
}

void BenchmarkPanel::stop_run() {
    m_cancel = true;
    m_running = false;
    if (m_thread.joinable()) m_thread.join();
}

void BenchmarkPanel::bench_thread_func(BenchParams params) {
    std::string cmd = build_bench_command(params);
    fprintf(stderr, "CMD: %s\n", cmd.c_str());
    stream_command(cmd, [this](const std::string& line) {
        // Append to log
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_log.push_back(line);
            if ((int)m_log.size() > MAX_LOG) m_log.pop_front();
        }

        // Collect telemetry sample
        if (m_telemetry) {
            auto s = m_telemetry->latest();
            m_temp_samples.push_back(s.pkg_temp_c);
            m_power_samples.push_back(s.pkg_power_w);
        }

        // Try to parse as result
        if (is_bench_line(line)) {
            BenchResult r = parse_bench_line(line);
            if (r.valid) {
                std::lock_guard<std::mutex> lk(m_mutex);
                if (r.test.find("pp") == 0)
                    m_current.pp = r;
                else if (r.test.find("tg") == 0)
                    m_current.tg = r;
            }
        }
    }, m_cancel);

    // Finalize
    if (!m_temp_samples.empty())
        m_current.avg_temp_c = std::accumulate(
            m_temp_samples.begin(), m_temp_samples.end(), 0.0f)
            / m_temp_samples.size();
    if (!m_power_samples.empty())
        m_current.avg_power_w = std::accumulate(
            m_power_samples.begin(), m_power_samples.end(), 0.0f)
            / m_power_samples.size();

    m_current.complete = true;

    // Save to DB
    if (m_storage && m_current.pp.valid && m_current.tg.valid) {
        BenchRecord rec;
        rec.model_name  = params.model_path;
        rec.n_threads   = params.n_threads;
        rec.cpu_mask    = params.cpu_mask;
        rec.flash_attn  = params.flash_attn;
        rec.mmap_off    = params.mmap_off;
        rec.kv_type     = params.kv_type;
        rec.n_prompt    = params.n_prompt;
        rec.n_gen       = params.n_gen;
        rec.pp_tps      = m_current.pp.tps;
        rec.pp_std      = m_current.pp.std;
        rec.tg_tps      = m_current.tg.tps;
        rec.tg_std      = m_current.tg.std;
        rec.avg_temp_c  = m_current.avg_temp_c;
        rec.avg_power_w = m_current.avg_power_w;
        rec.notes       = m_notes_buf;
        m_storage->insert_bench(rec);
        m_history_dirty = true;
    }

    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_completed.push_back(m_current);
    }
    m_running = false;
}

// ---- Thread sweep ----------------------------------------------------------

void BenchmarkPanel::start_sweep(const DashboardConfig& cfg) {
    if (m_sweep_running || m_running) return;
    m_sweep_cancel   = false;
    m_sweep_results.clear();
    m_sweep_progress = 0;
    m_sweep_running  = true;
    m_sweep_thread   = std::thread(
        &BenchmarkPanel::sweep_thread_func, this, cfg);
}

void BenchmarkPanel::sweep_thread_func(DashboardConfig cfg) {
    for (int t : m_sweep_thread_counts) {
        if (m_sweep_cancel) break;

        cfg.n_threads   = t;
        cfg.use_taskset = false;
        BenchParams params = cfg.to_bench_params();
        std::string cmd = build_bench_command(params);

        BenchRun run;
        run.params = params;

        stream_command(cmd, [&](const std::string& line) {
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                m_log.push_back("[sweep t=" + std::to_string(t)
                                + "] " + line);
                if ((int)m_log.size() > MAX_LOG) m_log.pop_front();
            }
            if (is_bench_line(line)) {
                BenchResult r = parse_bench_line(line);
                if (r.valid) {
                    if (r.test.find("pp") == 0) run.pp = r;
                    else if (r.test.find("tg") == 0) run.tg = r;
                }
            }
        }, m_sweep_cancel);

        run.complete = true;
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_sweep_results.push_back({t, run});
        }
        m_sweep_progress++;
    }
    m_sweep_running = false;
}

// ---- Render ----------------------------------------------------------------

void BenchmarkPanel::render(const DashboardConfig& cfg) {
    render_controls(cfg);
    ImGui::Separator();

    if (ImGui::BeginTabBar("bench_tabs")) {
        if (ImGui::BeginTabItem("Live Output")) {
            render_live_output();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Current Result")) {
            render_current_result();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("History")) {
            render_history();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Comparison Plot")) {
            render_comparison_plot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Thread Sweep")) {
            render_thread_sweep(cfg);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void BenchmarkPanel::render_controls(const DashboardConfig& cfg) {
    ImGui::TextColored({0.4f, 0.9f, 0.4f, 1.0f}, "Benchmark Controls");

    if (m_running) {
        ImGui::BeginDisabled();
        ImGui::Button("Run Benchmark");
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) stop_run();
        ImGui::SameLine();
        ImGui::TextColored({1.0f,0.8f,0.0f,1.0f}, "Running...");
    } else {
        if (ImGui::Button("Run Benchmark")) start_run(cfg);
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(200.0f);
    ImGui::InputText("Notes##notes", m_notes_buf, sizeof(m_notes_buf));

    // Show current command
    BenchParams bp = cfg.to_bench_params();
    std::string cmd = build_bench_command(bp);
    ImGui::TextDisabled("%s", cmd.c_str());
}

void BenchmarkPanel::render_live_output() {
    std::lock_guard<std::mutex> lk(m_mutex);
    ImGui::BeginChild("log_child", ImVec2(0, 300), true);
    for (auto& line : m_log) {
        // Color result lines differently
        if (!line.empty() && line[0] == '|')
            ImGui::TextColored({0.3f, 0.9f, 0.9f, 1.0f}, "%s", line.c_str());
        else
            ImGui::TextUnformatted(line.c_str());
    }
    // Auto-scroll
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}

void BenchmarkPanel::render_current_result() {
    std::lock_guard<std::mutex> lk(m_mutex);

    if (!m_current.complete && !m_running) {
        ImGui::TextDisabled("No run yet. Press Run Benchmark.");
        return;
    }

    ImGui::Text("Threads: %d  |  KV: %s  |  FA: %s  |  mmap off: %s",
        m_current.params.n_threads,
        m_current.params.kv_type.c_str(),
        m_current.params.flash_attn ? "on" : "off",
        m_current.params.mmap_off   ? "yes" : "no");

    if (!m_current.params.cpu_mask.empty())
        ImGui::Text("Affinity: taskset -c %s",
                    m_current.params.cpu_mask.c_str());

    ImGui::Separator();

    // Big number display
    auto big_metric = [](const char* label, float val, float std,
                         ImVec4 col) {
        ImGui::TextColored(col, "%s", label);
        ImGui::SameLine();
        ImGui::TextColored(col, "%.2f", val);
        ImGui::SameLine();
        ImGui::TextDisabled("± %.2f t/s", std);
    };

    if (m_current.pp.valid)
        big_metric("PP:", m_current.pp.tps, m_current.pp.std,
                   {0.3f, 0.9f, 0.3f, 1.0f});
    else
        ImGui::TextDisabled("PP: waiting...");

    if (m_current.tg.valid)
        big_metric("TG:", m_current.tg.tps, m_current.tg.std,
                   {0.3f, 0.7f, 1.0f, 1.0f});
    else
        ImGui::TextDisabled("TG: waiting...");

    ImGui::Separator();
    ImGui::Text("Avg temp: %.1f C   Avg power: %.1f W",
                m_current.avg_temp_c, m_current.avg_power_w);

    // Bar chart of recent completed runs
    if (!m_completed.empty()) {
        ImGui::Separator();
        ImGui::TextColored({0.4f, 0.9f, 0.4f, 1.0f},
                           "Session Results (tg t/s)");
        if (ImPlot::BeginPlot("##session_tg",
                              ImVec2(-1, 180),
                              ImPlotFlags_NoLegend))
        {
            std::vector<float> tg_vals;
            std::vector<float> xs;
            for (int i = 0; i < (int)m_completed.size(); i++) {
                tg_vals.push_back(m_completed[i].tg.tps);
                xs.push_back((float)i);
            }
            ImPlot::SetupAxes("Run", "tg (t/s)");
            ImPlot::PlotBars("tg", tg_vals.data(), (int)tg_vals.size());
            ImPlot::EndPlot();
        }
    }
}

void BenchmarkPanel::render_history() {
    // Reload if dirty
    if (m_history_dirty && m_storage) {
        m_history = m_storage->get_all_bench();
        m_history_dirty = false;
    }

    if (ImGui::Button("Refresh")) m_history_dirty = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear All")) {
        if (m_storage) {
            m_storage->clear_all();
            m_history_dirty = true;
        }
    }

    if (m_history.empty()) {
        ImGui::TextDisabled("No records yet.");
        return;
    }

    if (ImGui::BeginTable("hist_table", 8,
        ImGuiTableFlags_Borders     |
        ImGuiTableFlags_RowBg       |
        ImGuiTableFlags_ScrollY     |
        ImGuiTableFlags_SortMulti,
        ImVec2(0, 300)))
    {
        ImGui::TableSetupColumn("ID");
        ImGui::TableSetupColumn("Time");
        ImGui::TableSetupColumn("Threads");
        ImGui::TableSetupColumn("KV");
        ImGui::TableSetupColumn("PP t/s");
        ImGui::TableSetupColumn("TG t/s");
        ImGui::TableSetupColumn("Temp C");
        ImGui::TableSetupColumn("Notes");
        ImGui::TableHeadersRow();

        for (auto& r : m_history) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            bool sel = (m_selected_history == (int)r.id);
            if (ImGui::Selectable(std::to_string(r.id).c_str(),
                                  sel,
                                  ImGuiSelectableFlags_SpanAllColumns))
                m_selected_history = (int)r.id;
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(r.timestamp.c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%d%s", r.n_threads,
                        r.cpu_mask.empty() ? ""
                        : (" [" + r.cpu_mask + "]").c_str());
            ImGui::TableSetColumnIndex(3);
            ImGui::TextUnformatted(r.kv_type.c_str());
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.2f ± %.2f", r.pp_tps, r.pp_std);
            ImGui::TableSetColumnIndex(5);
            // Color-code tg: green if good, yellow if mediocre
            float tg = r.tg_tps;
            ImVec4 col = tg > 25.0f ? ImVec4(0.3f,0.9f,0.3f,1.0f)
                       : tg > 15.0f ? ImVec4(0.9f,0.8f,0.2f,1.0f)
                                    : ImVec4(0.9f,0.3f,0.3f,1.0f);
            ImGui::TableSetColumnIndex(5);
            ImGui::TextColored(col, "%.2f ± %.2f", r.tg_tps, r.tg_std);
            ImGui::TableSetColumnIndex(6);
            ImGui::Text("%.1f", r.avg_temp_c);
            ImGui::TableSetColumnIndex(7);
            ImGui::TextUnformatted(r.notes.c_str());
        }
        ImGui::EndTable();
    }

    // Delete selected
    if (m_selected_history >= 0) {
        if (ImGui::Button("Delete Selected")) {
            if (m_storage) {
                m_storage->delete_bench(m_selected_history);
                m_selected_history = -1;
                m_history_dirty    = true;
            }
        }
    }
}

void BenchmarkPanel::render_comparison_plot() {
    if (m_history_dirty && m_storage) {
        m_history = m_storage->get_all_bench();
        m_history_dirty = false;
    }
    if (m_history.empty()) {
        ImGui::TextDisabled("No history to plot.");
        return;
    }

    // Build vectors for plotting: last 20 records reversed (oldest first)
    int n = std::min((int)m_history.size(), 20);
    std::vector<float> pp_vals(n), tg_vals(n),
                       pp_err(n),  tg_err(n), xs(n);
    for (int i = 0; i < n; i++) {
        auto& r  = m_history[n - 1 - i];
        xs[i]    = (float)i;
        pp_vals[i] = r.pp_tps;
        tg_vals[i] = r.tg_tps;
        pp_err[i]  = r.pp_std;
        tg_err[i]  = r.tg_std;
    }

    if (ImPlot::BeginPlot("Benchmark History", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Run (oldest→newest)", "t/s");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0,
            *std::max_element(pp_vals.begin(), pp_vals.end()) * 1.2f,
            ImGuiCond_Always);
        ImPlot::PlotErrorBars("PP ± std",
            xs.data(), pp_vals.data(), pp_err.data(), n);
        ImPlot::PlotLine("PP t/s", xs.data(), pp_vals.data(), n);
        ImPlot::PlotErrorBars("TG ± std",
            xs.data(), tg_vals.data(), tg_err.data(), n);
        ImPlot::PlotLine("TG t/s", xs.data(), tg_vals.data(), n);
        ImPlot::EndPlot();
    }

    // Throughput per watt
    ImGui::Separator();
    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "TG / Watt");
    if (ImPlot::BeginPlot("TG per Watt", ImVec2(-1, 200))) {
        std::vector<float> tpw(n);
        for (int i = 0; i < n; i++) {
            auto& r = m_history[n-1-i];
            tpw[i] = r.avg_power_w > 0.0f
                   ? r.tg_tps / r.avg_power_w : 0.0f;
        }
        ImPlot::SetupAxes("Run", "t/s/W");
        ImPlot::PlotBars("tg/W", xs.data(), tpw.data(), n, 0.67f);
        ImPlot::EndPlot();
    }
}

void BenchmarkPanel::render_thread_sweep(const DashboardConfig& cfg) {
    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Thread Count Sweep");
    ImGui::TextWrapped(
        "Runs llama-bench at t=1,2,4,6,8,10 sequentially "
        "to find the optimal thread count for tg.");

    if (m_sweep_running) {
        ImGui::TextColored({1.0f,0.8f,0.0f,1.0f},
            "Sweep running... (%d / %d)",
            m_sweep_progress,
            (int)m_sweep_thread_counts.size());
        if (ImGui::Button("Cancel Sweep")) {
            m_sweep_cancel = true;
        }
    } else {
        if (ImGui::Button("Start Thread Sweep")) start_sweep(cfg);
    }

    std::lock_guard<std::mutex> lk(m_mutex);

    if (m_sweep_results.empty()) {
        ImGui::TextDisabled("No sweep data yet.");
        return;
    }

    // Table
    if (ImGui::BeginTable("sweep_table", 3,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Threads");
        ImGui::TableSetupColumn("PP t/s");
        ImGui::TableSetupColumn("TG t/s");
        ImGui::TableHeadersRow();

        float best_tg = 0.0f;
        for (auto& [t, run] : m_sweep_results)
            best_tg = std::max(best_tg, run.tg.tps);

        for (auto& [t, run] : m_sweep_results) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%d", t);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%.2f ± %.2f", run.pp.tps, run.pp.std);
            ImGui::TableSetColumnIndex(2);
            bool is_best = (run.tg.tps == best_tg && best_tg > 0.0f);
            if (is_best)
                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg,
                    ImGui::ColorConvertFloat4ToU32(
                        {0.1f,0.4f,0.1f,1.0f}));
            ImGui::Text("%.2f ± %.2f", run.tg.tps, run.tg.std);
        }
        ImGui::EndTable();
    }

    // Plot
    if (m_sweep_results.size() >= 2) {
        std::vector<float> ts, pp_v, tg_v;
        for (auto& [t, run] : m_sweep_results) {
            ts.push_back((float)t);
            pp_v.push_back(run.pp.tps);
            tg_v.push_back(run.tg.tps);
        }
        if (ImPlot::BeginPlot("Throughput vs Threads", ImVec2(-1, 250))) {
            ImPlot::SetupAxes("Threads", "t/s");
            ImPlot::SetupAxisLimits(ImAxis_X1,ts.front(), ts.back(), ImGuiCond_Always);
            ImPlot::PlotLine("PP t/s", ts.data(), pp_v.data(),(int)ts.size());
            ImPlot::PlotLine("TG t/s", ts.data(), tg_v.data(),(int)ts.size());
            ImPlot::PlotScatter("PP t/s", ts.data(), pp_v.data(),(int)ts.size());
            ImPlot::PlotScatter("TG t/s", ts.data(), tg_v.data(),(int)ts.size());
            ImPlot::EndPlot();
        }
    }
}