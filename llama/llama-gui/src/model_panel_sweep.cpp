#include "model_panel_sweep.h"
#include "imgui.h"
#include "implot.h"
#include "process_utils.h"
#include <algorithm>
#include <sstream>
#include <numeric>
#include <sys/resource.h>  // getrusage for peak RSS

ModelSweepPanel::ModelSweepPanel() {}

ModelSweepPanel::~ModelSweepPanel() {
    m_cancel = true;
    if (m_thread.joinable()) m_thread.join();
}

// Reuses BenchmarkPanel's parse_bench_line logic conceptually — since that's
// a private method on BenchmarkPanel, this is a local copy. Consider lifting
// parse_bench_line/is_bench_line to a free function in process_utils if you
// want a single source of truth (recommend doing this — see note below).
#include <sstream>

bool is_bench_line(const std::string& line) {
    return !line.empty() && line[0] == '|';
}

BenchResult parse_bench_line(const std::string& line) {
    BenchResult r;
    if (line.empty() || line[0] != '|') return r;

    std::vector<std::string> cols;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, '|')) {
        size_t s = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        cols.push_back(s != std::string::npos ? tok.substr(s, e - s + 1) : "");
    }

    if (cols.size() < 12) return r;

    std::string test_col = cols[10];
    std::string tps_col  = cols[11];
    if (test_col.empty() || tps_col.empty()) return r;
    if (test_col.find("test") != std::string::npos) return r; // header row

    float tps = 0.0f, std_val = 0.0f;
    size_t pm = tps_col.find("±");
    if (pm != std::string::npos) {
        try {
            tps     = std::stof(tps_col.substr(0, pm));
            std_val = std::stof(tps_col.substr(pm + 3)); // ± is 3 bytes UTF-8
        } catch (...) { return r; }
    } else {
        try { tps = std::stof(tps_col); } catch (...) { return r; }
    }

    r.test  = test_col;
    r.tps   = tps;
    r.std   = std_val;
    r.valid = true;
    return r;
}

void ModelSweepPanel::start_sweep(const DashboardConfig& cfg) {
    if (m_running || m_models.empty()) return;
    m_cancel = false;
    m_results.clear();
    m_progress = 0;
    m_running = true;
    m_thread = std::thread(&ModelSweepPanel::sweep_thread_func, this, cfg);
}

void ModelSweepPanel::sweep_thread_func(DashboardConfig cfg) {
    for (auto& spec : m_models) {
        if (m_cancel) break;

        BenchParams params = cfg.to_bench_params();
        params.model_path = spec.path;
        params.kv_type = spec.quant == "F16" ? "f16" : "q8_0";
        std::string cmd = build_bench_command(params);

        ModelSweepRun run;
        run.spec = spec;

        // All of these need to be declared HERE, before the lambda, so the
        // lambda can capture them by reference and the code after
        // stream_command() can read the final accumulated values.
        std::vector<float> temps, powers;
        std::vector<float> ram_used_samples, swap_samples;
        float peak_ram_mb = 0.0f;

        int exit_code = stream_command(cmd, [&](const std::string& line) {
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                m_log.push_back("[" + spec.label + "] " + line);
                if ((int)m_log.size() > MAX_LOG) m_log.pop_front();
            }
            if (m_telemetry) {
                auto s = m_telemetry->latest();
                temps.push_back(s.pkg_temp_c);
                powers.push_back(s.pkg_power_w);
            }

            MemSnapshot mem = read_meminfo();
            ram_used_samples.push_back(mem.used_mb);
            swap_samples.push_back(mem.swap_used_mb);
            peak_ram_mb = std::max(peak_ram_mb, mem.used_mb);

            if (is_bench_line(line)) {
                BenchResult r = parse_bench_line(line);
                if (r.valid) {
                    if (r.test.find("pp") == 0) run.pp = r;
                    else if (r.test.find("tg") == 0) run.tg = r;
                }
            }
        }, m_cancel);

        run.oom = (exit_code != 0 && !run.tg.valid);

        if (!temps.empty())
            run.avg_temp_c = std::accumulate(temps.begin(), temps.end(), 0.0f) / temps.size();
        if (!powers.empty())
            run.avg_power_w = std::accumulate(powers.begin(), powers.end(), 0.0f) / powers.size();

        run.peak_rss_mb  = peak_ram_mb;
        run.peak_swap_mb = swap_samples.empty() ? 0.0f
            : *std::max_element(swap_samples.begin(), swap_samples.end());

        run.complete = true;


        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_results.push_back(run);
        }
        m_progress++;

        if (m_storage && run.tg.valid) {
            BenchRecord rec;
            rec.model_name   = spec.path;
            rec.model_size_b = spec.size_b;
            rec.quant        = spec.quant;
            rec.n_threads    = params.n_threads;
            rec.cpu_mask     = params.cpu_mask;
            rec.flash_attn   = params.flash_attn;
            rec.mmap_off     = params.mmap_off;
            rec.kv_type      = params.kv_type;
            rec.n_prompt     = params.n_prompt;
            rec.n_gen        = params.n_gen;
            rec.pp_tps       = run.pp.tps;
            rec.pp_std       = run.pp.std;
            rec.tg_tps       = run.tg.tps;
            rec.tg_std       = run.tg.std;
            rec.avg_temp_c   = run.avg_temp_c;
            rec.avg_power_w  = run.avg_power_w;
            rec.peak_rss_mb  = run.peak_rss_mb;
            rec.notes        = run.oom ? "OOM/crash suspected" : "";
            m_storage->insert_bench(rec);
        }

        {
            std::string perf_cmd = "perf stat --no-scale -d -d -d " + cmd + " 2>&1";

            std::string perf_output;
            stream_command(perf_cmd, [&](const std::string& line) {
                perf_output += line + "\n";
                {
                    std::lock_guard<std::mutex> lk(m_mutex);
                    m_log.push_back("[perf " + spec.label + "] " + line);
                    if ((int)m_log.size() > MAX_LOG) m_log.pop_front();
                }
            }, m_cancel);

            PerfCaptureResult pr;
            pr.label = spec.label;
            pr.raw_output = perf_output;
            std::lock_guard<std::mutex> lk(m_mutex);
            m_perf_results.push_back(pr);
        }
    }
    m_running = false;
}

void ModelSweepPanel::render(const DashboardConfig& cfg) {
    if (ImGui::BeginTabBar("model_sweep_tabs")) {
        if (ImGui::BeginTabItem("Sweep")) {
            render_controls(cfg);
            ImGui::Separator();
            render_model_list();
            ImGui::Separator();

            std::lock_guard<std::mutex> lk(m_mutex);
            if (!m_log.empty()) {
                ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Live Output");
                ImGui::BeginChild("sweep_log", ImVec2(0, 150), true);
                for (auto& line : m_log) ImGui::TextUnformatted(line.c_str());
                if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                    ImGui::SetScrollHereY(1.0f);
                ImGui::EndChild();
                ImGui::Separator();
            }

            render_results_table();
            render_scaling_plot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Perf Stat")) {
            render_perf_tab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void ModelSweepPanel::render_controls(const DashboardConfig& cfg) {
    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Model Size Scaling Sweep");
    ImGui::TextWrapped(
        "Runs llama-bench sequentially across the configured model ladder. "
        "Uses the current thread/KV/flash-attn settings from Configuration.");

    if (m_running) {
        ImGui::TextColored({1.0f,0.8f,0.0f,1.0f},
            "Running %d / %d: %s", m_progress, (int)m_models.size(),
            m_progress < (int)m_models.size() ? m_models[m_progress].label.c_str() : "");
        if (ImGui::Button("Cancel Sweep")) m_cancel = true;
    } else {
        if (ImGui::Button("Start Size Sweep")) start_sweep(cfg);
    }
}

void ModelSweepPanel::render_model_list() {
    if (ImGui::TreeNode("Model Ladder (edit before running)")) {
        for (int i = 0; i < (int)m_models.size(); i++) {
            ImGui::PushID(i);
            ImGui::Text("%s  %.0fB  %s", m_models[i].label.c_str(),
                        m_models[i].size_b, m_models[i].quant.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton("Remove")) {
                m_models.erase(m_models.begin() + i);
                ImGui::PopID();
                break; // avoid iterating a mutated vector
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::SetNextItemWidth(80);
        ImGui::InputText("Label", m_new_model_label, sizeof(m_new_model_label));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("SizeB", &m_new_model_size);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputText("Quant", m_new_model_quant, sizeof(m_new_model_quant));
        ImGui::InputText("Path", m_new_model_path, sizeof(m_new_model_path));
        if (ImGui::Button("Add to ladder")) {
            if (strlen(m_new_model_path) > 0) {
                m_models.push_back({m_new_model_label, m_new_model_path,
                                     m_new_model_size, m_new_model_quant});
            }
        }
        ImGui::TreePop();
    }
}

void ModelSweepPanel::render_results_table() {
    if (m_results.empty()) {
        ImGui::TextDisabled("No sweep results yet.");
        return;
    }
    if (ImGui::BeginTable("size_sweep_table", 8,   // was 7
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("Model");
        ImGui::TableSetupColumn("Quant");
        ImGui::TableSetupColumn("PP t/s");
        ImGui::TableSetupColumn("TG t/s");
        ImGui::TableSetupColumn("Peak RAM (MB)");
        ImGui::TableSetupColumn("Peak Swap (MB)");   // NEW
        ImGui::TableSetupColumn("Temp C");
        ImGui::TableSetupColumn("Status");
        ImGui::TableHeadersRow();

        for (auto& r : m_results) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s (%.0fB)", r.spec.label.c_str(), r.spec.size_b);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(r.spec.quant.c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.2f", r.pp.tps);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.2f", r.tg.tps);
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.0f", r.peak_rss_mb);
            ImGui::TableSetColumnIndex(5);
            // Highlight red if swap was actually used — that's the real "thrashing" signal
            if (r.peak_swap_mb > 10.0f)
                ImGui::TextColored({0.9f,0.3f,0.3f,1.0f}, "%.0f", r.peak_swap_mb);
            else
                ImGui::Text("%.0f", r.peak_swap_mb);
            ImGui::TableSetColumnIndex(6);
            ImGui::Text("%.1f", r.avg_temp_c);
            ImGui::TableSetColumnIndex(7);
            if (r.oom)
                ImGui::TextColored({0.9f,0.3f,0.3f,1.0f}, "OOM/crash");
            else
                ImGui::TextColored({0.3f,0.9f,0.3f,1.0f}, "OK");
        }
        ImGui::EndTable();
    }
}

void ModelSweepPanel::render_scaling_plot() {
    if (m_results.size() < 2) return;
    std::vector<float> sizes, tg_vals, pp_vals;
    for (auto& r : m_results) {
        if (r.oom) continue; // skip failed runs from the curve
        sizes.push_back(r.spec.size_b);
        tg_vals.push_back(r.tg.tps);
        pp_vals.push_back(r.pp.tps);
    }
    if (sizes.size() < 2) return;

    ImGui::TextColored({0.4f,0.9f,0.4f,1.0f}, "Throughput vs Model Size");
    if (ImPlot::BeginPlot("Size Scaling", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Model Size (B params)", "t/s");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::PlotLine("PP t/s", sizes.data(), pp_vals.data(), (int)sizes.size());
        ImPlot::PlotScatter("PP t/s", sizes.data(), pp_vals.data(), (int)sizes.size());
        ImPlot::PlotLine("TG t/s", sizes.data(), tg_vals.data(), (int)sizes.size());
        ImPlot::PlotScatter("TG t/s", sizes.data(), tg_vals.data(), (int)sizes.size());
        ImPlot::EndPlot();
    }
}

void ModelSweepPanel::render_perf_tab() {
    std::lock_guard<std::mutex> lk(m_mutex);
    if (m_perf_results.empty()) {
        ImGui::TextDisabled("No perf stat data yet. Run the sweep first.");
        return;
    }

    if (ImGui::BeginTabBar("perf_model_tabs")) {
        for (auto& pr : m_perf_results) {
            if (ImGui::BeginTabItem(pr.label.c_str())) {
                ImGui::BeginChild(("perf_child_" + pr.label).c_str(),
                                   ImVec2(0, -1), true);   // taller — was 300
                ImGui::TextUnformatted(pr.raw_output.c_str());
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
}