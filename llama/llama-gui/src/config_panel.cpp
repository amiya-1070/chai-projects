#include "config_panel.h"
#include "imgui.h"
#include "process_utils.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace fs = std::filesystem;

// ---- DashboardConfig helpers -----------------------------------------------

BenchParams DashboardConfig::to_bench_params() const {
    BenchParams p;
    p.model_path  = base_gguf_path;
    p.llama_bench = llama_bench;
    p.n_threads   = n_threads;
    p.n_prompt    = n_prompt;
    p.n_gen       = n_gen;
    p.n_repeat    = n_repeat;
    p.delay       = delay_s;
    p.flash_attn  = flash_attn;
    p.mmap_off    = mmap_off;
    p.kv_type     = KV_TYPES[kv_type_idx];
    p.cpu_mask    = use_taskset ? cpu_mask : "";
    return p;
}

InferParams DashboardConfig::to_infer_params() const {
    InferParams p;
    
    p.model_path = base_gguf_path;
    p.llama_cli   = llama_cli;
    p.n_threads   = n_threads;
    p.n_predict   = n_predict;
    p.flash_attn  = flash_attn;
    p.mmap_off    = mmap_off;
    p.kv_type     = KV_TYPES[kv_type_idx];
    p.cpu_mask    = use_taskset ? cpu_mask : "";
    return p;
}

// ---- ConfigPanel -----------------------------------------------------------

ConfigPanel::ConfigPanel() {
    load_core_topology();
    detect_defaults();
}

bool ConfigPanel::changed() {
    bool c = m_changed;
    m_changed = false;
    return c;
}

void ConfigPanel::load_core_topology() {
    m_cores.clear();
    float global_max = 0.0f;

    for (int i = 0; i < 16; i++) {
        std::string path = "/sys/devices/system/cpu/cpu"
                         + std::to_string(i)
                         + "/cpufreq/cpuinfo_max_freq";
        std::ifstream f(path);
        if (!f.is_open()) break;
        long val = 0;
        f >> val;
        CoreInfo c;
        c.logical_id    = i;
        c.max_freq_mhz  = val / 1000.0f;
        if (c.max_freq_mhz > global_max) global_max = c.max_freq_mhz;
        m_cores.push_back(c);
    }
    // Mark P-cores: max freq >= 95% of global max
    for (auto& c : m_cores)
        c.is_p_core = (c.max_freq_mhz >= global_max * 0.95f);
}

void ConfigPanel::detect_defaults() {
    if (m_detected) return;
    m_detected = true;

    // Try to find llama-bench and llama-cli relative to common build paths
    std::vector<std::string> search = {
        "/media/amiyaun/New Volume/chai-projects/llama/llama.cpp/build/bin/llama-bench",
        "/media/amiyaun/New Volume/chai-projects/llama/llama.cpp/build/bin/llama-cli",
        "./build/bin/llama-bench",
        "./build/bin/llama-cli",
    };

    for (auto& p : search) {
        if (fs::exists(p)) {
            if (p.find("llama-bench") != std::string::npos
                && m_cfg.llama_bench[0] == '\0')
                std::strncpy(m_cfg.llama_bench, p.c_str(),
                             sizeof(m_cfg.llama_bench) - 1);
            if (p.find("llama-cli") != std::string::npos
                && m_cfg.llama_cli[0] == '\0')
                std::strncpy(m_cfg.llama_cli, p.c_str(),
                             sizeof(m_cfg.llama_cli) - 1);
        }
    }

    // Default DB path
    if (m_cfg.db_path[0] == '\0')
        std::strncpy(m_cfg.db_path,
                     "/media/amiyaun/New Volume/chai-projects/llama/llm-benchmark/bench_history.db",
                     sizeof(m_cfg.db_path) - 1);

    // Default model
    if (m_cfg.base_gguf_path[0] == '\0') {
        std::string mp =
            "/media/amiyaun/New Volume/chai-projects/llama/llm-benchmark/"
            "Llama-3.2-1B-Instruct-f16.gguf";
        if (fs::exists(mp))
            std::strncpy(m_cfg.base_gguf_path, mp.c_str(),sizeof(m_cfg.base_gguf_path) - 1);
    }

        // KL helper script
    if (m_cfg.kl_helper_script[0] == '\0') {
        std::string p =
            "/media/amiyaun/New Volume/chai-projects/llama/llama-gui/kl_helper.py";
        if (fs::exists(p))
            std::strncpy(m_cfg.kl_helper_script, p.c_str(),
                        sizeof(m_cfg.kl_helper_script) - 1);
    }

    // Base HF checkpoint directory
    if (m_cfg.base_transformers_path[0] == '\0') {
        std::string p =
            "/media/amiyaun/New Volume/chai-projects/llama/models/llama-3.2-1B-Instruct";
        if (fs::exists(p))
            std::strncpy(m_cfg.base_transformers_path, p.c_str(),
                        sizeof(m_cfg.base_transformers_path) - 1);
    }

    // Finetuned checkpoint directory
    if (m_cfg.finetuned_transformers_path[0] == '\0') {
        std::string p =
            "/media/amiyaun/New Volume/chai-projects/llama/models/genomics-llama-1b";
        if (fs::exists(p))
            std::strncpy(m_cfg.finetuned_transformers_path, p.c_str(),
                        sizeof(m_cfg.finetuned_transformers_path) - 1);
    }
}

// ---- Render ----------------------------------------------------------------

void ConfigPanel::render() {
    render_paths();
    ImGui::Separator();
    render_threading();
    ImGui::Separator();
    render_flags();
    ImGui::Separator();
    render_bench_params();
}

void ConfigPanel::render_paths() {
    ImGui::TextColored({0.88f, 0.76f, 1.0f, 1.0f}, "Paths");

    auto path_input = [&](const char* label, char* buf, size_t bufsz) {
        if (ImGui::InputText(label, buf, bufsz))
            m_changed = true;
        // drag-and-drop hint
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Enter full path");
    };

    path_input("Base model (.gguf)##basegguf",
           m_cfg.base_gguf_path, sizeof(m_cfg.base_gguf_path));

    path_input("Finetuned model (.gguf)##ftgguf",
            m_cfg.finetuned_gguf_path, sizeof(m_cfg.finetuned_gguf_path));

    path_input("llama-bench##bench",
            m_cfg.llama_bench, sizeof(m_cfg.llama_bench));

    path_input("llama-cli##cli",
            m_cfg.llama_cli, sizeof(m_cfg.llama_cli));

    path_input("Database (.db)##db",
            m_cfg.db_path, sizeof(m_cfg.db_path));


    path_input("KL helper script##klscript",
           m_cfg.kl_helper_script,
           sizeof(m_cfg.kl_helper_script));
    path_input("Base model HF ID##basehf",
            m_cfg.base_model_hf_id,
            sizeof(m_cfg.base_model_hf_id));
    path_input("Finetuned model HF ID##fthf",
            m_cfg.finetuned_model_hf_id,
            sizeof(m_cfg.finetuned_model_hf_id));
    

    // SHA256 verify button
    ImGui::SameLine();
    if (ImGui::SmallButton("Verify SHA256")) {
        std::string cmd = std::string("sha256sum \"")
                        + m_cfg.base_gguf_path + "\"";
        std::string result = run_command(cmd);
        // result stored in a static buffer for display
        static char sha_result[256] = "";
        std::strncpy(sha_result, result.c_str(), sizeof(sha_result) - 1);
        ImGui::OpenPopup("sha256_popup");
    }
    if (ImGui::BeginPopup("sha256_popup")) {
        static char sha_result[256] = "";
        ImGui::TextWrapped("%s", sha_result);
        ImGui::EndPopup();
    }
}

void ConfigPanel::render_threading() {
    ImGui::TextColored({0.88f, 0.76f, 1.0f, 1.0f}, "Threading");

    // CPU topology table
    if (ImGui::TreeNode("CPU Topology")) {
        if (ImGui::BeginTable("cpu_topo", 3,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("CPU");
            ImGui::TableSetupColumn("Max MHz");
            ImGui::TableSetupColumn("Type");
            ImGui::TableHeadersRow();
            for (auto& c : m_cores) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%d", c.logical_id);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.0f", c.max_freq_mhz);
                ImGui::TableSetColumnIndex(2);
                if (c.is_p_core)
                    ImGui::TextColored({0.82f,0.68f,1.0f,1.0f}, "P-core");
                else
                    ImGui::TextColored({0.90f,0.84f,0.2f,1.0f}, "E-core");
            }
            ImGui::EndTable();
        }
        ImGui::TreePop();
    }

    if (ImGui::SliderInt("Threads##t", &m_cfg.n_threads, 1, 12))
        m_changed = true;

    // Quick presets
    ImGui::Text("Presets:");
    ImGui::SameLine();
    if (ImGui::SmallButton("P-only (t=4)")) {
        m_cfg.n_threads   = 4;
        m_cfg.use_taskset = true;
        std::strncpy(m_cfg.cpu_mask, "0-3", sizeof(m_cfg.cpu_mask)-1);
        m_changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("E-only (t=8)")) {
        m_cfg.n_threads   = 8;
        m_cfg.use_taskset = true;
        std::strncpy(m_cfg.cpu_mask, "4-11", sizeof(m_cfg.cpu_mask)-1);
        m_changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("All (t=8)")) {
        m_cfg.n_threads   = 8;
        m_cfg.use_taskset = false;
        m_changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("All (t=10)")) {
        m_cfg.n_threads   = 10;
        m_cfg.use_taskset = false;
        m_changed = true;
    }

    if (ImGui::Checkbox("Use taskset (CPU affinity mask)", &m_cfg.use_taskset))
        m_changed = true;

    if (m_cfg.use_taskset) {
        ImGui::SameLine();
        if (ImGui::InputText("##mask", m_cfg.cpu_mask,
                             sizeof(m_cfg.cpu_mask)))
            m_changed = true;
        ImGui::SetItemTooltip(
            "Examples: 0-3 (P-cores), 4-11 (E-cores), 0,2,4-7");
    }
}

void ConfigPanel::render_flags() {
    ImGui::TextColored({0.88f, 0.76f, 1.0f, 1.0f}, "Runtime Flags");

    if (ImGui::Checkbox("Flash Attention (-fa on)", &m_cfg.flash_attn))
        m_changed = true;
    ImGui::SameLine();
    if (ImGui::Checkbox("Disable mmap (-mmp 0)", &m_cfg.mmap_off))
        m_changed = true;

    ImGui::Text("KV Cache Type:");
    ImGui::SameLine();
    if (ImGui::Combo("##kv", &m_cfg.kv_type_idx,
                     DashboardConfig::KV_TYPES, 3))
        m_changed = true;
}

void ConfigPanel::render_bench_params() {
    ImGui::TextColored({0.88f, 0.76f, 1.0f, 1.0f}, "Benchmark Parameters");

    if (ImGui::InputInt("Prompt tokens (-p)", &m_cfg.n_prompt))
        m_changed = true;
    if (ImGui::InputInt("Gen tokens (-n)",    &m_cfg.n_gen))
        m_changed = true;
    if (ImGui::InputInt("Repeats (-r)",       &m_cfg.n_repeat))
        m_changed = true;
    if (ImGui::InputInt("Delay (s)",          &m_cfg.delay_s))
        m_changed = true;

    ImGui::Separator();
    
    ImGui::TextColored({0.88f, 0.76f, 1.0f, 1.0f}, "Inference Parameters");
    if (ImGui::InputInt("Max tokens (n_predict)", &m_cfg.n_predict))
        m_changed = true;

    // Show the final command that will be run
    if (ImGui::TreeNode("Preview command")) {
        BenchParams bp = m_cfg.to_bench_params();
        std::string cmd = build_bench_command(bp);
        ImGui::TextWrapped("%s", cmd.c_str());
        ImGui::TreePop();
    }
}