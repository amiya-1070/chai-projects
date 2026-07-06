#pragma once
#include <string>
#include <vector>
#include "process_utils.h"

// All user-configurable settings, shared across panels
struct DashboardConfig {
    // Paths
    char model_path[512]    = "";
    char llama_bench[512]   = "";
    char llama_cli[512]     = "";
    char db_path[512]       = "";

    // Threading
    int  n_threads          = 8;
    bool use_taskset        = false;
    char cpu_mask[64]       = "0-3";  // e.g. "0-3", "4-11", "0,2,4-7"

    // Flags
    bool flash_attn         = true;
    bool mmap_off           = true;

    // KV cache
    int  kv_type_idx        = 1;      // 0=f16, 1=q8_0, 2=q4_0
    static constexpr const char* KV_TYPES[] = {"f16", "q8_0", "q4_0"};

    // Benchmark params
    int  n_prompt           = 512;
    int  n_gen              = 200;
    int  n_repeat           = 3;
    int  delay_s            = 0;

    // Inference params
    int  n_predict          = 512;

    // Derived helpers
    BenchParams  to_bench_params()  const;
    InferParams  to_infer_params()  const;

    char base_gguf_path[512] = "";
    char finetuned_gguf_path[512] = "";
    char kl_helper_script[512]     = "";
    char base_transformers_path[512] = "";
    char finetuned_transformers_path[512] = "";

    char finetuned_model_path[512] = "";   // kept for reference, not used for spawn
    char base_model_hf_id[256]     = "meta-llama/Llama-3.2-1B-Instruct";
    char finetuned_model_hf_id[256]= "coconutpdf/genomics-llama-1b";
};

class ConfigPanel {
public:
    ConfigPanel();

    // Renders the config panel. Call inside ImGui window.
    void render();

    DashboardConfig& config() { return m_cfg; }
    const DashboardConfig& config() const { return m_cfg; }

    // Returns true if config has changed since last call
    bool changed();

private:
    void render_paths();
    void render_threading();
    void render_flags();
    void render_bench_params();
    void detect_defaults();

    DashboardConfig m_cfg;
    bool            m_changed   = false;
    bool            m_detected  = false;

    // CPU topology info for display
    struct CoreInfo {
        int  logical_id  = 0;
        float max_freq_mhz = 0.0f;
        bool is_p_core   = false;
    };
    std::vector<CoreInfo> m_cores;
    void load_core_topology();
};