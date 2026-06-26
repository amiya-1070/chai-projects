#pragma once
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstdint>
#include <string>

// One snapshot of all telemetry at a point in time
struct TelemetrySnapshot {
    double timestamp_s      = 0.0;

    // Per-CPU frequency (MHz), indexed by logical CPU id
    std::vector<float> cpu_freq_mhz;

    // Package-level
    float pkg_temp_c        = 0.0f;
    float pkg_power_w       = 0.0f;

    // Memory
    float ram_used_gib      = 0.0f;
    float ram_total_gib     = 0.0f;

    // Perf counters (aggregated across all threads)
    float ipc               = 0.0f;
    float cache_miss_rate   = 0.0f;  // LLC miss / LLC ref
    float dtlb_miss_rate    = 0.0f;
};

// Rolling window of telemetry history
struct TelemetryHistory {
    static constexpr int MAX_SAMPLES = 600; // 2 min at 200ms

    std::deque<float> timestamps;
    std::deque<float> pkg_temp;
    std::deque<float> pkg_power;
    std::deque<float> ram_used;
    std::deque<float> avg_freq;   // average across active CPUs
    std::deque<float> ipc;
    std::deque<float> cache_miss_rate;

    void push(const TelemetrySnapshot& s);
    void get_arrays(
        std::vector<float>& out_t,
        std::vector<float>& out_temp,
        std::vector<float>& out_power,
        std::vector<float>& out_ram,
        std::vector<float>& out_freq,
        std::vector<float>& out_ipc,
        std::vector<float>& out_cache_miss
    ) const;
};

class TelemetryCollector {
public:
    TelemetryCollector();
    ~TelemetryCollector();

    void start();
    void stop();

    // Thread-safe snapshot of the latest reading
    TelemetrySnapshot latest() const;

    // Thread-safe copy of history for plotting
    TelemetryHistory history() const;

private:
    void poll_loop();
    TelemetrySnapshot collect_once();

    float read_pkg_temp();
    float read_pkg_power();
    float read_cpu_freq(int cpu_id);
    void  read_meminfo(float& used_gib, float& total_gib);

    std::thread             m_thread;
    std::atomic<bool>       m_running{false};
    mutable std::mutex      m_mutex;
    TelemetrySnapshot       m_latest;
    TelemetryHistory        m_history;

    int m_n_cpus = 12;

    // hwmon paths, resolved at init
    std::string m_temp_path;
    std::string m_power_path;

    double m_start_time = 0.0;
};