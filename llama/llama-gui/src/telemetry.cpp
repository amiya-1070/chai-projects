#include "telemetry.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace fs = std::filesystem;

static double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---- TelemetryHistory -------------------------------------------------------

void TelemetryHistory::push(const TelemetrySnapshot& s) {
    auto push_trim = [&](std::deque<float>& d, float v) {
        d.push_back(v);
        if ((int)d.size() > MAX_SAMPLES) d.pop_front();
    };
    float t = (float)s.timestamp_s;
    push_trim(timestamps,       t);
    push_trim(pkg_temp,         s.pkg_temp_c);
    push_trim(pkg_power,        s.pkg_power_w);
    push_trim(ram_used,         s.ram_used_gib);
    push_trim(ipc,              s.ipc);
    push_trim(cache_miss_rate,  s.cache_miss_rate);

    float avg = 0.0f;
    if (!s.cpu_freq_mhz.empty()) {
        avg = std::accumulate(s.cpu_freq_mhz.begin(),
                              s.cpu_freq_mhz.end(), 0.0f)
              / (float)s.cpu_freq_mhz.size();
    }
    push_trim(avg_freq, avg);
}

void TelemetryHistory::get_arrays(
    std::vector<float>& out_t,
    std::vector<float>& out_temp,
    std::vector<float>& out_power,
    std::vector<float>& out_ram,
    std::vector<float>& out_freq,
    std::vector<float>& out_ipc,
    std::vector<float>& out_cache_miss) const
{
    out_t          = {timestamps.begin(),      timestamps.end()};
    out_temp       = {pkg_temp.begin(),        pkg_temp.end()};
    out_power      = {pkg_power.begin(),       pkg_power.end()};
    out_ram        = {ram_used.begin(),        ram_used.end()};
    out_freq       = {avg_freq.begin(),        avg_freq.end()};
    out_ipc        = {ipc.begin(),             ipc.end()};
    out_cache_miss = {cache_miss_rate.begin(), cache_miss_rate.end()};
}

// ---- TelemetryCollector -----------------------------------------------------

TelemetryCollector::TelemetryCollector() {
    m_start_time = now_seconds();

    // Resolve hwmon paths for package temperature and power
    for (auto& entry : fs::directory_iterator("/sys/class/hwmon")) {
        std::string name_path = entry.path().string() + "/name";
        std::ifstream f(name_path);
        std::string name;
        std::getline(f, name);
        if (name == "coretemp" || name == "acpitz") {
            // look for temp1_input (package temp)
            std::string tp = entry.path().string() + "/temp1_input";
            if (fs::exists(tp)) m_temp_path = tp;
        }
        if (name == "intel_rapl_mmio" || name == "intel_rapl") {
            std::string pp = entry.path().string() + "/power1_average";
            if (fs::exists(pp)) m_power_path = pp;
        }
    }
}

TelemetryCollector::~TelemetryCollector() {
    stop();
}

void TelemetryCollector::start() {
    m_running = true;
    m_thread = std::thread(&TelemetryCollector::poll_loop, this);
}

void TelemetryCollector::stop() {
    m_running = false;
    if (m_thread.joinable()) m_thread.join();
}

TelemetrySnapshot TelemetryCollector::latest() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_latest;
}

TelemetryHistory TelemetryCollector::history() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_history;
}

void TelemetryCollector::poll_loop() {
    while (m_running.load()) {
        TelemetrySnapshot s = collect_once();
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_latest = s;
            m_history.push(s);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

TelemetrySnapshot TelemetryCollector::collect_once() {
    TelemetrySnapshot s;
    s.timestamp_s = now_seconds() - m_start_time;

    s.cpu_freq_mhz.resize(m_n_cpus);
    for (int i = 0; i < m_n_cpus; i++)
        s.cpu_freq_mhz[i] = read_cpu_freq(i);

    s.pkg_temp_c  = read_pkg_temp();
    s.pkg_power_w = read_pkg_power();
    read_meminfo(s.ram_used_gib, s.ram_total_gib);

    // IPC and cache miss rate: read from perf via sysfs if available,
    // otherwise leave at 0 (perf_event_open requires privileges)
    // For now poll from /proc/stat as a placeholder;
    // full perf_event_open integration goes in a follow-up
    s.ipc            = 0.0f;
    s.cache_miss_rate = 0.0f;

    return s;
}

float TelemetryCollector::read_pkg_temp() {
    if (m_temp_path.empty()) return 0.0f;
    std::ifstream f(m_temp_path);
    int val = 0;
    f >> val;
    return val / 1000.0f; // millidegrees to degrees
}

float TelemetryCollector::read_pkg_power() {
    if (m_power_path.empty()) return 0.0f;
    std::ifstream f(m_power_path);
    long val = 0;
    f >> val;
    return val / 1e6f; // microwatts to watts
}

float TelemetryCollector::read_cpu_freq(int cpu_id) {
    std::string path = "/sys/devices/system/cpu/cpu"
                     + std::to_string(cpu_id)
                     + "/cpufreq/scaling_cur_freq";
    std::ifstream f(path);
    if (!f.is_open()) return 0.0f;
    long val = 0;
    f >> val;
    return val / 1000.0f; // kHz to MHz
}

void TelemetryCollector::read_meminfo(float& used_gib, float& total_gib) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    long total_kb = 0, available_kb = 0;
    while (std::getline(f, line)) {
        if (line.rfind("MemTotal:", 0) == 0)
            std::sscanf(line.c_str(), "MemTotal: %ld kB", &total_kb);
        if (line.rfind("MemAvailable:", 0) == 0)
            std::sscanf(line.c_str(), "MemAvailable: %ld kB", &available_kb);
    }
    total_gib = total_kb / (1024.0f * 1024.0f);
    used_gib  = (total_kb - available_kb) / (1024.0f * 1024.0f);
}