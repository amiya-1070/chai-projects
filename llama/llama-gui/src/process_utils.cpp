#include "process_utils.h"
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <array>

std::string run_command(const std::string& cmd) {
    std::string result;
    std::array<char, 4096> buf;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    while (fgets(buf.data(), buf.size(), pipe)) {
        result += buf.data();
    }
    pclose(pipe);
    return result;
}

int stream_command(
    const std::string& cmd,
    std::function<void(const std::string&)> on_line,
    std::atomic<bool>& cancel)
{
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return -1;

    std::array<char, 4096> buf;
    while (!cancel.load() && fgets(buf.data(), buf.size(), pipe)) {
        std::string line(buf.data());
        if (!line.empty() && line.back() == '\n')
            line.pop_back();
        on_line(line);
    }
    return pclose(pipe);
}

int stream_command_chars(
    const std::string& cmd,
    std::function<void(const std::string&)> on_chunk,
    std::atomic<bool>& cancel)
{
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return -1;

    char buf[64];
    size_t n;
    while (!cancel.load()
           && (n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        on_chunk(std::string(buf, n));
    }
    return pclose(pipe);
}


std::string build_bench_command(const BenchParams& p) {
    std::ostringstream cmd;

    
    if (!p.cpu_mask.empty())
        cmd << "taskset -c " << p.cpu_mask << " ";

    cmd << "\"" << p.llama_bench << "\"";
    cmd << " -m \"" << p.model_path << "\"";
    cmd << " -t " << p.n_threads;
    cmd << " -p " << p.n_prompt;
    cmd << " -n " << p.n_gen;
    cmd << " -r " << p.n_repeat;

    if (p.delay > 0)
        cmd << " --delay " << p.delay;
    if (p.flash_attn)
        cmd << " -fa on";
    if (p.mmap_off)
        cmd << " -mmp 0";

    cmd << " -ctk " << p.kv_type;
    cmd << " -ctv " << p.kv_type;

    return cmd.str();
}

std::string build_infer_command(const InferParams& p) {
    std::ostringstream cmd;

    
    if (!p.cpu_mask.empty())
        cmd << "taskset -c " << p.cpu_mask << " ";

    cmd << "\"" << p.llama_cli << "\"";
    
    cmd << " -m \"" << p.model_path << "\"";
    cmd << " -t " << p.n_threads;
    cmd << " -n " << p.n_predict;

    
    if (p.flash_attn)
        cmd << " -fa on";

    cmd << " -ctk " << p.kv_type;
    cmd << " -ctv " << p.kv_type;
    cmd << " --log-disable";
    cmd << " -p \"" << p.prompt << "\"";
    cmd << " 2>&1";

    return cmd.str();
}