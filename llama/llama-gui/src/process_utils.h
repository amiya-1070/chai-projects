//process.utils
#pragma once
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>

// Runs a command and returns all stdout as a string.
// Blocks until the process exits.
std::string run_command(const std::string& cmd);

// Spawns a process and streams stdout line by line to a callback.
// Runs in the calling thread — use from a dedicated thread.
// Returns the process exit code.
int stream_command(
    const std::string& cmd,
    std::function<void(const std::string& line)> on_line,
    std::atomic<bool>& cancel
);

int stream_command_chars(
    const std::string& cmd,
    std::function<void(const std::string& chunk)> on_chunk,
    std::atomic<bool>& cancel
);

// Builds a taskset + llama-bench command string from parameters.
struct BenchParams {
    std::string model_path;
    int         n_threads    = 8;
    int         n_prompt     = 512;
    int         n_gen        = 200;
    int         n_repeat     = 3;
    int         delay        = 0;
    bool        flash_attn   = true;
    bool        mmap_off     = true;
    std::string kv_type      = "q8_0"; // f16, q8_0, q4_0
    std::string cpu_mask     = "";     // empty = no taskset
    std::string llama_bench  = "";     // path to llama-bench binary
};

std::string build_bench_command(const BenchParams& p);

// Builds a llama-cli command string for inference.
struct InferParams {
    std::string model_path;
    int         n_threads   = 8;
    int         n_predict   = 512;
    bool        flash_attn  = true;
    bool        mmap_off    = true;
    std::string kv_type     = "q8_0";
    std::string cpu_mask    = "";
    std::string llama_cli   = "";
    std::string prompt      = "";
};

// Parsed result from one llama-bench output line
struct BenchResult {
    std::string test;        // "pp512" or "tg200" etc
    float       tps   = 0.0f;
    float       std   = 0.0f;
    bool        valid = false;
};

struct MemSnapshot {
    float total_mb      = 0.0f;
    float available_mb  = 0.0f;
    float used_mb        = 0.0f;
    float swap_used_mb   = 0.0f;
};

MemSnapshot read_meminfo();

// Returns true if `line` looks like a llama-bench markdown table row
// (as opposed to a header, separator, or log line).
bool is_bench_line(const std::string& line);

// Parses a llama-bench markdown table row into a BenchResult.
// Returns an invalid (valid=false) result if the line doesn't parse.
BenchResult parse_bench_line(const std::string& line);

std::string build_infer_command(const InferParams& p);