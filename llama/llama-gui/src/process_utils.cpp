#include <sys/types.h>
#include "process_utils.h"
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <array>
#include <fstream>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>

MemSnapshot read_meminfo() {
    MemSnapshot m;
    std::ifstream f("/proc/meminfo");
    std::string key;
    long value_kb;
    std::string unit;
    long swap_total = 0, swap_free = 0;

    while (f >> key >> value_kb >> unit) {
        if (key == "MemTotal:")          m.total_mb     = value_kb / 1024.0f;
        else if (key == "MemAvailable:") m.available_mb = value_kb / 1024.0f;
        else if (key == "SwapTotal:")    swap_total = value_kb;
        else if (key == "SwapFree:")     swap_free  = value_kb;
    }
    m.used_mb      = m.total_mb - m.available_mb;
    m.swap_used_mb = (swap_total - swap_free) / 1024.0f;
    return m;
}

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


extern char **environ;

int stream_command(
    const std::string& cmd,
    std::function<void(const std::string&)> on_line,
    std::atomic<bool>& cancel)
{
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);

    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setpgroup(&attr, 0);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);

    const char* argv[] = { "/bin/sh", "-c", cmd.c_str(), nullptr };

    pid_t pid;
    int rc = posix_spawn(&pid, "/bin/sh", &actions, &attr,
                          const_cast<char* const*>(argv), environ);

    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);

    close(pipefd[1]);

    if (rc != 0) {
        close(pipefd[0]);
        return -1;
    }

    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    std::string leftover;
    char buf[4096];
    bool killed = false;

    while (true) {
        if (cancel.load()) {
            kill(-pid, SIGTERM);
            killed = true;
            for (int i = 0; i < 20; i++) {
                int status;
                pid_t r = waitpid(pid, &status, WNOHANG);
                if (r == pid) break;
                usleep(10000);
            }
            kill(-pid, SIGKILL);
            break;
        }

        struct pollfd pfd { pipefd[0], POLLIN, 0 };
        int pr = poll(&pfd, 1, 100);

        if (pr > 0 && (pfd.revents & POLLIN)) {
            ssize_t n = read(pipefd[0], buf, sizeof(buf));
            if (n > 0) {
                leftover.append(buf, n);
                size_t pos;
                while ((pos = leftover.find('\n')) != std::string::npos) {
                    on_line(leftover.substr(0, pos));
                    leftover.erase(0, pos + 1);
                }
            } else if (n == 0) {
                break;
            }
        } else if (pr == 0) {
            // timeout, loop back
        } else {
            break;
        }

        int status;
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) break;
    }

    close(pipefd[0]);

    if (!killed) {
        int status = 0;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    } else {
        waitpid(pid, nullptr, 0);
        return -1;
    }
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