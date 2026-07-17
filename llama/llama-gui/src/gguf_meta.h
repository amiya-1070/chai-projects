#pragma once
#include <string>
#include <cstdint>
#include <optional>

// Model architecture metadata relevant to KV-cache size computation,
// extracted directly from a GGUF file's header (no need to load weights).

struct GGUFModelMeta {
    bool        valid = false;
    std::string error;

    std::string architecture;
    uint32_t    n_layer      = 0;
    uint32_t    n_head       = 0;
    uint32_t    n_head_kv    = 0;
    uint32_t    n_embd       = 0;
    uint32_t    head_dim     = 0;   // from attention.key_length if present, else derived as n_embd/n_head
    bool        head_dim_from_derivation = false;  // true if key_length was absent and we had to fall back
};

// Reads only the GGUF header + metadata KV section from the file at
// `path` — stops before tensor data, so this is fast even on multi-GB
// files. Returns a GGUFModelMeta with valid=false and `error` set on
// any parse failure (bad magic, truncated file, missing required keys).
GGUFModelMeta read_gguf_meta(const std::string& path);

inline double kv_type_bytes_per_element(const std::string& kv_type) {
    if (kv_type == "f16")  return 2.0;
    if (kv_type == "q8_0") return 34.0 / 32.0;
    if (kv_type == "q4_0") return 18.0 / 32.0;
    return 2.0;
}