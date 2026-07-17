#include <iostream>
#include "gguf_meta.h"
#include <fstream>
#include <cstring>
#include <vector>

namespace {

// GGUF metadata value types (from the GGUF spec)
enum GGUFType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct Reader {
    std::ifstream& f;
    bool ok = true;

    template <typename T>
    T read_pod() {
        T val{};
        f.read(reinterpret_cast<char*>(&val), sizeof(T));
        if (!f) ok = false;
        return val;
    }

    std::string read_string() {
        uint64_t len = read_pod<uint64_t>();
        if (!ok || len > (1ull << 30)) { ok = false; return {}; } // sanity cap: 1GB string is never legitimate
        std::string s(len, '\0');
        f.read(s.data(), static_cast<std::streamsize>(len));
        if (!f) ok = false;
        return s;
    }

    // Skips a value of the given type without interpreting it — used for
    // metadata keys we don't care about, so we don't need full type
    // handling for every possible value kind.
    void skip_value(uint32_t type) {
        switch (type) {
            case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL:
                read_pod<uint8_t>(); break;
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
                read_pod<uint16_t>(); break;
            case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
                read_pod<uint32_t>(); break;
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
                read_pod<uint64_t>(); break;
            case GGUF_TYPE_STRING:
                read_string(); break;
            case GGUF_TYPE_ARRAY: {
                uint32_t elem_type = read_pod<uint32_t>();
                uint64_t count = read_pod<uint64_t>();
                for (uint64_t i = 0; i < count && ok; i++) {
                    skip_value(elem_type);
                }
                break;
            }
            default:
                ok = false; // unknown type — can't safely skip, abort parse
                break;
        }
    }

    // Reads a value we DO care about as a best-effort uint32, for the
    // numeric integer types we expect config fields to use. Returns
    // nullopt if the type isn't a plain integer we can coerce.
    std::optional<uint32_t> read_uint_value(uint32_t type) {
        switch (type) {
            case GGUF_TYPE_UINT8:  return read_pod<uint8_t>();
            case GGUF_TYPE_INT8:   return static_cast<uint32_t>(read_pod<int8_t>());
            case GGUF_TYPE_UINT16: return read_pod<uint16_t>();
            case GGUF_TYPE_INT16:  return static_cast<uint32_t>(read_pod<int16_t>());
            case GGUF_TYPE_UINT32: return read_pod<uint32_t>();
            case GGUF_TYPE_INT32:  return static_cast<uint32_t>(read_pod<int32_t>());
            case GGUF_TYPE_UINT64: return static_cast<uint32_t>(read_pod<uint64_t>());
            case GGUF_TYPE_INT64:  return static_cast<uint32_t>(read_pod<int64_t>());
            default:
                skip_value(type); // consume it so the stream position stays correct
                return std::nullopt;
        }
    }
};

} // namespace

GGUFModelMeta read_gguf_meta(const std::string& path) {
    GGUFModelMeta meta;

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        meta.error = "Could not open file: " + path;
        return meta;
    }

    Reader r{f};

    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "GGUF", 4) != 0) {
        meta.error = "Not a GGUF file (bad magic)";
        return meta;
    }

    uint32_t version = r.read_pod<uint32_t>();
    (void)version; // not currently validated against a minimum; GGUF has been stable across v2/v3

    uint64_t tensor_count = r.read_pod<uint64_t>();
    uint64_t kv_count = r.read_pod<uint64_t>();
    if (!r.ok) {
        meta.error = "Truncated or malformed GGUF header";
        return meta;
    }
    (void)tensor_count; // we stop before tensor info entirely; not needed

    for (uint64_t i = 0; i < kv_count && r.ok; i++) {
        std::string key = r.read_string();
        if (!r.ok) break;
        uint32_t value_type = r.read_pod<uint32_t>();
        if (!r.ok) break;
        
        std::cerr << "KEY: " << key << "  TYPE: " << value_type << "\n";  // TEMP DEBUG
        // We only care about a handful of numeric keys. Everything else
        // (tokenizer vocab, chat template strings, etc.) gets skipped.
        bool matched = false;

        auto ends_with = [](const std::string& s, const std::string& suffix) {
            return s.size() >= suffix.size() &&
                   s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        if (ends_with(key, ".block_count")) {
            auto v = r.read_uint_value(value_type);
            if (v) { meta.n_layer = *v; matched = true; }
        } else if (ends_with(key, ".attention.head_count_kv")) {
            auto v = r.read_uint_value(value_type);
            if (v) { meta.n_head_kv = *v; matched = true; }
        } else if (ends_with(key, ".attention.head_count")) {
            auto v = r.read_uint_value(value_type);
            if (v) { meta.n_head = *v; matched = true; }
        } else if (ends_with(key, ".embedding_length")) {
            auto v = r.read_uint_value(value_type);
            if (v) { meta.n_embd = *v; matched = true; }
        } else if (ends_with(key, ".attention.key_length")) {
            auto v = r.read_uint_value(value_type);
            if (v) { meta.head_dim = *v; matched = true; }   // authoritative value, takes priority
        } else if (key == "general.architecture" && value_type == GGUF_TYPE_STRING) {
            meta.architecture = r.read_string();
            matched = true;
        }

        if (!matched) {
            r.skip_value(value_type);
        }
    }

    if (!r.ok) {
        meta.error = "Error while parsing metadata key-value section";
        return meta;
    }

    if (meta.n_layer == 0 || meta.n_head == 0 || meta.n_embd == 0) {
        meta.error = "Missing required architecture fields in GGUF metadata "
                      "(block_count / attention.head_count / embedding_length)";
        return meta;
    }

    if (meta.n_head_kv == 0) {
        meta.n_head_kv = meta.n_head;
    }

    if (meta.head_dim == 0) {
        // attention.key_length was absent in this file — fall back to the
        // standard derivation. This assumes n_embd is evenly divisible by
        // n_head and that K/V projections share Q's per-head dimension,
        // which holds for standard architectures but isn't guaranteed for
        // all of them (hence preferring the direct key_length when present).
        if (meta.n_head == 0) {
            meta.error = "Cannot derive head_dim: n_head is zero";
            return meta;
        }
        meta.head_dim = meta.n_embd / meta.n_head;
        meta.head_dim_from_derivation = true;
    }

    meta.valid = true;
    return meta;
    
}