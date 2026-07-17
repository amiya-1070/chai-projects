#include "gguf_meta.h"
#include <iostream>

int main(int argc, char** argv) {
    std::string path = "/media/amiyaun/New Volume/chai-projects/llama/llm-benchmark/Llama-3.2-1B-Instruct-f16.gguf";  // <-- replace with your actual GGUF path

    if (argc > 1) {
        path = argv[1];  // also allow passing it as a command-line arg
    }

    
    std::cout << "Reading GGUF metadata from: " << path << "\n\n";
    

    GGUFModelMeta meta = read_gguf_meta(path);

    if (!meta.valid) {
        std::cerr << "FAILED: " << meta.error << "\n";
        return 1;
    }
    

    std::cout << "architecture: " << meta.architecture << "\n";
    std::cout << "n_layer:      " << meta.n_layer << "\n";
    std::cout << "n_head:       " << meta.n_head << "\n";
    std::cout << "n_head_kv:    " << meta.n_head_kv << "\n";
    std::cout << "n_embd:       " << meta.n_embd << "\n";
    std::cout << "head_dim:     " << meta.head_dim << " (derived: n_embd / n_head)\n";
    std::cout << "head_dim:     " << meta.head_dim
               << (meta.head_dim_from_derivation ? " (derived: n_embd / n_head)" : " (from attention.key_length)")
               << "\n";

    bool has_gqa = meta.n_head_kv != meta.n_head;
    std::cout << "\nGQA: " << (has_gqa ? "yes" : "no (standard MHA)") << "\n";

    // Sanity-check KV cache formula using these values, at a couple of
    // example context lengths, so we can eyeball whether the numbers
    // look plausible against what you already know about this model.
    auto kv_cache_bytes = [&](uint64_t context_len, int dtype_bytes) -> uint64_t {
        return context_len
             * 2ull  // K and V
             * meta.n_layer
             * meta.n_head_kv
             * meta.head_dim
             * dtype_bytes;
    };

    std::cout << "\nEstimated KV-cache size (fp16, 2 bytes/elem):\n";
    for (uint64_t ctx : {512ull, 2048ull, 4096ull, 8192ull}) {
        uint64_t bytes = kv_cache_bytes(ctx, 2);
        std::cout << "  context=" << ctx << ": "
                  << (bytes / (1024.0 * 1024.0)) << " MB\n";
    }

    return 0;
}