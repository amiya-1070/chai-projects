//storage.h
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "sqlite3.h"

// One complete benchmark run record
struct BenchRecord {
    int64_t     id          = 0;
    std::string timestamp;
    std::string model_name;
    float       model_size_b = 0.0f;   // NEW
    std::string quant        = "";     // NEW
    int         n_threads   = 0;
    std::string cpu_mask;
    bool        flash_attn  = false;
    bool        mmap_off    = false;
    std::string kv_type;
    int         n_prompt    = 0;
    int         n_gen       = 0;
    float       pp_tps      = 0.0f;
    float       pp_std      = 0.0f;
    float       tg_tps      = 0.0f;
    float       tg_std      = 0.0f;
    float       avg_temp_c  = 0.0f;
    float       avg_power_w = 0.0f;
    float       peak_rss_mb = 0.0f;    // NEW
    std::string notes;
};

class Storage {
public:
    Storage();
    ~Storage();

    // Opens or creates the database at the given path.
    // Returns true on success.
    bool open(const std::string& db_path);
    void close();

    // Insert a new benchmark record. Sets record.id on success.
    bool insert_bench(BenchRecord& record);

    // Retrieve all records, newest first.
    std::vector<BenchRecord> get_all_bench() const;

    // Retrieve a single record by id.
    bool get_bench(int64_t id, BenchRecord& out) const;

    // Delete a record by id.
    bool delete_bench(int64_t id);

    // Clear all records.
    bool clear_all();

    bool is_open() const { return m_db != nullptr; }

private:
    bool create_tables();

    bool migrate_schema(); 

    sqlite3* m_db = nullptr;
};