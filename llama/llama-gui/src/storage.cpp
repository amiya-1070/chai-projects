//storage.cpp
#include "storage.h"
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>

static std::string iso_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

Storage::Storage() {}

Storage::~Storage() {
    close();
}


void Storage::close() {
    if (m_db) {
        sqlite3_close(m_db);
        m_db = nullptr;
    }
}

bool Storage::create_tables() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS bench_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            model_name  TEXT    NOT NULL,
            n_threads   INTEGER NOT NULL,
            cpu_mask    TEXT    NOT NULL DEFAULT '',
            flash_attn  INTEGER NOT NULL DEFAULT 0,
            mmap_off    INTEGER NOT NULL DEFAULT 0,
            kv_type     TEXT    NOT NULL DEFAULT 'f16',
            n_prompt    INTEGER NOT NULL DEFAULT 512,
            n_gen       INTEGER NOT NULL DEFAULT 200,
            pp_tps      REAL    NOT NULL DEFAULT 0,
            pp_std      REAL    NOT NULL DEFAULT 0,
            tg_tps      REAL    NOT NULL DEFAULT 0,
            tg_std      REAL    NOT NULL DEFAULT 0,
            avg_temp_c  REAL    NOT NULL DEFAULT 0,
            avg_power_w REAL    NOT NULL DEFAULT 0,
            notes       TEXT    NOT NULL DEFAULT ''
        );
    )";
    char* err = nullptr;
    if (sqlite3_exec(m_db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        sqlite3_free(err);
        return false;
    }
    return true;
}

bool Storage::migrate_schema() {
    // SQLite has no "ALTER TABLE ADD COLUMN IF NOT EXISTS", so just try
    // and swallow the "duplicate column" error if it's already been run.
    const char* alters[] = {
        "ALTER TABLE bench_runs ADD COLUMN model_size_b REAL NOT NULL DEFAULT 0;",
        "ALTER TABLE bench_runs ADD COLUMN quant TEXT NOT NULL DEFAULT '';",
        "ALTER TABLE bench_runs ADD COLUMN peak_rss_mb REAL NOT NULL DEFAULT 0;"
    };
    for (const char* sql : alters) {
        char* err = nullptr;
        sqlite3_exec(m_db, sql, nullptr, nullptr, &err);
        sqlite3_free(err); // ignore errors — column may already exist
    }
    return true;
}

bool Storage::open(const std::string& db_path) {
    if (sqlite3_open(db_path.c_str(), &m_db) != SQLITE_OK) {
        m_db = nullptr;
        return false;
    }
    if (!create_tables()) return false;
    return migrate_schema();
}

bool Storage::insert_bench(BenchRecord& record) {
    if (!m_db) return false;
    record.timestamp = iso_timestamp();

    const char* sql = R"(
        INSERT INTO bench_runs
            (timestamp, model_name, n_threads, cpu_mask, flash_attn,
             mmap_off, kv_type, n_prompt, n_gen,
             pp_tps, pp_std, tg_tps, tg_std,
             avg_temp_c, avg_power_w, notes,
             model_size_b, quant, peak_rss_mb)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
    )";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    sqlite3_bind_text (stmt, 1,  record.timestamp.c_str(),  -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (stmt, 2,  record.model_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int  (stmt, 3,  record.n_threads);
    sqlite3_bind_text (stmt, 4,  record.cpu_mask.c_str(),   -1, SQLITE_TRANSIENT);
    sqlite3_bind_int  (stmt, 5,  record.flash_attn ? 1 : 0);
    sqlite3_bind_int  (stmt, 6,  record.mmap_off   ? 1 : 0);
    sqlite3_bind_text (stmt, 7,  record.kv_type.c_str(),    -1, SQLITE_TRANSIENT);
    sqlite3_bind_int  (stmt, 8,  record.n_prompt);
    sqlite3_bind_int  (stmt, 9,  record.n_gen);
    sqlite3_bind_double(stmt, 10, record.pp_tps);
    sqlite3_bind_double(stmt, 11, record.pp_std);
    sqlite3_bind_double(stmt, 12, record.tg_tps);
    sqlite3_bind_double(stmt, 13, record.tg_std);
    sqlite3_bind_double(stmt, 14, record.avg_temp_c);
    sqlite3_bind_double(stmt, 15, record.avg_power_w);
    sqlite3_bind_text (stmt, 16, record.notes.c_str(),      -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 17, record.model_size_b);
    sqlite3_bind_text (stmt, 18, record.quant.c_str(),      -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 19, record.peak_rss_mb);

    bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
    if (ok) record.id = sqlite3_last_insert_rowid(m_db);
    sqlite3_finalize(stmt);
    return ok;
}

static BenchRecord row_to_record(sqlite3_stmt* stmt) {
    BenchRecord r;
    r.id          = sqlite3_column_int64 (stmt, 0);
    r.timestamp   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    r.model_name  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    r.n_threads   = sqlite3_column_int   (stmt, 3);
    r.cpu_mask    = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
    r.flash_attn  = sqlite3_column_int   (stmt, 5) != 0;
    r.mmap_off    = sqlite3_column_int   (stmt, 6) != 0;
    r.kv_type     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
    r.n_prompt    = sqlite3_column_int   (stmt, 8);
    r.n_gen       = sqlite3_column_int   (stmt, 9);
    r.pp_tps      = (float)sqlite3_column_double(stmt, 10);
    r.pp_std      = (float)sqlite3_column_double(stmt, 11);
    r.tg_tps      = (float)sqlite3_column_double(stmt, 12);
    r.tg_std      = (float)sqlite3_column_double(stmt, 13);
    r.avg_temp_c  = (float)sqlite3_column_double(stmt, 14);
    r.avg_power_w = (float)sqlite3_column_double(stmt, 15);
    r.notes       = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 16));
    r.model_size_b = (float)sqlite3_column_double(stmt, 17);
    r.quant        = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 18));
    r.peak_rss_mb  = (float)sqlite3_column_double(stmt, 19);
    return r;
}

std::vector<BenchRecord> Storage::get_all_bench() const {
    std::vector<BenchRecord> records;
    if (!m_db) return records;

    const char* sql =
        "SELECT id,timestamp,model_name,n_threads,cpu_mask,flash_attn,"
        "mmap_off,kv_type,n_prompt,n_gen,pp_tps,pp_std,tg_tps,tg_std,"
        "avg_temp_c,avg_power_w,notes,model_size_b,quant,peak_rss_mb "
        "FROM bench_runs ORDER BY id DESC;";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return records;

    while (sqlite3_step(stmt) == SQLITE_ROW)
        records.push_back(row_to_record(stmt));

    sqlite3_finalize(stmt);
    return records;
}

bool Storage::get_bench(int64_t id, BenchRecord& out) const {
    if (!m_db) return false;

    const char* sql =
        "SELECT id,timestamp,model_name,n_threads,cpu_mask,flash_attn,"
        "mmap_off,kv_type,n_prompt,n_gen,pp_tps,pp_std,tg_tps,tg_std,"
        "avg_temp_c,avg_power_w,notes,model_size_b,quant,peak_rss_mb "
        "FROM bench_runs ORDER BY id DESC;";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    sqlite3_bind_int64(stmt, 1, id);
    bool found = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        out = row_to_record(stmt);
        found = true;
    }
    sqlite3_finalize(stmt);
    return found;
}

bool Storage::delete_bench(int64_t id) {
    if (!m_db) return false;
    const char* sql = "DELETE FROM bench_runs WHERE id=?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;
    sqlite3_bind_int64(stmt, 1, id);
    bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return ok;
}

bool Storage::clear_all() {
    if (!m_db) return false;
    char* err = nullptr;
    bool ok = (sqlite3_exec(m_db, "DELETE FROM bench_runs;",
                            nullptr, nullptr, &err) == SQLITE_OK);
    sqlite3_free(err);
    return ok;
}