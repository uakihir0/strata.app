// src/kernels/ngram.cpp - P2.S4: the PLE n-gram hash and the IQ4_NL table read.
//
// See include/strata/kernels/ngram.hpp for the semantics, the rival readings, and the note on MADV_RANDOM.
#include "strata/kernels/ngram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <stdexcept>

namespace strata::kernels {

PleConsts ple_artifact_consts() {
    // docs/gguf-dump-shard1.txt, verbatim.  Written once here rather than derived, because `head_offsets` is
    // ALSO in the metadata as its own array - and deriving one from the other would hide a mismatch between
    // them instead of surfacing it.  The parity test checks the two agree.
    PleConsts c{};
    c.mult[0] = 23703573157769ull;
    c.mult[1] = 20109073645365ull;
    c.mult[2] = 8052911324071ull;
    const uint64_t vocab[PLE_N_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    const uint64_t offset[PLE_N_HEADS] = {
        0,        20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    for (int i = 0; i < PLE_N_HEADS; ++i) {
        c.vocab[i] = vocab[i];
        c.offset[i] = offset[i];
    }
    return c;
}

uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n) {
    // The first term is an ASSIGNMENT and the rest are XORed into it, which is how the source writes it
    // (`uint64_t mixed = ctx[0]*m[0]; for j=1.. mixed ^= ctx[j]*m[j];`).  Every product wraps mod 2^64,
    // which is what the `(uint64_t)` casts in the source make explicit.
    uint64_t mixed = (uint64_t) ctx[0] * mult[0];
    for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * mult[j];
    return mixed;
}

void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out) {
    const int n_prev = NGRAM_SIZE - 1;
    for (int i = 0; i < n_tokens; ++i) {
        int64_t ctx[NGRAM_SIZE];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < NGRAM_SIZE; ++s) {
            // `prev` is OLDEST FIRST, so predecessor `s` positions back is entry (n_prev - s): s=1 reads the
            // NEWEST.  Reading index (s-1) instead walks the window backwards, which still produces indices
            // in range and so cannot be caught by a range check - only by an oracle.
            const int32_t t = cut ? TOKEN_NULL : prev[i * n_prev + (n_prev - s)];
            // The cut is evaluated BEFORE the value is stored, so the position whose predecessor was EOS is
            // itself EOS.  Storing first and then cutting would leave position s holding the real token while
            // position s+1 became EOS - one token of history too much.
            cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;
            ctx[s] = cut ? PLE_EOS_TOKEN_ID : t;
        }
        for (int n = 2; n <= NGRAM_SIZE; ++n) {
            const uint64_t mixed = ngram_mixed(ctx, c.mult, n);
            const int base = (n - 2) * HEADS_PER_NGRAM;
            for (int g = 0; g < HEADS_PER_NGRAM; ++g) {
                const int h = base + g;
                out[i * PLE_N_HEADS + h] = (uint32_t) (mixed % c.vocab[h] + c.offset[h]);
            }
        }
    }
}

namespace {
const int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
}

int iq4nl_code(int code) { return kIq4Nl[code & 15]; }

void iq4nl_dequant_row(const uint8_t* row, float* out160) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const float d = f32_from_f16(dbits);
        const uint8_t* qs = blk + 2;
        // SPLIT HALVES: qs[j] holds elements j and j+16, not 2j and 2j+1.
        for (int j = 0; j < 16; ++j) {
            out160[b * 32 + j] = d * (float) kIq4Nl[qs[j] & 0x0F];
            out160[b * 32 + j + 16] = d * (float) kIq4Nl[qs[j] >> 4];
        }
    }
}

namespace {
struct Fp8Table {
    float v[256];
    Fp8Table() {
        for (int x = 0; x < 256; ++x) {
            const int s = x >> 7, e = (x >> 3) & 15, m = x & 7;
            float f;
            if (e == 15 && m == 7) f = 0.0f;                     // NaN in e4m3fn; never in a real table, read as 0
            else if (e == 0) f = std::ldexp((float) m / 8.0f, -6);
            else f = std::ldexp(1.0f + (float) m / 8.0f, e - 7);
            v[x] = s ? -f : f;
        }
    }
};
const Fp8Table kFp8;
}  // namespace

void fp8_e4m3_dequant_row(const uint8_t* row, float scale, float* out160) {
    for (int j = 0; j < PLE_HEAD_DIM; ++j) out160[j] = kFp8.v[row[j]] * scale;
}

// ---------------------------------------------------------------------------------------------------
namespace {

struct PleTableInfo {
    const uint8_t* data = nullptr;
    uint64_t rows = 0;
    uint64_t table_offset = 0;
    uint64_t table_bytes = 0;
    bool q5_0 = false;
    bool fp8 = false;
    float scale = 1.0f;
    uint32_t row_bytes = PLE_ROW_BYTES;
};

bool inspect_ple_table(const GgufFile& file, const GgufFile& metadata,
                       PleIo mode, PleTableInfo& info, std::string& err) {
    const TensorInfo* tensor = file.find("per_layer_token_embd.weight");
    if (tensor == nullptr) {
        err = "per_layer_token_embd.weight is not in " + file.path();
        return false;
    }
    if (tensor->shape.size() != 2 ||
        tensor->shape[0] != static_cast<uint64_t>(PLE_HEAD_DIM)) {
        err = "per_layer_token_embd.weight has an unexpected shape";
        return false;
    }

    PleTableInfo candidate;
    candidate.rows = tensor->shape[1];
    candidate.q5_0 = std::strcmp(tensor->type_name(), "Q5_0") == 0;
    candidate.row_bytes = candidate.q5_0
        ? static_cast<uint32_t>((PLE_HEAD_DIM / 32) * 22)
        : static_cast<uint32_t>(PLE_ROW_BYTES);
    if (std::strcmp(tensor->type_name(), "I8") == 0) {
        const MetaValue* format = metadata.get("strata.ple.format");
        const MetaValue* scale = metadata.get("strata.ple.scale");
        double scale_value = 0.0;
        bool valid_scale_type = false;
        bool signed_integer_scale = false;
        if (scale != nullptr) {
            switch (scale->type) {
            case MetaType::F32:
            case MetaType::F64:
                scale_value = scale->f;
                valid_scale_type = true;
                break;
            case MetaType::U8:
            case MetaType::U16:
            case MetaType::U32:
            case MetaType::U64:
                scale_value = static_cast<double>(scale->u);
                valid_scale_type = true;
                break;
            case MetaType::I8:
                signed_integer_scale = true;
                valid_scale_type =
                    scale->u <=
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int8_t>::max());
                break;
            case MetaType::I16:
                signed_integer_scale = true;
                valid_scale_type =
                    scale->u <=
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int16_t>::max());
                break;
            case MetaType::I32:
                signed_integer_scale = true;
                valid_scale_type =
                    scale->u <=
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int32_t>::max());
                break;
            case MetaType::I64:
                signed_integer_scale = true;
                valid_scale_type =
                    scale->u <=
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max());
                break;
            default:
                break;
            }
            if (valid_scale_type && signed_integer_scale) {
                scale_value = static_cast<double>(scale->u);
            }
        }
        const float min_positive_float =
            std::numeric_limits<float>::denorm_min() > 0.0f
                ? std::numeric_limits<float>::denorm_min()
                : std::numeric_limits<float>::min();
        const double min_positive_scale =
            static_cast<double>(min_positive_float);
        const double max_scale =
            static_cast<double>(std::numeric_limits<float>::max());
        if (format == nullptr || format->s != "f8_e4m3" ||
            !valid_scale_type || !std::isfinite(scale_value) ||
            scale_value < min_positive_scale || scale_value > max_scale) {
            err = "per_layer_token_embd.weight is I8 without "
                  "strata.ple.format = f8_e4m3 and a finite, positive "
                  "strata.ple.scale representable as float";
            return false;
        }
        candidate.fp8 = true;
        candidate.scale = static_cast<float>(scale_value);
        candidate.row_bytes = PLE_ROW_BYTES_FP8;
    } else if (!candidate.q5_0 &&
               std::strcmp(tensor->type_name(), "IQ4_NL") != 0) {
        err = std::string("per_layer_token_embd.weight is ") +
              tensor->type_name() +
              ", not IQ4_NL, Q5_0 or FP8 (I8)";
        return false;
    }
    if (candidate.q5_0 && mode == PleIo::Direct) {
        err = "Q5_0 PLE requires --ple-io mmap";
        return false;
    }
    if (file.data_start() > file.file_size() ||
        candidate.rows >
            std::numeric_limits<uint64_t>::max() / candidate.row_bytes) {
        err = "PLE table extent overflows the GGUF file size";
        return false;
    }
    candidate.table_bytes =
        candidate.rows * static_cast<uint64_t>(candidate.row_bytes);
    const uint64_t payload_bytes = file.file_size() - file.data_start();
    const uint64_t tensor_offset = tensor->offset;
    const bool alone = file.tensors().size() == 1;
    const bool outside_payload =
        tensor_offset > payload_bytes ||
        candidate.table_bytes > payload_bytes - tensor_offset;
    if (outside_payload ||
        (alone &&
         (tensor_offset != 0 || candidate.table_bytes != payload_bytes))) {
        char buffer[256];
        std::snprintf(
            buffer, sizeof buffer,
            "PLE table size mismatch: %llu rows x %u B = %llu at offset %llu, "
            "but the file holds %llu from data_start %llu",
            static_cast<unsigned long long>(candidate.rows),
            candidate.row_bytes,
            static_cast<unsigned long long>(candidate.table_bytes),
            static_cast<unsigned long long>(tensor_offset),
            static_cast<unsigned long long>(payload_bytes),
            static_cast<unsigned long long>(file.data_start()));
        err = buffer;
        return false;
    }
    candidate.table_offset = file.data_start() + tensor_offset;
    candidate.data = file.tensor_data(*tensor);
    info = candidate;
    err.clear();
    return true;
}

}  // namespace

struct PleTable::Impl {
    GgufFile* file = nullptr;
    std::shared_ptr<const GgufModel> mapped_model;
    const uint8_t* data = nullptr;
    uint64_t n_rows = 0;
    bool q5_0 = false;                // #296: Q5_0 rows (110 B), the mapped reader only
    mutable uint64_t bytes_read = 0;
    // Direct mode (plan v0.3 P2): the mapping above is released after the header parse and every row comes
    // from an unbuffered SSD read into `raw`.
    PleIo mode = PleIo::Mmap;
    strata::ngram::PleReader reader;
    strata::ngram::PleReader::Ticket ticket;
    bool pending = false;
    bool locked = false;
    uint32_t rows[PLE_N_HEADS] = {};
    uint8_t raw[PLE_N_HEADS * PLE_ROW_BYTES_MAX] = {};
    bool fp8 = false;                 // F8_E4M3 rows (tools/ple_fp8_pack.py); else IQ4_NL
    float scale = 1.0f;               // the FP8 table's one scale
    uint32_t rb = PLE_ROW_BYTES;      // bytes per row
    void decode(const uint8_t* row, float* out160) const {
        if (fp8) fp8_e4m3_dequant_row(row, scale, out160);
        else if (q5_0)
            for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) strata::dequantize_q5_0(row + (size_t) b * 22, out160 + b * 32);
        else iq4nl_dequant_row(row, out160);
    }
};

PleTable::PleTable() : impl_(new Impl) {}
PleTable::~PleTable() { close(); delete impl_; }

bool PleTable::open(const std::string& gguf_path, std::string& err) {
    return open(gguf_path, err, PleIoOptions{});
}

bool PleTable::open(const std::string& gguf_path, std::string& err, const PleIoOptions& io) {
    close();
    try {
        impl_->file = new GgufFile(gguf_path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    PleTableInfo table;
    if (!inspect_ple_table(*impl_->file, *impl_->file, io.mode, table, err)) {
        close();
        return false;
    }
    impl_->fp8 = table.fp8;
    impl_->q5_0 = table.q5_0;
    impl_->rb = table.row_bytes;
    impl_->scale = table.scale;
    impl_->n_rows = table.rows;
    impl_->data = table.data;
    if (io.mode == PleIo::Direct) {
        // The parse above is the validated source of the offset; the mapping itself is not kept, so no page of
        // the table can enter this process's working set or the file cache through it.
        const uint64_t n_rows = table.rows;
        delete impl_->file;
        impl_->file = nullptr;
        impl_->data = nullptr;
        if (!impl_->reader.open(gguf_path, table.table_offset, n_rows,
                                io.max_inflight, io.cache_rows, err, io.io_thread,
                                table.row_bytes)) {
            close();
            return false;
        }
        impl_->reader.set_keepalive(io.keepalive_ms, io.keepalive_window_s);
        impl_->n_rows = n_rows;
    }
    if (io.mode == PleIo::Mmap && table.data != nullptr) {
        const long page_bytes = ::sysconf(_SC_PAGESIZE);
        if (page_bytes > 0) {
            const uintptr_t page = static_cast<uintptr_t>(page_bytes);
            const uintptr_t table_begin = reinterpret_cast<uintptr_t>(table.data);
            const uintptr_t a0 = table_begin - table_begin % page;
            const uintptr_t a1 = table_begin + static_cast<uintptr_t>(table.table_bytes);
            const size_t bytes = static_cast<size_t>(a1 - a0);
            // Tensor rows are random accesses; advise macOS on the page-aligned tensor range only.
            (void)madvise(reinterpret_cast<void*>(a0), bytes, MADV_RANDOM);
            if (io.lock) {
                if (mlock(reinterpret_cast<const void*>(a0), bytes) == 0) {
                    impl_->locked = true;
                } else {
                    std::fprintf(stderr, "strata: PLE table mlock failed (%s; raise `ulimit -l`): touching its pages instead\n",
                                 std::strerror(errno));
                    volatile uint8_t sink = 0;
                    for (uintptr_t address = a0; address < a1; address += page)
                        sink = static_cast<uint8_t>(sink + *(const volatile uint8_t*)address);
                    (void)sink;
                }
            }
        }
    }
    impl_->mode = io.mode;
    return true;
}

bool PleTable::open_mapped(std::shared_ptr<const GgufModel> model,
                           std::size_t shard_index, std::string& err) {
    close();
    err.clear();
    if (!model) {
        err = "PleTable::open_mapped needs an owned GgufModel";
        return false;
    }
    if (shard_index >= model->size()) {
        err = "PleTable::open_mapped shard index is out of range";
        return false;
    }
    std::size_t actual_shard = 0;
    if (model->find("per_layer_token_embd.weight", &actual_shard) == nullptr ||
        actual_shard != shard_index) {
        err = "PleTable::open_mapped shard does not contain the PLE tensor";
        return false;
    }

    PleTableInfo table;
    if (!inspect_ple_table(model->shard(shard_index), model->meta(),
                           PleIo::Mmap, table, err)) {
        return false;
    }
    impl_->mapped_model = model;
    impl_->data = table.data;
    impl_->n_rows = table.rows;
    impl_->q5_0 = table.q5_0;
    impl_->fp8 = table.fp8;
    impl_->scale = table.scale;
    impl_->rb = table.row_bytes;
    impl_->mode = PleIo::Mmap;
    err.clear();
    return true;
}

void PleTable::reader_view(PleTableReader& view) noexcept {
    view.table_ = this;
}

void PleTable::close() {
    impl_->reader.close();
    impl_->pending = false;
    impl_->locked = false;   // the unmap below releases the lock
    impl_->mode = PleIo::Mmap;
    delete impl_->file;
    impl_->file = nullptr;
    impl_->mapped_model.reset();
    impl_->data = nullptr;
    impl_->n_rows = 0;
    impl_->rb = PLE_ROW_BYTES;
    impl_->q5_0 = false;
    impl_->fp8 = false;
}

bool PleTable::is_open() const { return impl_->data != nullptr || impl_->reader.is_open(); }
bool PleTable::locked() const { return impl_->locked; }
const char* PleTable::format() const { return impl_->fp8 ? "F8_E4M3" : impl_->q5_0 ? "Q5_0" : "IQ4_NL"; }
PleIo PleTable::mode() const { return impl_->mode; }
uint64_t PleTable::rows() const { return impl_->n_rows; }
uint64_t PleTable::bytes_read() const { return impl_->bytes_read; }

bool PleTableReader::is_open() const {
    return table_ != nullptr && table_->is_open();
}

uint64_t PleTableReader::rows() const {
    return is_open() ? table_->rows() : 0u;
}

const char* PleTableReader::format() const {
    return is_open() ? table_->format() : "";
}

PleIo PleTableReader::mode() const {
    return is_open() ? table_->mode() : PleIo::Mmap;
}

bool PleTableReader::issue(const uint32_t* rows16) {
    return is_open() && table_->issue(rows16);
}

bool PleTableReader::collect(float* out2560, std::string& err) {
    if (!is_open()) {
        err = "PLE table reader view is unavailable";
        return false;
    }
    return table_->collect(out2560, err);
}

bool PleTableReader::gather_batch(const uint32_t* rows, size_t n_tokens,
                                  float* out, std::string& err) {
    if (!is_open()) {
        err = "PLE table reader view is unavailable";
        return false;
    }
    return table_->gather_batch(rows, n_tokens, out, err);
}

void PleTable::read_row(uint32_t row, float* out160) const {
    if (impl_->mode == PleIo::Direct && impl_->reader.is_open()) {
        uint8_t raw[PLE_ROW_BYTES_MAX];
        std::string err;
        const auto t = impl_->reader.issue(&row, 1, raw);
        if (!impl_->reader.collect(t, err)) {
            std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
            return;
        }
        impl_->decode(raw, out160);
        impl_->bytes_read += impl_->rb;
        return;
    }
    if (impl_->data == nullptr || row >= impl_->n_rows) {
        std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
        return;
    }
    impl_->decode(impl_->data + (size_t) row * impl_->rb, out160);
    impl_->bytes_read += impl_->rb;
}

bool PleTable::issue(const uint32_t* rows16) {
    std::memcpy(impl_->rows, rows16, sizeof impl_->rows);
    if (impl_->mode == PleIo::Direct) {
        if (impl_->pending) return false;              // one token in flight per table
        impl_->ticket = impl_->reader.issue(impl_->rows, PLE_N_HEADS, impl_->raw);
        impl_->pending = true;
        return true;
    }
    impl_->pending = true;
    return true;
}

bool PleTable::collect(float* out2560, std::string& err) {
    if (!impl_->pending) { err = "PleTable::collect without issue"; return false; }
    impl_->pending = false;
    if (impl_->mode == PleIo::Direct) {
        if (!impl_->reader.collect(impl_->ticket, err)) return false;
        for (int h = 0; h < PLE_N_HEADS; ++h)
            impl_->decode(impl_->raw + (size_t) h * impl_->rb, out2560 + (size_t) h * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) PLE_N_HEADS * impl_->rb;
        return true;
    }
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(impl_->rows[h], out2560 + (size_t) h * PLE_HEAD_DIM);
    return true;
}

bool PleTable::gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err) {
    if (impl_->pending) { err = "PleTable::gather_batch while a token is in flight"; return false; }
    const size_t n = n_tokens * (size_t) PLE_N_HEADS;
    if (impl_->mode == PleIo::Direct) {
        std::vector<uint8_t> raw(n * impl_->rb);
        const auto ticket = impl_->reader.issue(rows, n, raw.data());
        if (!impl_->reader.collect(ticket, err)) return false;
        for (size_t i = 0; i < n; ++i) impl_->decode(raw.data() + i * impl_->rb, out + i * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) n * impl_->rb;
        return true;
    }
    for (size_t i = 0; i < n; ++i) read_row(rows[i], out + i * PLE_HEAD_DIM);
    return true;
}

void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_delay_us(us); }

std::string PleTable::io_report() const {
    if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
    const strata::ngram::ReaderStats s = impl_->reader.snapshot();
    char buf[400];
    int n = std::snprintf(buf, sizeof buf,
                  "ple io: %llu rows, %.1f%% row-cache hits, %llu SSD reads (%.1f MB), read p50 %.0f us p99 %.0f us, "
                  "blocked %.3f ms total (submit %.3f ms), cache %llu/%llu rows",
                  (unsigned long long) s.requests, s.requests ? 100.0 * (double) s.cache_hits / (double) s.requests : 0.0,
                  (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                  s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                  (unsigned long long) impl_->reader.cache_capacity());
    if (s.keepalive_reads > 0 && n > 0 && n < (int) sizeof buf)
        std::snprintf(buf + n, sizeof buf - (size_t) n, ", SSD kept awake by %llu reads (slowest %.1f ms)",
                      (unsigned long long) s.keepalive_reads, s.keepalive_us_max / 1000.0);
    return buf;
}

void PleTable::gather(const uint32_t* rows16, float* out2560) const {
    if (impl_->mode == PleIo::Direct) {
        // `gather` stays const for its existing callers; the reader's state is the table's I/O state.
        PleTable* self = const_cast<PleTable*>(this);
        std::string err;
        if (!self->issue(rows16) || !self->collect(out2560, err)) {
            std::fprintf(stderr, "PleTable::gather: %s\n", err.empty() ? "a token is already in flight" : err.c_str());
            std::memset(out2560, 0, (size_t) NG_N_EMBD * sizeof(float));
        }
        return;
    }
    // HEAD-SLOWEST, which is what `ggml_get_rows` does and what the source's own comment says: head h's 160
    // values occupy [h*160, (h+1)*160).  A head-fastest layout would put element (d, h) at d*16 + h and needs
    // a real transpose - a reshape of the same flat buffer compares equal and would make the check vacuous.
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(rows16[h], out2560 + (size_t) h * PLE_HEAD_DIM);
}

}  // namespace strata::kernels
