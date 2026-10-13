// Utility kernels of docs/mac/04-metal-kernels.md, section 5.13, against independent host references (class B).
// strata_util_copy_strided is a bit-exact strided copy. strata_util_to_f16 and strata_util_to_bf16 are checked
// against the float32 -> float16 and float32 -> BF16 rules that appendix A.10 measured on Metal; the references
// are anchored to A.10's measured values first. Every T from 1 to 8 runs, and row t of each T-row dispatch must
// equal the T = 1 dispatch of row t (section 10.1, step 5). The inputs are synthetic: these utilities read no model
// tensors, so there is no real input set and `--input-set real` exits with the skip code.
#include "synth.hpp"
#include "twin.hpp"

#include "metallib_sha256.inc"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

// The MTP row width of section 5.13 (10,240 floats) and the row counts of section 10.1, step 5.
constexpr std::size_t kCols = 10240;
constexpr std::uint32_t kMaxRows = 8;
constexpr std::uint64_t kThreadsPerGroup = 256;

// Parameter blocks of the kernels in src/metal/kernels/safe/util.metal.
struct CopyArgs {
    std::uint32_t cols;
    std::uint32_t src_stride;
    std::uint32_t dst_stride;
};
static_assert(sizeof(CopyArgs) == 12);

struct BF16Args {
    std::uint32_t cols;
    std::uint32_t has_lo;
};
static_assert(sizeof(BF16Args) == 8);

int g_failures = 0;

void check(bool ok, const std::string& label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", label.c_str(), detail.empty() ? "" : ": ", detail.c_str());
    std::fflush(stdout);
    g_failures += !ok;
}

std::uint32_t bits_of(float value) { return std::bit_cast<std::uint32_t>(value); }

float float_from_bits(std::uint32_t bits) { return std::bit_cast<float>(bits); }

float bf16_to_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

bool is_denormal(float value) {
    const std::uint32_t bits = bits_of(value);
    return (bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u;
}

// The GPU treats a float32 denormal as a signed zero: the sign survives, which the edge set shows on this Mac for
// negative inputs. Appendix A.10 records +0; its measured inputs were positive, so the two agree there.
float flush_denormal(float value) {
    return is_denormal(value) ? std::bit_cast<float>(bits_of(value) & 0x80000000u) : value;
}

// Oracle for float32 -> float16 (section 5.13, appendix A.10): a float32 denormal becomes a signed zero, every NaN
// becomes 0x7e00, and every other value is rounded to nearest even by the host's `_Float16` conversion, which keeps
// subnormal halves and overflows to infinity from 65520.
std::uint16_t half_reference(float x) {
    if (is_denormal(x)) return static_cast<std::uint16_t>((bits_of(x) >> 16) & 0x8000u);
    if (std::isnan(x)) return 0x7e00u;
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(x));
}

// Oracle for float32 -> BF16 round-to-nearest-even. It decides on the discarded low half directly, not with the bias
// form of the kernel. A NaN keeps the top half with the quiet bit set (section 3.6).
std::uint16_t bf16_reference(float x) {
    const std::uint32_t bits = bits_of(x);
    const std::uint32_t high = bits >> 16;
    if (std::isnan(x)) return static_cast<std::uint16_t>(high | 0x40u);
    const std::uint32_t low = bits & 0xffffu;
    const bool round_up = low > 0x8000u || (low == 0x8000u && (high & 1u) != 0u);
    return static_cast<std::uint16_t>(high + (round_up ? 1u : 0u));
}

bool is_bf16_nan(std::uint16_t bits) {
    return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0u;
}

struct LowExpectation {
    std::uint16_t bits = 0;
    // The residual is NaN (a NaN input, or infinity minus infinity). The specification fixes only the class.
    bool nan = false;
    // The residual is a nonzero float32 denormal, whose result flush is not established. No test input reaches this.
    bool unmodelled = false;
};

// The low part is the BF16 of the residual x - hi, with both float operands flushed as the GPU flushes them (see
// flush_denormal). The residual is formed in double. For finite normal operands it has at most 16 significant bits
// inside the float range, so converting it to float is exact.
LowExpectation low_reference(float x, std::uint16_t hi) {
    const double residual = static_cast<double>(flush_denormal(x)) -
                            static_cast<double>(flush_denormal(bf16_to_float(hi)));
    if (std::isnan(residual)) return {0, true, false};
    if (residual != 0.0 && std::fabs(residual) < 0x1.0p-126) return {0, false, true};
    return {bf16_reference(static_cast<float>(residual)), false, false};
}

// Edge bit patterns: signed zeros, both ends of the float32 denormal range, the half subnormal and overflow
// boundaries, infinities, NaN payloads of both signs, and the round-to-even ties of float16 and BF16.
constexpr std::array kEdgeBits = {
    0x00000000u, 0x80000000u,  // +0, -0
    0x00000001u, 0x80000001u,  // smallest float32 denormal, both signs
    0x007fffffu, 0x807fffffu,  // largest float32 denormal, both signs
    0x00800000u, 0x80800000u,  // smallest normal, both signs
    0x33000000u, 0xb3000000u,  // +-2^-25: a float16 tie that rounds to even (zero)
    0x33000001u, 0xb3000001u,  // just above the tie: 0x0001 and 0x8001
    0x33800000u, 0x38800000u,  // 2^-24 (smallest half subnormal) and 2^-14 (smallest half normal)
    0x387fc000u, 0xb87fc000u,  // 0x03ff and 0x83ff (A.10)
    0x477fefffu, 0xc77fefffu,  // +-65519.996: the largest finite half, 0x7bff and 0xfbff
    0x477ff000u, 0xc77ff000u,  // +-65520: overflows to infinity
    0x477fe000u, 0x47800000u,  // 65504 (the largest half below the overflow point) and 65536
    0x7f7fffffu, 0xff7fffffu,  // +-FLT_MAX: infinity in half and, by rounding up, in BF16
    0x7f800000u, 0xff800000u,  // +-infinity
    0x7fc00000u, 0xffc00001u,  // quiet NaNs with payloads of both signs: 0x7e00 in half
    0x7f800001u, 0x7fffffffu,  // signalling and all-ones NaNs
    0x3f808000u, 0x3f818000u,  // BF16 ties: 1 + 2^-8 to 1.0, and 1 + 3*2^-8 to 1 + 2^-6
    0x3f800000u, 0x3f807fffu,  // 1.0 and the value just below a BF16 tie
    0x3f801000u, 0x3f803000u,  // float16 ties: 1 + 2^-11 to 1.0, and 1 + 3*2^-11 to 1 + 2^-9
    0x3a000000u, 0xbf800001u,  // 2^-11, and -1 with its lowest mantissa bit set
    0x40490fdbu, 0xc0490fdbu,  // +-pi
    0x7f7f0000u, 0x00000000u,  // a BF16-exact large value, and +0
};

// The A.10 anchors that the fixture set repeats: the measured tie, the measured 0x03ff case and the overflow pair.
constexpr std::array kFixtureBits = {
    0x33000000u, 0x387fc000u, 0x477fefffu, 0x477ff000u,
    0xc77ff000u, 0x7f800000u, 0x00000000u, 0x3f808000u,
};

// Random inputs are normal values at log-uniform magnitudes from 2^-30 to 2^20, so the half subnormal and overflow
// ranges are reached as well as the common ones.
std::vector<float> make_inputs(twin::InputSet set, std::size_t count, std::mt19937_64& rng) {
    std::vector<float> values(count);
    if (set == twin::InputSet::random) {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_int_distribution<int> exponent(-30, 20);
        for (float& value : values) value = std::ldexp(normal(rng), exponent(rng));
        return values;
    }
    const std::span<const std::uint32_t> table = set == twin::InputSet::edge
                                                     ? std::span<const std::uint32_t>(kEdgeBits)
                                                     : std::span<const std::uint32_t>(kFixtureBits);
    for (std::size_t i = 0; i < count; ++i) values[i] = float_from_bits(table[i % table.size()]);
    return values;
}

bool allocate(Context& context, twin::GuardedBuffer& buffer, const std::string& label, std::size_t bytes,
              std::string& err) {
    if (!buffer.allocate(context, label, bytes, err)) return false;
    if (!buffer.view()) {
        err = label + " is outside its Metal allocation";
        return false;
    }
    return true;
}

void store_u32(std::span<std::uint8_t> bytes, std::uint32_t value) {
    std::memcpy(bytes.data(), &value, sizeof(value));
}

std::vector<std::uint8_t> bytes_of_u16(std::span<const std::uint16_t> values) {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(std::uint16_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

// Each element of `expected` carries the bit-flipped value (section 10.1, step 2) as its initial contents, so a kernel
// that leaves an element unwritten fails the comparison in its own class.
std::vector<std::uint8_t> flipped_bits(std::vector<std::uint8_t> expected, std::size_t element_bytes) {
    for (std::size_t offset = 0; offset < expected.size(); offset += element_bytes) expected[offset] ^= 0x01u;
    return expected;
}

bool dispatch_util(Context& context, const char* kernel, std::vector<twin::Binding> buffers,
                   const void* parameters, std::size_t parameter_bytes, unsigned parameter_index,
                   std::uint64_t threads, std::string& err) {
    twin::Dispatch work;
    work.buffers = std::move(buffers);
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {static_cast<std::uint32_t>((threads + kThreadsPerGroup - 1) / kThreadsPerGroup), 1, 1};
    work.threads_per_group = {0, 0, 0};  // the manifest's threadgroup shape, checked by dispatch_manifest
    work.parameters = parameters;
    work.parameter_bytes = parameter_bytes;
    work.parameter_index = parameter_index;
    return twin::dispatch_manifest(context, kernel, std::move(work), err);
}

// Self-check of the references. The anchors are A.10's measured Metal results and the special-value rules of section
// 5.13, so a wrong oracle fails here before any kernel is blamed.
void test_reference_anchors() {
    check(half_reference(float_from_bits(0x477fefffu)) == 0x7bffu, "half reference: 65519.996 is the largest half");
    check(half_reference(float_from_bits(0x477ff000u)) == 0x7c00u, "half reference: 65520 overflows to infinity");
    check(half_reference(float_from_bits(0xc77ff000u)) == 0xfc00u, "half reference: -65520 overflows to -infinity");
    check(half_reference(float_from_bits(0x33000000u)) == 0x0000u, "half reference: 2^-25 ties to even zero");
    check(half_reference(float_from_bits(0x33000001u)) == 0x0001u, "half reference: above 2^-25 keeps 0x0001");
    check(half_reference(float_from_bits(0x387fc000u)) == 0x03ffu, "half reference: A.10 value gives 0x03ff");
    check(half_reference(float_from_bits(0x00000001u)) == 0x0000u &&
              half_reference(float_from_bits(0x80000001u)) == 0x8000u,
          "half reference: float32 denormals become signed zeros");
    check(half_reference(float_from_bits(0x80000000u)) == 0x8000u, "half reference: -0 keeps its sign");
    check(half_reference(float_from_bits(0x7fc00000u)) == 0x7e00u &&
              half_reference(float_from_bits(0xffc00001u)) == 0x7e00u,
          "half reference: every NaN becomes 0x7e00");
    check(bf16_reference(float_from_bits(0x3f808000u)) == 0x3f80u, "BF16 reference: tie to even 1.0");
    check(bf16_reference(float_from_bits(0x3f818000u)) == 0x3f82u, "BF16 reference: tie to even 1 + 2^-6");
    check(bf16_reference(float_from_bits(0x3f807fffu)) == 0x3f80u &&
              bf16_reference(float_from_bits(0x3f808001u)) == 0x3f81u,
          "BF16 reference: values off the tie round to nearest");
    check(bf16_reference(float_from_bits(0x7f7fffffu)) == 0x7f80u, "BF16 reference: FLT_MAX rounds to infinity");
    check(bf16_reference(float_from_bits(0x7fc00001u)) == 0x7fc0u &&
              bf16_reference(float_from_bits(0xffc00001u)) == 0xffc0u &&
              bf16_reference(float_from_bits(0x7f800001u)) == 0x7fc0u,
          "BF16 reference: NaNs keep the top half with the quiet bit set");
    // 1 + 3*2^-8 rounds to 1 + 2^-6, so the residual is -2^-8, whose BF16 is 0xbb80.
    const float tie = float_from_bits(0x3f818000u);
    const LowExpectation low = low_reference(tie, bf16_reference(tie));
    check(!low.nan && !low.unmodelled && low.bits == 0xbb80u, "BF16 low reference: the tie residual is -2^-8");
    // The GPU flushes a denormal operand: x = -denormal with hi = -0 gives +0 (edge set, element 3).
    const float denormal = float_from_bits(0x80000001u);
    check(low_reference(denormal, bf16_reference(denormal)).bits == 0x0000u,
          "BF16 low reference: a negative denormal operand flushes to zero");
}

// ---- strata_util_copy_strided -----------------------------------------------------------------------------------

// The copy's oracle: destination element (r, c) equals source element r * src_stride + c, bit for bit. A zero
// src_stride repeats source row 0. The dst_stride padding after each row keeps the 0xFF scratch poison.
bool run_copy(Context& context, const std::vector<float>& source, std::uint32_t rows, std::uint32_t src_stride,
              std::uint32_t dst_stride, const twin::Options& options) {
    std::string err;
    const std::size_t src_floats =
        src_stride == 0 ? kCols : (rows - 1) * static_cast<std::size_t>(src_stride) + kCols;
    const std::size_t dst_floats = (rows - 1) * static_cast<std::size_t>(dst_stride) + kCols;
    const std::string tag = "copy_strided T=" + std::to_string(rows) +
                            " src_stride=" + std::to_string(src_stride) +
                            " dst_stride=" + std::to_string(dst_stride);

    std::vector<std::uint8_t> expected(dst_floats * 4, 0xffu);
    std::vector<twin::ByteRange> written;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < kCols; ++c) {
            const std::uint32_t bits = bits_of(source[r * static_cast<std::size_t>(src_stride) + c]);
            std::memcpy(&expected[(r * static_cast<std::size_t>(dst_stride) + c) * 4], &bits, sizeof(bits));
        }
        written.push_back({static_cast<std::size_t>(r) * dst_stride * 4, kCols * 4});
    }

    twin::GuardedBuffer src;
    twin::GuardedBuffer count;
    twin::GuardedBuffer dst;
    if (!allocate(context, src, "util copy source", src_floats * 4, err) ||
        !allocate(context, count, "util copy rows", 4, err) ||
        !allocate(context, dst, "util copy output", dst_floats * 4, err)) {
        check(false, tag + " allocation", err);
        return false;
    }
    std::memcpy(src.payload().data(), source.data(), src_floats * 4);
    store_u32(count.payload(), rows);
    // Initial contents: every written element starts with one flipped bit; the padding keeps the 0xFF poison.
    std::vector<std::uint8_t> initial = expected;
    for (const auto& range : written) {
        for (std::size_t offset = range.offset; offset < range.offset + range.length; offset += 4) {
            initial[offset] ^= 0x01u;
        }
    }
    std::memcpy(dst.payload().data(), initial.data(), initial.size());
    src.capture_before_dispatch();
    count.capture_before_dispatch();
    dst.capture_before_dispatch();

    const CopyArgs args{static_cast<std::uint32_t>(kCols), src_stride, dst_stride};
    if (!dispatch_util(context, "strata_util_copy_strided",
                       {{0, src.view()}, {1, dst.view()}, {2, count.view()}}, &args, sizeof(args), 3,
                       static_cast<std::uint64_t>(rows) * kCols, err)) {
        check(false, tag + " dispatch", err);
        return false;
    }
    bool ok = true;
    if (const auto failure = twin::compare_bytes(dst.payload(), expected)) {
        twin::report_failure(tag.c_str(), options, rows, *failure);
        ok = false;
    }
    const std::vector<twin::ByteRange> none;
    if (dst.first_unwritten_change(written)) ok = false;
    if (src.first_unwritten_change(none) || count.first_unwritten_change(none)) ok = false;

    // Row independence: a one-row dispatch of row t, from the same source row, must equal row t of the T-row run.
    std::vector<std::uint8_t> single_rows(static_cast<std::size_t>(rows) * kCols * 4);
    std::vector<std::uint8_t> many_rows(static_cast<std::size_t>(rows) * kCols * 4);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::size_t row_offset = static_cast<std::size_t>(r) * dst_stride * 4;
        std::memcpy(&many_rows[static_cast<std::size_t>(r) * kCols * 4], dst.payload().data() + row_offset,
                    kCols * 4);
        const std::vector<std::uint8_t> row_expected(expected.begin() + row_offset,
                                                     expected.begin() + row_offset + kCols * 4);
        twin::GuardedBuffer one;
        twin::GuardedBuffer one_count;
        if (!allocate(context, one, "util copy one-row output", kCols * 4, err) ||
            !allocate(context, one_count, "util copy one-row rows", 4, err)) {
            check(false, tag + " one-row allocation", err);
            return false;
        }
        const auto one_poison = flipped_bits(row_expected, 4);
        std::memcpy(one.payload().data(), one_poison.data(), one_poison.size());
        store_u32(one_count.payload(), 1);
        one.capture_before_dispatch();
        one_count.capture_before_dispatch();
        const std::size_t source_offset = src_stride == 0 ? 0 : static_cast<std::size_t>(r) * src_stride * 4;
        const CopyArgs one_args{static_cast<std::uint32_t>(kCols), src_stride, dst_stride};
        if (!dispatch_util(context, "strata_util_copy_strided",
                           {{0, src.view().slice(source_offset, kCols * 4)},
                            {1, one.view()},
                            {2, one_count.view()}},
                           &one_args, sizeof(one_args), 3, kCols, err)) {
            check(false, tag + " one-row dispatch", err);
            return false;
        }
        if (const auto failure = twin::compare_bytes(one.payload(), row_expected)) {
            twin::report_failure((tag + " one-row").c_str(), options, 1, *failure);
            ok = false;
        }
        if (one.first_unwritten_change(std::vector<twin::ByteRange>{{0, kCols * 4}})) ok = false;
        std::memcpy(&single_rows[static_cast<std::size_t>(r) * kCols * 4], one.payload().data(), kCols * 4);
    }
    if (const auto failure = twin::row_t_independent(single_rows, many_rows, kCols * 4, rows, 4)) {
        twin::report_failure((tag + " row_t_independent").c_str(), options, rows, *failure);
        ok = false;
    }
    check(ok, tag + " matches the bit-exact copy oracle, guards and row independence");
    return ok;
}

// ---- strata_util_to_f16 -----------------------------------------------------------------------------------------

bool run_to_f16(Context& context, const std::vector<float>& values, std::uint32_t rows,
                const twin::Options& options) {
    std::string err;
    const std::size_t n = static_cast<std::size_t>(rows) * kCols;
    const std::string tag = "to_f16 T=" + std::to_string(rows);
    std::vector<std::uint16_t> expected(n);
    for (std::size_t i = 0; i < n; ++i) expected[i] = half_reference(values[i]);
    const auto expected_bytes = bytes_of_u16(expected);

    twin::GuardedBuffer x;
    twin::GuardedBuffer count;
    twin::GuardedBuffer y;
    if (!allocate(context, x, "util to_f16 input", n * 4, err) ||
        !allocate(context, count, "util to_f16 rows", 4, err) ||
        !allocate(context, y, "util to_f16 output", n * 2, err)) {
        check(false, tag + " allocation", err);
        return false;
    }
    std::memcpy(x.payload().data(), values.data(), n * 4);
    store_u32(count.payload(), rows);
    if (!y.poison_bytes_against(expected_bytes, err)) {
        check(false, tag + " poison", err);
        return false;
    }
    x.capture_before_dispatch();
    count.capture_before_dispatch();
    y.capture_before_dispatch();

    const std::uint32_t cols = static_cast<std::uint32_t>(kCols);
    if (!dispatch_util(context, "strata_util_to_f16", {{0, x.view()}, {1, y.view()}, {2, count.view()}},
                       &cols, sizeof(cols), 3, n, err)) {
        check(false, tag + " dispatch", err);
        return false;
    }
    bool ok = true;
    if (const auto failure = twin::compare_bytes(y.payload(), expected_bytes)) {
        twin::report_failure(tag.c_str(), options, rows, *failure);
        ok = false;
    }
    const std::vector<twin::ByteRange> none;
    if (x.first_unwritten_change(none) || count.first_unwritten_change(none)) ok = false;
    if (y.first_unwritten_change(std::vector<twin::ByteRange>{{0, n * 2}})) ok = false;

    std::vector<std::uint8_t> single_rows(n * 2);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::size_t row = static_cast<std::size_t>(r) * kCols;
        const std::vector<std::uint8_t> row_expected(expected_bytes.begin() + row * 2,
                                                     expected_bytes.begin() + (row + kCols) * 2);
        twin::GuardedBuffer one_count;
        twin::GuardedBuffer one_y;
        if (!allocate(context, one_y, "util to_f16 one-row output", kCols * 2, err) ||
            !allocate(context, one_count, "util to_f16 one-row rows", 4, err)) {
            check(false, tag + " one-row allocation", err);
            return false;
        }
        std::memcpy(one_y.payload().data(), flipped_bits(row_expected, 2).data(), kCols * 2);
        store_u32(one_count.payload(), 1);
        one_y.capture_before_dispatch();
        one_count.capture_before_dispatch();
        if (!dispatch_util(context, "strata_util_to_f16",
                           {{0, x.view().slice(row * 4, kCols * 4)}, {1, one_y.view()}, {2, one_count.view()}},
                           &cols, sizeof(cols), 3, kCols, err)) {
            check(false, tag + " one-row dispatch", err);
            return false;
        }
        if (const auto failure = twin::compare_bytes(one_y.payload(), row_expected)) {
            twin::report_failure((tag + " one-row").c_str(), options, 1, *failure);
            ok = false;
        }
        std::memcpy(&single_rows[row * 2], one_y.payload().data(), kCols * 2);
    }
    if (const auto failure = twin::row_t_independent(single_rows, y.payload(), kCols * 2, rows, 2)) {
        twin::report_failure((tag + " row_t_independent").c_str(), options, rows, *failure);
        ok = false;
    }
    check(ok, tag + " matches the A.10 float16 oracle, guards and row independence");
    return ok;
}

// ---- strata_util_to_bf16 ----------------------------------------------------------------------------------------

// Checks one output array. A NaN-class expectation accepts any NaN of the BF16 format; its initial contents are a
// finite value, so an unwritten element fails.
std::optional<twin::Failure> compare_bf16_low(std::span<const std::uint16_t> actual,
                                              std::span<const LowExpectation> expected) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].unmodelled) {
            return twin::Failure{i, actual[i], 0, "low residual is a float32 denormal; its flush is not modelled"};
        }
        const bool matches = expected[i].nan ? is_bf16_nan(actual[i]) : actual[i] == expected[i].bits;
        if (!matches) {
            return twin::Failure{i, actual[i], expected[i].bits,
                                 expected[i].nan ? "BF16 low part is not a NaN" : "BF16 low part differs"};
        }
    }
    return std::nullopt;
}

bool run_to_bf16(Context& context, const std::vector<float>& values, std::uint32_t rows, bool has_lo,
                 const twin::Options& options) {
    std::string err;
    const std::size_t n = static_cast<std::size_t>(rows) * kCols;
    const std::string tag = "to_bf16 T=" + std::to_string(rows) + (has_lo ? " lo" : " no-lo");
    std::vector<std::uint16_t> hi(n);
    std::vector<LowExpectation> lo(n);
    for (std::size_t i = 0; i < n; ++i) {
        hi[i] = bf16_reference(values[i]);
        lo[i] = low_reference(values[i], hi[i]);
    }
    const auto hi_bytes = bytes_of_u16(hi);
    // A low element with a NaN-class expectation starts at zero, which is finite and therefore not a NaN.
    std::vector<std::uint16_t> lo_initial(n);
    for (std::size_t i = 0; i < n; ++i) lo_initial[i] = lo[i].nan ? 0u : static_cast<std::uint16_t>(lo[i].bits ^ 1u);

    twin::GuardedBuffer x;
    twin::GuardedBuffer count;
    twin::GuardedBuffer y;
    twin::GuardedBuffer ylo;
    if (!allocate(context, x, "util to_bf16 input", n * 4, err) ||
        !allocate(context, count, "util to_bf16 rows", 4, err) ||
        !allocate(context, y, "util to_bf16 output", n * 2, err) ||
        !allocate(context, ylo, "util to_bf16 low output", n * 2, err)) {
        check(false, tag + " allocation", err);
        return false;
    }
    std::memcpy(x.payload().data(), values.data(), n * 4);
    store_u32(count.payload(), rows);
    if (!y.poison_bytes_against(hi_bytes, err)) {
        check(false, tag + " poison", err);
        return false;
    }
    if (has_lo) {
        std::memcpy(ylo.payload().data(), lo_initial.data(), n * 2);
    } else {
        std::memset(ylo.payload().data(), 0xff, n * 2);  // scratch poison: the low output must not be written
    }
    x.capture_before_dispatch();
    count.capture_before_dispatch();
    y.capture_before_dispatch();
    ylo.capture_before_dispatch();

    const BF16Args args{static_cast<std::uint32_t>(kCols), has_lo ? 1u : 0u};
    if (!dispatch_util(context, "strata_util_to_bf16",
                       {{0, x.view()}, {1, y.view()}, {2, ylo.view()}, {3, count.view()}},
                       &args, sizeof(args), 4, n, err)) {
        check(false, tag + " dispatch", err);
        return false;
    }
    bool ok = true;
    const std::vector<twin::ByteRange> none;
    if (const auto failure = twin::compare_bytes(y.payload(), hi_bytes)) {
        twin::report_failure((tag + " high").c_str(), options, rows, *failure);
        ok = false;
    }
    if (y.first_unwritten_change(std::vector<twin::ByteRange>{{0, n * 2}})) ok = false;
    if (x.first_unwritten_change(none) || count.first_unwritten_change(none)) ok = false;
    if (has_lo) {
        const auto actual = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(ylo.payload().data()), n);
        if (const auto failure = compare_bf16_low(actual, lo)) {
            twin::report_failure((tag + " low").c_str(), options, rows, *failure);
            ok = false;
        }
        if (ylo.first_unwritten_change(std::vector<twin::ByteRange>{{0, n * 2}})) ok = false;
    } else {
        if (ylo.first_unwritten_change(none)) ok = false;
    }

    if (has_lo) {
        std::vector<std::uint8_t> single_hi(n * 2);
        std::vector<std::uint8_t> single_lo(n * 2);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::size_t row = static_cast<std::size_t>(r) * kCols;
            twin::GuardedBuffer one_count;
            twin::GuardedBuffer one_y;
            twin::GuardedBuffer one_lo;
            const std::vector<std::uint16_t> row_hi(hi.begin() + row, hi.begin() + row + kCols);
            const std::vector<std::uint16_t> row_lo_initial(lo_initial.begin() + row,
                                                            lo_initial.begin() + row + kCols);
            if (!allocate(context, one_y, "util to_bf16 one-row high", kCols * 2, err) ||
                !allocate(context, one_lo, "util to_bf16 one-row low", kCols * 2, err) ||
                !allocate(context, one_count, "util to_bf16 one-row rows", 4, err)) {
                check(false, tag + " one-row allocation", err);
                return false;
            }
            const auto row_hi_bytes = bytes_of_u16(row_hi);
            std::memcpy(one_y.payload().data(), flipped_bits(row_hi_bytes, 2).data(), kCols * 2);
            std::memcpy(one_lo.payload().data(), bytes_of_u16(row_lo_initial).data(), kCols * 2);
            store_u32(one_count.payload(), 1);
            one_y.capture_before_dispatch();
            one_lo.capture_before_dispatch();
            one_count.capture_before_dispatch();
            if (!dispatch_util(context, "strata_util_to_bf16",
                               {{0, x.view().slice(row * 4, kCols * 4)},
                                {1, one_y.view()},
                                {2, one_lo.view()},
                                {3, one_count.view()}},
                               &args, sizeof(args), 4, kCols, err)) {
                check(false, tag + " one-row dispatch", err);
                return false;
            }
            const std::span<const std::uint16_t> row_low_actual(
                reinterpret_cast<const std::uint16_t*>(one_lo.payload().data()), kCols);
            if (const auto failure = compare_bf16_low(
                    row_low_actual, std::span<const LowExpectation>(lo).subspan(row, kCols))) {
                twin::report_failure((tag + " one-row low").c_str(), options, 1, *failure);
                ok = false;
            }
            if (const auto failure = twin::compare_bytes(one_y.payload(), row_hi_bytes)) {
                twin::report_failure((tag + " one-row high").c_str(), options, 1, *failure);
                ok = false;
            }
            std::memcpy(&single_hi[row * 2], one_y.payload().data(), kCols * 2);
            std::memcpy(&single_lo[row * 2], one_lo.payload().data(), kCols * 2);
        }
        if (const auto failure = twin::row_t_independent(single_hi, y.payload(), kCols * 2, rows, 2)) {
            twin::report_failure((tag + " high row_t_independent").c_str(), options, rows, *failure);
            ok = false;
        }
        if (const auto failure = twin::row_t_independent(single_lo, ylo.payload(), kCols * 2, rows, 2)) {
            twin::report_failure((tag + " low row_t_independent").c_str(), options, rows, *failure);
            ok = false;
        }
    }
    check(ok, tag + " matches the A.10 BF16 oracle, guards and row independence");
    return ok;
}

// ---- driver --------------------------------------------------------------------------------------------------------

bool run_set(Context& context, twin::InputSet set, twin::Options options) {
    std::mt19937_64 rng(options.seed ^ (static_cast<std::uint64_t>(set) << 40));
    std::string err;
    const std::size_t count = kMaxRows * kCols;
    const std::vector<float> values = make_inputs(set, count, rng);
    check(values.size() == count, std::string("synthesize ") + twin::input_set_name(set) + " inputs");

    // The copy source is long enough for the largest strided case (T = 8, padded strides).
    const std::vector<float> copy_source = make_inputs(set, 8 * (kCols + 64) + kCols, rng);
    bool completed = true;
    for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows) {
        completed &= run_copy(context, copy_source, rows, static_cast<std::uint32_t>(kCols),
                              static_cast<std::uint32_t>(kCols), options);
        completed &= run_copy(context, copy_source, rows, 0u, static_cast<std::uint32_t>(kCols), options);
        completed &= run_copy(context, copy_source, rows, static_cast<std::uint32_t>(kCols + 32),
                              static_cast<std::uint32_t>(kCols + 64), options);
        completed &= run_to_f16(context, values, rows, options);
        completed &= run_to_bf16(context, values, rows, true, options);
    }
    completed &= run_to_bf16(context, values, kMaxRows, false, options);
    return completed;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: metal_util_test strata.metallib [--seed N] [--input-set random|edge|fixture]\n");
        return 2;
    }

    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    if (options.input_set_explicit && options.input_set == twin::InputSet::real) {
        std::printf("SKIP: metal_util_test has no real set (the utilities read no model tensors)\n");
        return 77;
    }

    test_reference_anchors();
    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;

    std::vector<twin::InputSet> sets;
    if (options.input_set_explicit) {
        sets.push_back(options.input_set);
    } else {
        sets = {twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
    }
    for (const twin::InputSet set : sets) {
        if (!run_set(*context, set, options)) ++g_failures;
    }
    std::printf("metal_util_test: %s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
