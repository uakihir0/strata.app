// Shared Metal twin harness primitives (docs/mac/04-metal-kernels.md §10).
#pragma once

#include "oracles/common.hpp"
#include "strata/metal/context.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace twin {

enum class InputSet { random, edge, fixture, real };

const char* input_set_name(InputSet set) noexcept;
std::optional<InputSet> parse_input_set(const std::string& name) noexcept;

struct Options {
    std::uint64_t seed = 1;
    InputSet input_set = InputSet::random;
    bool input_set_explicit = false;
};

bool parse_options(int argc, char** argv, int first_option, Options& options, std::string& err);

struct ByteRange {
    std::size_t offset = 0;
    std::size_t length = 0;
};

struct Failure {
    std::size_t index = 0;
    std::uint64_t actual = 0;
    std::uint64_t expected = 0;
    std::string reason;
};

enum class Comparison { bits, units, relative_l2 };

/// A 4 KiB poison guard on each side of the logical view, backed by a Context-owned shared buffer.
class GuardedBuffer {
public:
    static constexpr std::size_t kGuardBytes = 4096;

    bool allocate(strata::metal::Context& context, const std::string& label,
                  std::size_t payload_bytes, std::string& err);
    strata::metal::BufferRef view() const noexcept { return view_; }
    std::span<std::uint8_t> payload() noexcept;
    std::span<const std::uint8_t> payload() const noexcept;
    std::span<const std::uint8_t> allocation() const noexcept;

    bool poison_bytes_against(std::span<const std::uint8_t> expected, std::string& err);
    bool poison_floats_against(std::span<const tf> expected, Comparison comparison, std::string& err);
    bool poison_floats_against(std::span<const double> expected, Comparison comparison, std::string& err);

    void capture_before_dispatch();
    std::optional<Failure> first_unwritten_change(std::span<const ByteRange> written) const;

private:
    strata::metal::BufferRef storage_{};
    strata::metal::BufferRef view_{};
    std::size_t payload_bytes_ = 0;
    std::vector<std::uint8_t> before_;
};

struct Binding {
    unsigned index = 0;
    strata::metal::BufferRef buffer{};
};

struct GridSize {
    std::uint32_t x = 1;
    std::uint32_t y = 1;
    std::uint32_t z = 1;
};

enum class GridMode { threads, threadgroups };

struct Dispatch {
    void* pipeline = nullptr;  // id<MTLComputePipelineState>, including a test-only runtime pipeline
    std::string label;
    std::vector<Binding> buffers;
    GridMode grid_mode = GridMode::threads;
    GridSize grid{};
    GridSize threads_per_group{};
    std::size_t threadgroup_memory_bytes = 0;
    unsigned threadgroup_memory_index = 0;
    const void* parameters = nullptr;
    std::size_t parameter_bytes = 0;
    unsigned parameter_index = 0;
};

bool dispatch(strata::metal::Context& context, const Dispatch& work, std::string& err);
bool dispatch_manifest(strata::metal::Context& context, const char* kernel_name,
                       Dispatch work, std::string& err);

struct Tolerance {
    Comparison comparison = Comparison::bits;
    double n = 0.0;
    double epsilon = 0.0;
    std::size_t row_width = 0;  // required for relative_l2
};

std::optional<Failure> compare_bytes(std::span<const std::uint8_t> actual,
                                     std::span<const std::uint8_t> expected);
std::optional<Failure> compare_floats(std::span<const float> actual,
                                      std::span<const tf> expected,
                                      Tolerance tolerance);
std::optional<Failure> compare_relative_l2(std::span<const float> actual,
                                           std::span<const double> expected,
                                           double epsilon, std::size_t row_width);
std::optional<Failure> row_t_independent(std::span<const std::uint8_t> single_row_outputs,
                                         std::span<const std::uint8_t> many_rows,
                                         std::size_t row_bytes, std::uint32_t rows,
                                         std::size_t element_bytes = 1);

void report_failure(const char* label, const Options& options, std::uint32_t T,
                    const Failure& failure);

}  // namespace twin
