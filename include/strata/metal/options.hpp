// include/strata/metal/options.hpp - command-line options shared by the Metal driver and its host tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strata::metal {

enum class OptionDisposition : std::uint8_t {
    Keep,
    Mac,
    Ignore,
    New,
};

struct OptionSpec {
    std::string_view name;
    std::uint8_t value_count;
    OptionDisposition disposition;
    bool warn_when_ignored;
};

/// The full option inventory and the documented Metal disposition for each flag.
const std::vector<OptionSpec>& option_specs() noexcept;
const OptionSpec* find_option(std::string_view name) noexcept;

struct Options {
    std::unordered_map<std::string, std::vector<std::string>> values;
    std::unordered_set<std::string> present;
    std::vector<std::string> warnings;

    bool has(std::string_view name) const;
    std::string_view value(std::string_view name, std::size_t index = 0) const;
    std::string_view last_value(std::string_view name) const;
};

struct OptionParseResult {
    Options options;
    int exit_code = 0;
    std::string error;
};

/// Parse argv entries excluding argv[0]. Argument errors use exit code 2.
OptionParseResult parse_options(const std::vector<std::string_view>& arguments);
void print_usage(std::FILE* stream);

}  // namespace strata::metal
