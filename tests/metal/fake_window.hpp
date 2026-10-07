// tests/metal/fake_window.hpp - deterministic serve backend for H6 transcripts.
#pragma once

#include "strata/metal/serve.hpp"

#include <chrono>
#include <string>

namespace strata::metal::test {

class FakeWindow final : public ServeBackend {
public:
    explicit FakeWindow(bool fail_gpu = false, std::chrono::milliseconds delay = {},
                        bool report_length_after_stop = false);

    const ServeInfo& serve_info() const noexcept override { return info_; }
    bool run_request(const ServeRequest& request, const ServeCallbacks& callbacks,
                     ServeOutcome& outcome, ServeFailure& failure) override;

private:
    ServeInfo info_;
    bool fail_gpu_;
    std::chrono::milliseconds delay_;
    bool report_length_after_stop_;
};

}  // namespace strata::metal::test
