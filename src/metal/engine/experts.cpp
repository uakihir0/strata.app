// src/metal/engine/experts.cpp - tiered expert reads, lookahead and DONE counters.
#include "strata/metal/experts.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <cstdlib>
#include <exception>
#include <future>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <pthread/qos.h>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

class FetchPool {
public:
    explicit FetchPool(std::size_t worker_count) {
        workers_.reserve(worker_count);
        try {
            for (std::size_t i = 0; i < worker_count; ++i) {
                workers_.emplace_back([this] { run(); });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~FetchPool() { stop(); }
    FetchPool(const FetchPool&) = delete;
    FetchPool& operator=(const FetchPool&) = delete;

    std::future<std::string> submit(std::function<std::string()> function) {
        auto task = std::make_shared<std::packaged_task<std::string()>>(
            std::move(function));
        std::future<std::string> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("expert fetch pool is stopping");
            tasks_.emplace_back([task] { (*task)(); });
        }
        ready_.notify_one();
        return result;
    }

private:
    void run() {
        (void)::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        workers_.clear();
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

std::size_t fetch_worker_count() {
    const char* configured = std::getenv("STRATA_FETCH_THREADS");
    if (configured == nullptr || configured[0] == '\0') return kExpertFetchWorkers;
    std::uint64_t parsed = 0;
    const char* end = configured;
    while (*end != '\0') ++end;
    const auto result = std::from_chars(configured, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) return 1;
    return static_cast<std::size_t>(std::clamp<std::uint64_t>(parsed, 1, 64));
}

struct PendingExpert {
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;
    std::size_t layout_index = 0;
    SlotPool::Lease lease;
    bool cache_hit = false;
    std::array<void*, 3> destinations{};
};

struct PendingRead {
    std::size_t expert_index = 0;
    std::uint32_t role = 0;
    std::future<std::string> result;
};

std::string expert_error(std::uint32_t layer, std::uint32_t expert,
                         const std::string& detail) {
    return "ERR expert read failed: layer " + std::to_string(layer) +
           " expert " + std::to_string(expert) + ": " + detail;
}

}  // namespace

struct ExpertLeaseSet::Impl {
    std::mutex mutex;
    std::array<std::vector<SlotPool::Lease>, kTieredMaxLayers> leases;
    std::array<bool, kTieredMaxLayers> retired_layers{};
    bool completed = false;
};

ExpertLeaseSet::ExpertLeaseSet(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExpertLeaseSet::~ExpertLeaseSet() = default;

bool ExpertLeaseSet::retain(std::uint32_t layer,
                            std::vector<SlotPool::Lease>& leases,
                            std::string& error) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->completed) {
        error = "expert command buffer completed before its plan was resolved";
        return false;
    }
    if (layer >= impl_->leases.size()) {
        error = "expert command-buffer layer is out of range";
        return false;
    }
    if (impl_->retired_layers[layer]) {
        error = "expert command-buffer layer was retired before its plan was resolved";
        return false;
    }
    std::vector<SlotPool::Lease>& retained = impl_->leases[layer];
    if (leases.size() > retained.max_size() - retained.size()) {
        error = "expert command-buffer lease count overflows";
        return false;
    }
    try {
        retained.reserve(retained.size() + leases.size());
    } catch (const std::exception& exception) {
        error = "cannot retain expert slots: " + std::string(exception.what());
        return false;
    } catch (...) {
        error = "cannot retain expert slots";
        return false;
    }
    for (SlotPool::Lease& lease : leases) {
        retained.emplace_back(std::move(lease));
    }
    leases.clear();
    error.clear();
    return true;
}

void ExpertLeaseSet::retire_layer(std::uint32_t layer) noexcept {
    if (layer >= impl_->leases.size()) return;
    std::vector<SlotPool::Lease> released;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->retired_layers[layer] = true;
        released.swap(impl_->leases[layer]);
    }
}

void ExpertLeaseSet::complete() noexcept {
    std::array<std::vector<SlotPool::Lease>, kTieredMaxLayers> released;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->completed = true;
        for (std::size_t layer = 0; layer < impl_->leases.size(); ++layer) {
            released[layer].swap(impl_->leases[layer]);
            impl_->retired_layers[layer] = true;
        }
    }
}

struct ExpertStore::Impl {
    explicit Impl(std::size_t worker_count) : fetch_pool(worker_count) {}

    TieredStorage* storage = nullptr;
    std::uint32_t n_layers = 0;
    std::uint32_t n_expert = 0;
    FetchPool fetch_pool;
    std::mutex operation_mutex;

    std::size_t index(std::uint32_t layer, std::uint32_t expert) const {
        return static_cast<std::size_t>(layer) * n_expert + expert;
    }

    bool validate_layer(std::uint32_t layer, std::string& error) const {
        if (layer >= n_layers) {
            error = "tiered expert layer is out of range";
            return false;
        }
        return true;
    }

    bool acquire_for_read(std::uint32_t layer, std::uint32_t expert,
                          std::vector<PendingExpert>& pending,
                          std::string& error) {
        const std::size_t layout_index = index(layer, expert);
        const TieredExpert& metadata = storage->layout().experts[layout_index];
        SlotPool* slots = storage->slot_pool();
        if (slots == nullptr) {
            error = expert_error(layer, expert, "tiered slot pool is unavailable");
            return false;
        }
        SlotPool::Lease lease =
            slots->acquire(layer, expert, metadata.blob_bytes, error);
        if (!lease) {
            const std::string detail = error.empty() ? "cannot acquire an expert slot" : error;
            error = expert_error(layer, expert, detail);
            return false;
        }
        PendingExpert entry;
        entry.layer = layer;
        entry.expert = expert;
        entry.layout_index = layout_index;
        entry.cache_hit = lease.view().cache_hit;
        entry.lease = std::move(lease);
        pending.emplace_back(std::move(entry));
        return true;
    }

    bool prepare_destinations(PendingExpert& entry, std::string& error) const {
        const TieredExpert& metadata = storage->layout().experts[entry.layout_index];
        const ExpertSlotView& view = entry.lease.view();
        std::uint64_t offset = 0;
        for (std::uint32_t role = 0; role < metadata.roles.size(); ++role) {
            const std::uint64_t bytes = metadata.roles[role].byte_count;
            if (bytes == 0 || offset > view.byte_count ||
                bytes > view.byte_count - offset ||
                offset > std::numeric_limits<std::uint64_t>::max() - view.gpu_address ||
                offset > std::numeric_limits<std::size_t>::max()) {
                error = "tiered expert role does not fit its slot";
                return false;
            }
            entry.destinations[role] =
                static_cast<std::uint8_t*>(view.contents) +
                static_cast<std::size_t>(offset);
            offset += bytes;
        }
        if (offset != metadata.blob_bytes || offset != view.byte_count ||
            view.contents == nullptr) {
            error = "tiered expert blob and role sizes do not match its slot";
            return false;
        }
        return true;
    }

    bool read_and_publish(std::vector<PendingExpert>& pending,
                          std::string& error) {
        std::vector<PendingRead> reads;
        reads.reserve(pending.size() * 3);
        std::vector<std::string> expert_errors(pending.size());
        for (std::size_t i = 0; i < pending.size(); ++i) {
            PendingExpert& entry = pending[i];
            if (entry.cache_hit) continue;
            if (!prepare_destinations(entry, expert_errors[i])) continue;
            for (std::uint32_t role = 0; role < 3; ++role) {
                try {
                    reads.push_back({
                        i,
                        role,
                        fetch_pool.submit([this, &entry, role, destination = entry.destinations[role]] {
                            std::string read_error;
                            if (!storage->read_expert_role(
                                    entry.layer, entry.expert, role, destination,
                                    static_cast<std::size_t>(
                                        storage->layout().experts[entry.layout_index]
                                            .roles[role].byte_count),
                                    read_error)) {
                                return read_error;
                            }
                            return std::string{};
                        }),
                    });
                } catch (const std::exception& exception) {
                    expert_errors[i] = exception.what();
                    break;
                } catch (...) {
                    expert_errors[i] = "cannot queue expert read";
                    break;
                }
            }
        }

        for (PendingRead& read : reads) {
            try {
                const std::string result = read.result.get();
                if (!result.empty() && expert_errors[read.expert_index].empty()) {
                    expert_errors[read.expert_index] =
                        "role " + std::to_string(read.role) + ": " + result;
                }
            } catch (const std::exception& exception) {
                if (expert_errors[read.expert_index].empty()) {
                    expert_errors[read.expert_index] = exception.what();
                }
            } catch (...) {
                if (expert_errors[read.expert_index].empty()) {
                    expert_errors[read.expert_index] = "expert fetch worker failed";
                }
            }
        }

        for (std::size_t i = 0; i < pending.size(); ++i) {
            PendingExpert& entry = pending[i];
            if (entry.cache_hit || !expert_errors[i].empty()) {
                if (!expert_errors[i].empty() && error.empty()) {
                    error = expert_error(entry.layer, entry.expert, expert_errors[i]);
                }
                continue;
            }
            if (!entry.lease.publish(expert_errors[i])) {
                if (error.empty()) {
                    error = expert_error(entry.layer, entry.expert,
                                         expert_errors[i].empty()
                                             ? "cannot publish completed expert read"
                                             : expert_errors[i]);
                }
            }
        }
        return error.empty();
    }
};

ExpertStore::ExpertStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExpertStore::~ExpertStore() = default;

std::shared_ptr<ExpertLeaseSet> ExpertStore::prepare_command_buffer(
    CommandBuffer& command_buffer, std::string& error) {
    error.clear();
    std::shared_ptr<ExpertLeaseSet> lease_set;
    try {
        lease_set = std::shared_ptr<ExpertLeaseSet>(
            new ExpertLeaseSet(std::make_unique<ExpertLeaseSet::Impl>()));
    } catch (const std::exception& exception) {
        error = "cannot allocate expert command-buffer lease set: " +
                std::string(exception.what());
        return nullptr;
    } catch (...) {
        error = "cannot allocate expert command-buffer lease set";
        return nullptr;
    }
    if (!command_buffer.add_completion_handler(
            [lease_set](bool) { lease_set->complete(); }, error)) {
        return nullptr;
    }
    return lease_set;
}

std::unique_ptr<ExpertStore> ExpertStore::create(TieredStorage& storage,
                                                  const Geometry& geometry,
                                                  std::string& error) {
    error.clear();
    if (geometry.n_layers == 0 || geometry.n_layers > kTieredMaxLayers ||
        geometry.n_expert == 0 || geometry.n_expert > kTieredMaxExperts ||
        storage.slot_pool() == nullptr ||
        geometry.n_layers > std::numeric_limits<std::size_t>::max() /
                                geometry.n_expert ||
        storage.layout().experts.size() !=
            static_cast<std::size_t>(geometry.n_layers) * geometry.n_expert ||
        storage.layout().resident.size() <
            static_cast<std::size_t>(geometry.n_layers) * geometry.n_expert) {
        error = "ExpertStore dimensions do not match tiered storage";
        return nullptr;
    }
    for (std::uint32_t layer = 0; layer < geometry.n_layers; ++layer) {
        for (std::uint32_t expert = 0; expert < geometry.n_expert; ++expert) {
            const std::size_t index =
                static_cast<std::size_t>(layer) * geometry.n_expert + expert;
            const TieredExpert& entry = storage.layout().experts[index];
            if (entry.layer != layer || entry.expert != expert) {
                error = "ExpertStore layer/expert coordinates do not match tiered storage";
                return nullptr;
            }
        }
    }
    try {
        auto impl = std::make_unique<Impl>(fetch_worker_count());
        impl->storage = &storage;
        impl->n_layers = geometry.n_layers;
        impl->n_expert = geometry.n_expert;
        return std::unique_ptr<ExpertStore>(new ExpertStore(std::move(impl)));
    } catch (const std::exception& exception) {
        error = std::string("cannot create expert fetch workers: ") + exception.what();
        return nullptr;
    }
}

bool ExpertStore::resolve_layer(
                                const std::shared_ptr<ExpertLeaseSet>& lease_set,
                                std::uint32_t layer,
                                const TieredMoePlan& plan, ExpertLayerStats& stats,
                                std::string& error) {
    std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
    error.clear();
    if (!lease_set) {
        error = "ERR tiered expert resolution has no prepared command-buffer lease set";
        return false;
    }
    if (!impl_->validate_layer(layer, error)) return false;
    const auto invalid_plan = [&](const char* reason) {
        error = "ERR invalid tiered plan for layer " + std::to_string(layer) +
                ": " + reason;
        return false;
    };
    if (plan.n_groups > kTieredMoePlanGroups) {
        return invalid_plan("group count exceeds capacity");
    }
    if (plan.n_entries > kTieredMoePlanGroups) {
        return invalid_plan("entry count exceeds capacity");
    }
    if (plan.n_groups > plan.n_entries) {
        return invalid_plan("group count exceeds entry count");
    }
    if (plan.skip > 1) return invalid_plan("skip flag is not boolean");
    if (plan.n_entries != 0 && plan.n_entries % 10 != 0) {
        return invalid_plan("entry count is not a whole token row");
    }
    if (plan.grp_start[0] != 0 ||
        plan.grp_start[plan.n_groups] != static_cast<std::int32_t>(plan.n_entries)) {
        return invalid_plan("group range endpoints are inconsistent");
    }

    ExpertLayerStats resolved;
    resolved.lookups = plan.n_entries;
    std::vector<bool> destinations(plan.n_entries, false);
    std::vector<std::uint32_t> cold_experts;
    cold_experts.reserve(plan.n_groups);
    std::set<std::uint32_t> seen;
    const auto& layout = impl_->storage->layout();
    for (std::uint32_t group = 0; group < plan.n_groups; ++group) {
        const std::int32_t begin = plan.grp_start[group];
        const std::int32_t end = plan.grp_start[group + 1];
        const std::int32_t expert_value = plan.grp_e[group];
        const std::int32_t resident_flag = plan.grp_res[group];
        if (begin < 0 || end <= begin ||
            end > static_cast<std::int32_t>(plan.n_entries) ||
            expert_value < 0 ||
            static_cast<std::uint32_t>(expert_value) >= impl_->n_expert ||
            (resident_flag != 0 && resident_flag != 1) ||
            !seen.insert(static_cast<std::uint32_t>(expert_value)).second) {
            return invalid_plan("group bounds, expert id, residency flag, or uniqueness");
        }
        for (std::int32_t entry = begin; entry < end; ++entry) {
            const std::int32_t token = plan.ent_tok[entry];
            const std::int32_t destination = plan.ent_dst[entry];
            if (token < 0 ||
                token >= static_cast<std::int32_t>(plan.n_entries / 10) ||
                destination < 0 ||
                destination >= static_cast<std::int32_t>(plan.n_entries) ||
                destination / 10 != token ||
                destinations[static_cast<std::size_t>(destination)]) {
                error = "ERR invalid tiered plan entry index for layer " +
                        std::to_string(layer);
                return false;
            }
            destinations[static_cast<std::size_t>(destination)] = true;
        }
        const std::size_t expert_index =
            impl_->index(layer, static_cast<std::uint32_t>(expert_value));
        const bool resident = layout.resident[expert_index] >= 0;
        if ((resident_flag != 0) != resident) {
            error = "ERR tiered plan residency mismatch: layer " +
                    std::to_string(layer) + " expert " +
                    std::to_string(expert_value);
            return false;
        }
        const std::uint64_t group_lookups =
            static_cast<std::uint64_t>(end - begin);
        if (resident) {
            if (group_lookups >
                std::numeric_limits<std::uint64_t>::max() - resolved.hits) {
                error = "ERR tiered plan hit count overflows";
                return false;
            }
            resolved.hits += group_lookups;
        } else {
            cold_experts.push_back(static_cast<std::uint32_t>(expert_value));
        }
    }
    if (std::find(destinations.begin(), destinations.end(), false) !=
        destinations.end()) {
        error = "ERR incomplete tiered plan entry indices for layer " +
                std::to_string(layer);
        return false;
    }
    if (plan.n_miss != cold_experts.size() ||
        (plan.skip != 0) != cold_experts.empty() ||
        (plan.n_groups == 0 && plan.n_entries != 0)) {
        error = "ERR invalid tiered miss count for layer " + std::to_string(layer);
        return false;
    }
    if (plan.n_groups == 0 && plan.n_entries == 0) {
        stats = resolved;
        return true;
    }
    SlotPool* slots = impl_->storage->slot_pool();
    if (slots == nullptr ||
        cold_experts.size() > slots->half_capacity(layer % 2)) {
        error = "ERR tiered layer misses exceed slot-half capacity";
        return false;
    }

    std::vector<PendingExpert> pending;
    pending.reserve(cold_experts.size());
    for (const std::uint32_t expert : cold_experts) {
        if (!impl_->acquire_for_read(layer, expert, pending, error)) return false;
        PendingExpert& entry = pending.back();
        if (entry.cache_hit) {
            ++resolved.ram_blobs;
        } else if (impl_->storage->expert_pages_resident(layer, expert)) {
            ++resolved.ram_blobs;
        } else {
            ++resolved.file_blobs;
            const std::uint64_t bytes =
                layout.experts[entry.layout_index].blob_bytes;
            if (bytes > std::numeric_limits<std::uint64_t>::max() -
                            resolved.file_bytes) {
                error = expert_error(layer, expert, "file byte counter overflows");
                return false;
            }
            resolved.file_bytes += bytes;
        }
    }
    if (!impl_->read_and_publish(pending, error)) return false;

    BufferRef addresses = impl_->storage->expert_address_table();
    const std::uint64_t table_index =
        (static_cast<std::uint64_t>(layer) * impl_->n_expert) * 3;
    const std::uint64_t table_entries =
        static_cast<std::uint64_t>(impl_->n_layers) * impl_->n_expert * 3;
    if (addresses.contents == nullptr ||
        table_index > table_entries ||
        table_entries > addresses.length / sizeof(std::uint64_t)) {
        error = "ERR tiered expert address table is unavailable or undersized";
        return false;
    }

    std::vector<std::array<std::uint64_t, 3>> resolved_addresses;
    resolved_addresses.reserve(pending.size());
    std::vector<SlotPool::Lease> leases;
    leases.reserve(pending.size());
    for (PendingExpert& entry : pending) {
        const ExpertSlotView& view = entry.lease.view();
        const TieredExpert& metadata = layout.experts[entry.layout_index];
        std::array<std::uint64_t, 3> role_addresses{};
        std::uint64_t offset = 0;
        for (std::size_t role = 0; role < role_addresses.size(); ++role) {
            if (offset > std::numeric_limits<std::uint64_t>::max() -
                             view.gpu_address) {
                error = expert_error(entry.layer, entry.expert,
                                     "slot GPU address overflows");
                return false;
            }
            role_addresses[role] = view.gpu_address + offset;
            if (metadata.roles[role].byte_count >
                std::numeric_limits<std::uint64_t>::max() - offset) {
                error = expert_error(entry.layer, entry.expert,
                                     "role address offset overflows");
                return false;
            }
            offset += metadata.roles[role].byte_count;
        }
        resolved_addresses.emplace_back(role_addresses);
        leases.emplace_back(std::move(entry.lease));
    }

    if (!lease_set->retain(layer, leases, error)) {
        if (error.empty()) error = "cannot retain tiered expert slots";
        return false;
    }

    auto* table = static_cast<std::uint64_t*>(addresses.contents);
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const std::uint64_t expert_offset =
            (static_cast<std::uint64_t>(layer) * impl_->n_expert +
             pending[i].expert) * 3;
        table[expert_offset] = resolved_addresses[i][0];
        table[expert_offset + 1] = resolved_addresses[i][1];
        table[expert_offset + 2] = resolved_addresses[i][2];
    }
    std::atomic_thread_fence(std::memory_order_release);
    stats = resolved;
    return true;
}

bool ExpertStore::resolve_layer_after_plan(
    const std::shared_ptr<ExpertLeaseSet>& lease_set, std::uint32_t layer,
    Events& events, std::uint64_t plan_ready_value,
    std::uint64_t layer_ready_value, std::uint64_t timeout_ms,
    ExpertLayerStats& stats, std::string& error) {
    error.clear();
    if (!lease_set) {
        error = "ERR tiered expert resolution has no prepared command-buffer lease set";
        return false;
    }
    if (!impl_->validate_layer(layer, error)) return false;
    if (plan_ready_value == 0 ||
        plan_ready_value == std::numeric_limits<std::uint64_t>::max() ||
        layer_ready_value != plan_ready_value + 1) {
        error = "ERR tiered layer handshake requires consecutive positive event values";
        return false;
    }
    const std::optional<std::uint64_t> signaled_value = events.signaled_value();
    if (!signaled_value) {
        error = "ERR tiered layer handshake event is unavailable or was released";
        return false;
    }
    if (*signaled_value >= layer_ready_value) {
        error = "ERR tiered layer-ready event value was already signaled";
        return false;
    }

    const BufferRef plans = impl_->storage->plan_buffer();
    if (!plans || plans.contents == nullptr) {
        error = "ERR tiered shared plan buffer is unavailable";
        return false;
    }
    const std::size_t plan_offset =
        static_cast<std::size_t>(layer) * sizeof(TieredMoePlan);
    constexpr std::size_t header_bytes = 4 * sizeof(std::uint32_t);
    if (plan_offset > plans.length || header_bytes > plans.length - plan_offset) {
        error = "ERR tiered shared plan header is outside its buffer";
        return false;
    }
    if (!events.wait(plan_ready_value, timeout_ms, error)) return false;
    std::atomic_thread_fence(std::memory_order_acquire);

    std::array<std::uint32_t, 4> header{};
    const auto* plan_bytes = static_cast<const std::byte*>(plans.contents) + plan_offset;
    std::memcpy(header.data(), plan_bytes, header_bytes);
    const std::uint32_t n_groups = header[0];
    const std::uint32_t n_entries = header[1];
    const std::uint32_t n_miss = header[2];
    const std::uint32_t skip = header[3];
    const auto invalid_header = [&](const char* reason) {
        error = "ERR invalid tiered plan header for layer " +
                std::to_string(layer) + ": " + reason;
        return false;
    };
    if (n_groups == 0 || n_groups > kTieredMoePlanGroups ||
        n_entries > kTieredMoePlanGroups || n_groups > n_entries ||
        n_entries == 0 || n_entries % 10 != 0 || skip > 1) {
        return invalid_header("counts, row width, or skip flag are invalid");
    }

    ExpertLayerStats resolved;
    if (skip != 0) {
        if (n_miss != 0) return invalid_header("skipped plan reports misses");
        resolved.lookups = n_entries;
        resolved.hits = n_entries;
    } else {
        if (plan_offset > plans.length ||
            sizeof(TieredMoePlan) > plans.length - plan_offset) {
            error = "ERR tiered shared plan is outside its buffer";
            return false;
        }
        TieredMoePlan plan{};
        std::memcpy(&plan, plan_bytes, sizeof(plan));
        if (!resolve_layer(lease_set, layer, plan, resolved, error)) return false;
    }

    if (!events.signal_host(layer_ready_value, error)) return false;
    stats = resolved;
    error.clear();
    return true;
}

bool ExpertStore::prefetch_layer(std::uint32_t layer, std::string& error) {
    error.clear();
    if (!impl_->validate_layer(layer, error)) return false;
    std::vector<std::uint32_t> cold;
    cold.reserve(impl_->n_expert);
    const auto& resident = impl_->storage->layout().resident;
    for (std::uint32_t expert = 0; expert < impl_->n_expert; ++expert) {
        if (resident[impl_->index(layer, expert)] < 0) cold.push_back(expert);
    }
    return prefetch_experts(layer, cold, error);
}

bool ExpertStore::prefetch_experts(std::uint32_t layer,
                                   std::span<const std::uint32_t> experts,
                                   std::string& error) {
    std::lock_guard<std::mutex> operation_lock(impl_->operation_mutex);
    error.clear();
    if (!impl_->validate_layer(layer, error)) return false;
    std::set<std::uint32_t> unique;
    std::vector<PendingExpert> pending;
    pending.reserve(experts.size());
    SlotPool* slots = impl_->storage->slot_pool();
    if (slots == nullptr) {
        error = "tiered slot pool is unavailable";
        return false;
    }
    for (const std::uint32_t expert : experts) {
        if (expert >= impl_->n_expert) {
            error = "ERR lookahead expert id is out of range for layer " +
                    std::to_string(layer);
            return false;
        }
        if (!unique.insert(expert).second) continue;
        const std::size_t layout_index = impl_->index(layer, expert);
        if (impl_->storage->layout().resident[layout_index] >= 0) continue;
        std::string lookup_error;
        SlotPool::Lease cached = slots->lookup_ready(
            layer, expert,
            impl_->storage->layout().experts[layout_index].blob_bytes,
            lookup_error);
        if (!lookup_error.empty()) {
            error = expert_error(layer, expert, lookup_error);
            return false;
        }
        if (cached) continue;
        if (!impl_->acquire_for_read(layer, expert, pending, error)) return false;
    }
    if (pending.size() > slots->half_capacity(layer % 2)) {
        error = "ERR tiered prefetch exceeds slot-half capacity";
        return false;
    }
    return impl_->read_and_publish(pending, error);
}

}  // namespace strata::metal
