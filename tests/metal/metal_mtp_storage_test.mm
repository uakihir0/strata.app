// tests/metal/metal_mtp_storage_test.mm - inspected MTP files mapped into bounded Metal windows.
#include "strata/metal/model_storage.hpp"
#include "strata/metal/mtp_storage.hpp"
#include "strata/metal/mtp_runtime_storage.hpp"
#include "strata/metal/kv.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/plan.hpp"
#include "strata/metal/window_scratch.hpp"
#include "strata/metal/weights.hpp"

#include <cerrno>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

namespace fs = std::filesystem;
using namespace strata::metal;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint32_t read_u32_le(const void* address) {
    const auto* bytes = static_cast<const unsigned char*>(address);
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

bool exclusive_lock_is_blocked(const std::string& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return false;
    const bool blocked = ::flock(descriptor, LOCK_EX | LOCK_NB) != 0 &&
                         (errno == EWOULDBLOCK || errno == EAGAIN);
    if (!blocked) (void)::flock(descriptor, LOCK_UN);
    (void)::close(descriptor);
    return blocked;
}

bool exclusive_lock_is_available(const std::string& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return false;
    const bool available = ::flock(descriptor, LOCK_EX | LOCK_NB) == 0;
    if (available) (void)::flock(descriptor, LOCK_UN);
    (void)::close(descriptor);
    return available;
}

void create_sparse_file(const fs::path& path, std::uint64_t bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "could not create replacement MTP file");
    output.close();
    require(!output.fail(), "could not close replacement MTP file");
    fs::resize_file(path, bytes);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6 || std::string(argv[2]) != "--fixture-dir-fd") {
            std::cerr << "usage: metal_mtp_storage_test METALLIB "
                         "--fixture-dir-fd FD DEV INO\n";
            return 2;
        }
        int fixture_directory_fd = -1;
        std::uint64_t expected_device = 0;
        std::uint64_t expected_inode = 0;
        const std::string_view descriptor_text(argv[3]);
        const std::string_view device_text(argv[4]);
        const std::string_view inode_text(argv[5]);
        const auto [descriptor_end, descriptor_error] = std::from_chars(
            descriptor_text.data(),
            descriptor_text.data() + descriptor_text.size(),
            fixture_directory_fd);
        const auto [device_end, device_error] = std::from_chars(
            device_text.data(), device_text.data() + device_text.size(), expected_device);
        const auto [inode_end, inode_error] = std::from_chars(
            inode_text.data(), inode_text.data() + inode_text.size(), expected_inode);
        require(descriptor_error == std::errc{} &&
                    descriptor_end == descriptor_text.data() + descriptor_text.size() &&
                    fixture_directory_fd >= 0 &&
                    device_error == std::errc{} &&
                    device_end == device_text.data() + device_text.size() &&
                    inode_error == std::errc{} &&
                    inode_end == inode_text.data() + inode_text.size(),
                "pinned fixture directory identity is invalid");
        struct stat fixture_status {};
        require(::fstat(fixture_directory_fd, &fixture_status) == 0 &&
                    S_ISDIR(fixture_status.st_mode) &&
                    static_cast<std::uint64_t>(fixture_status.st_dev) == expected_device &&
                    static_cast<std::uint64_t>(fixture_status.st_ino) == expected_inode,
                "pinned fixture directory identity changed before the Metal test");
        require(::fchdir(fixture_directory_fd) == 0,
                "could not enter the pinned fixture directory");
        const fs::path root(".");
        const fs::path native = root / "tiny.gguf";
        const fs::path mtp_directory = root / "mtp-rt";
        const fs::path experts_path = mtp_directory / "experts.bin";
        const fs::path dense_path = mtp_directory / "dense.bin";

        WeightStoreOptions options;
        options.native_shard = native.string();
        options.pack_dir = (root / "pack").string();
        options.ple_gguf = native.string();
        options.mtp_dir = mtp_directory.string();
        options.family = "qwen";
        options.model = "tiny";
        Geometry geometry;
        WindowSources window_sources;
        MtpRuntimeSources sources;
        std::string error;
        require(WeightStore::inspect(options, geometry, error, &window_sources, &sources),
                "WeightStore inspection failed: " + error);
        require(sources.catalog != nullptr && sources.generation_lock.lock_lease() != nullptr,
                "inspection did not retain its MTP catalog and generation lock");

        ContextOptions context_options;
        context_options.metallib = argv[1];
        auto context = Context::open(context_options, error);
        require(context != nullptr, "Metal Context open failed: " + error);
        auto storage = MtpStorage::open(*context, sources, error);
        require(storage != nullptr, "MTP storage open failed: " + error);

        const MtpRuntimeLayout& layout = storage->layout();
        require(layout.experts.size() == kMtpExpertCount &&
                    layout.dense_weights.size() == sources.catalog->dense_tensors.size() &&
                    layout.dense_weights.size() == 29 &&
                    layout.mapped_file_bytes ==
                        sources.catalog->experts_file_size + sources.catalog->dense_file_size,
                "MTP storage layout does not cover the inspected catalog");
        require(storage->draft_vocab_ids().size() == 106'299,
                "MTP storage returned an incomplete draft vocabulary");
        for (std::size_t index = 0; index < storage->draft_vocab_ids().size(); ++index) {
            const std::uint32_t expected_id =
                (static_cast<std::uint32_t>(index) + 17) % 106'299;
            require(storage->draft_vocab_ids()[index] == expected_id,
                    "MTP storage changed draft-vocabulary order at index " +
                        std::to_string(index));
        }

        for (std::size_t index = 0; index < layout.experts.size(); ++index) {
            const MtpExpertBinding& binding = layout.experts[index];
            const BufferRef buffer = storage->expert(static_cast<std::uint32_t>(index));
            const BufferRef window =
                storage->expert_window(static_cast<std::size_t>(binding.window_index));
            require(buffer && context->buffers().owns(buffer) &&
                        buffer.length == binding.byte_count &&
                        buffer.gpu_address == window.gpu_address + binding.window_offset &&
                        buffer.contents == static_cast<std::uint8_t*>(window.contents) +
                                               binding.window_offset &&
                        read_u32_le(buffer.contents) ==
                            (0x4d540000u | binding.expert_index),
                    "MTP expert does not resolve to its mapped blob");
        }
        for (std::size_t row = 0; row < layout.dense_weights.size(); ++row) {
            const MtpDenseWeightBinding& binding = layout.dense_weights[row];
            const BufferRef buffer = storage->dense_weight(binding.tensor_name);
            const BufferRef window =
                storage->dense_window(static_cast<std::size_t>(binding.window_index));
            require(buffer && context->buffers().owns(buffer) &&
                        buffer.length == binding.byte_count &&
                        buffer.gpu_address == window.gpu_address + binding.window_offset &&
                        buffer.contents == static_cast<std::uint8_t*>(window.contents) +
                                               binding.window_offset &&
                        read_u32_le(buffer.contents) ==
                            (0x444e0000u | static_cast<std::uint32_t>(row)),
                    "MTP dense tensor does not resolve to its mapped row: " +
                        binding.tensor_name);
        }
        require(!storage->expert(kMtpExpertCount) &&
                    !storage->dense_weight("not-a-runtime-tensor") &&
                    !storage->expert_window(layout.expert_windows.size()) &&
                    !storage->dense_window(layout.dense_windows.size()),
                "MTP storage accepted an out-of-range lookup");
        for (std::size_t index = 0; index < layout.expert_windows.size(); ++index) {
            const BufferRef window = storage->expert_window(index);
            require(window && window.length % kWindowPageBytes == 0,
                    "MTP expert window is not page aligned");
        }
        for (std::size_t index = 0; index < layout.dense_windows.size(); ++index) {
            const BufferRef window = storage->dense_window(index);
            require(window && window.length % kWindowPageBytes == 0,
                    "MTP dense window is not page aligned");
        }

        const std::size_t allocations_before_failure = context->buffers().allocation_count();
        const std::uint64_t bytes_before_failure = context->buffers().allocated_size();
        const fs::path generation_lock_path = sources.generation_lock.path;
        const fs::path generation_lock_backup =
            generation_lock_path.string() + ".original";
        fs::rename(generation_lock_path, generation_lock_backup);
        {
            std::ofstream replacement_lock(generation_lock_path,
                                           std::ios::binary | std::ios::trunc);
            require(static_cast<bool>(replacement_lock),
                    "could not create replacement MTP generation lock");
        }
        auto replaced_lock = MtpStorage::open(*context, sources, error);
        require(!replaced_lock &&
                    error.find("artifact-set identity changed after WeightStore inspection") !=
                        std::string::npos,
                "MTP storage accepted a replaced generation-lock path");
        require(context->buffers().allocation_count() == allocations_before_failure &&
                    context->buffers().allocated_size() == bytes_before_failure,
                "replaced generation lock changed Metal buffer accounting");
        fs::remove(generation_lock_path);
        fs::rename(generation_lock_backup, generation_lock_path);

        const fs::path dense_backup = mtp_directory / "dense.original";
        fs::rename(dense_path, dense_backup);
        create_sparse_file(dense_path, sources.catalog->dense_file_size);
        auto replaced_file = MtpStorage::open(*context, sources, error);
        require(!replaced_file &&
                    error.find("identity or size changed after inspection") != std::string::npos,
                "MTP storage accepted a same-size replacement file");
        require(context->buffers().allocation_count() == allocations_before_failure &&
                    context->buffers().allocated_size() == bytes_before_failure,
                "failed MTP storage open did not roll back provisional Metal buffers");
        fs::remove(dense_path);
        fs::rename(dense_backup, dense_path);

        const fs::path dense_fifo_backup = mtp_directory / "dense.fifo-backup";
        fs::rename(dense_path, dense_fifo_backup);
        require(::mkfifo(dense_path.c_str(), 0600) == 0,
                "could not create an MTP dense FIFO");
        ::alarm(2);
        auto fifo_file = MtpStorage::open(*context, sources, error);
        ::alarm(0);
        const std::string fifo_error = error;
        fs::remove(dense_path);
        fs::rename(dense_fifo_backup, dense_path);
        require(!fifo_file &&
                    fifo_error.find("MTP runtime file is not a regular file") !=
                        std::string::npos,
                "MTP storage opened a FIFO or blocked before rejecting it");
        require(context->buffers().allocation_count() == allocations_before_failure &&
                    context->buffers().allocated_size() == bytes_before_failure,
                "FIFO rejection did not roll back the mapped expert window");

        const fs::path experts_backup = mtp_directory / "experts.original";
        fs::rename(experts_path, experts_backup);
        fs::create_symlink(experts_backup.filename(), experts_path);
        auto symlink_file = MtpStorage::open(*context, sources, error);
        require(!symlink_file && error.find("cannot open MTP runtime file") != std::string::npos,
                "MTP storage followed a symlink in place of an inspected file");
        fs::remove(experts_path);
        fs::rename(experts_backup, experts_path);

        storage.reset();
        sources = {};
        window_sources = {};
        require(exclusive_lock_is_blocked(generation_lock_path),
                "destroying MtpStorage released its generation lock before Context buffers");
        context.reset();
        require(exclusive_lock_is_available(generation_lock_path),
                "Context destruction did not release the MTP generation lock");

        Geometry rollback_geometry;
        WindowSources rollback_window_sources;
        MtpRuntimeSources rollback_sources;
        require(WeightStore::inspect(options, rollback_geometry, error,
                                     &rollback_window_sources, &rollback_sources),
                "re-inspect MTP fixture for the transaction test: " + error);
        PlanOptions plan_options;
        plan_options.working_set_bytes = 1ull << 40;
        plan_options.ram_bytes = 1ull << 40;
        plan_options.available_is_override = true;
        plan_options.reserve_mib = 0;
        plan_options.max_context = 13;
        plan_options.mtp_window = 8;
        plan_options.kv = KvFormat::Int8;
        plan_options.spec_enabled = true;
        plan_options.prefill_mode = PrefillMode::Fixed;
        plan_options.prefill_chunk = 9;
        plan_options.expert_cache = 0;
        PlanResult memory_plan;
        require(make_plan(rollback_geometry, plan_options, memory_plan, error) &&
                    memory_plan.fits && memory_plan.mtp_bytes != 0 &&
                    memory_plan.mtp_bind_bytes != 0 &&
                    memory_plan.mtp_params_bytes ==
                        10ull * kMtpParamsSlotBytes &&
                    memory_plan.mtp_kv_bytes != 0,
                "could not make a fitting MTP transaction plan: " + error);

        ContextOptions rollback_context_options;
        auto rollback_context = Context::open(rollback_context_options, error);
        require(rollback_context != nullptr,
                "open transaction-test Metal Context: " + error);
        const std::size_t allocations_before_runtime =
            rollback_context->buffers().allocation_count();
        const std::uint64_t bytes_before_runtime =
            rollback_context->buffers().allocated_size();
        PlanResult invalid_plan = memory_plan;
        ++invalid_plan.mtp_kv_bytes;
        require(MtpRuntimeStorage::open(*rollback_context, rollback_geometry, rollback_sources,
                                        invalid_plan, error) == nullptr &&
                    error.find("MTP K/V bytes do not match") != std::string::npos &&
                    rollback_context->buffers().allocation_count() ==
                        allocations_before_runtime,
                "MTP runtime storage accepted a mismatched planned K/V extent");

        PlanResult invalid_params_plan = memory_plan;
        ++invalid_params_plan.mtp_params_bytes;
        require(MtpRuntimeStorage::open(*rollback_context, rollback_geometry,
                                        rollback_sources, invalid_params_plan,
                                        error) == nullptr &&
                    error.find("MTP parameter bytes do not match") !=
                        std::string::npos &&
                    rollback_context->buffers().allocation_count() ==
                        allocations_before_runtime,
                "MTP runtime storage accepted a mismatched parameter-array extent");

        Geometry mismatched_catalog_geometry = rollback_geometry;
        PlanResult mismatched_catalog_plan = memory_plan;
        ++mismatched_catalog_geometry.mtp_bind_bytes;
        ++mismatched_catalog_plan.mtp_bind_bytes;
        require(MtpRuntimeStorage::open(
                    *rollback_context, mismatched_catalog_geometry, rollback_sources,
                    mismatched_catalog_plan, error) == nullptr &&
                    error.find("does not match its vocabulary and arena layout") !=
                        std::string::npos &&
                    rollback_context->buffers().allocation_count() ==
                        allocations_before_runtime &&
                    rollback_context->buffers().allocated_size() ==
                        bytes_before_runtime,
                "MTP runtime storage accepted bind geometry inconsistent with its catalog");

        require(MtpRuntimeStorage::open_for_test(
                    *rollback_context, rollback_geometry, rollback_sources,
                    memory_plan, true, error) == nullptr &&
                    error.find("injected MTP runtime failure after weights open") !=
                        std::string::npos &&
                    rollback_context->buffers().allocation_count() ==
                        allocations_before_runtime &&
                    rollback_context->buffers().allocated_size() ==
                        bytes_before_runtime,
                "later-stage MTP failure did not roll back the opened weights");
        rollback_sources = {};
        rollback_window_sources = {};
        require(exclusive_lock_is_available(generation_lock_path),
                "failed MTP runtime open retained its generation lock");
        require(WeightStore::inspect(options, rollback_geometry, error,
                                     &rollback_window_sources, &rollback_sources) &&
                    make_plan(rollback_geometry, plan_options, memory_plan, error) &&
                    memory_plan.fits,
                "could not reacquire MTP fixture sources after injected failure: " + error);

        std::unique_ptr<KvStore> provisional_main_kv;
        {
            auto outer = rollback_context->buffers().transaction();
            auto mismatched_main_kv = KvStore::open(
                *rollback_context, memory_plan.context + 1, memory_plan.kv, error);
            require(mismatched_main_kv != nullptr &&
                        mismatched_main_kv->allocation(),
                    "open live mismatched main KV for preflight coverage: " + error);
            const std::size_t allocations_with_mismatch =
                rollback_context->buffers().allocation_count();
            const std::uint64_t bytes_with_mismatch =
                rollback_context->buffers().allocated_size();
            require(MtpRuntimeStorage::open_with_main_kv(
                        *rollback_context, *mismatched_main_kv,
                        rollback_geometry, rollback_sources, memory_plan,
                        error) == nullptr &&
                        error.find("matching main QSA KV and RoPE store") !=
                            std::string::npos &&
                        rollback_context->buffers().allocation_count() ==
                            allocations_with_mismatch &&
                        rollback_context->buffers().allocated_size() ==
                            bytes_with_mismatch,
                    "MTP runtime accepted a live main KV for a different context plan");
        }
        require(rollback_context->buffers().allocation_count() ==
                        allocations_before_runtime &&
                    rollback_context->buffers().allocated_size() ==
                        bytes_before_runtime,
                "mismatched-main-KV preflight left provisional allocations behind");
        {
            auto outer = rollback_context->buffers().transaction();
            provisional_main_kv = KvStore::open(
                *rollback_context, memory_plan.context, memory_plan.kv, error);
            require(provisional_main_kv != nullptr &&
                        provisional_main_kv->allocation().length ==
                            memory_plan.kv_bytes,
                    "open provisional main KV for stale-view preflight coverage: " + error);
        }
        require(provisional_main_kv != nullptr &&
                    !provisional_main_kv->allocation() &&
                    rollback_context->buffers().allocation_count() ==
                        allocations_before_runtime &&
                    rollback_context->buffers().allocated_size() ==
                        bytes_before_runtime,
                "rolled-back main KV remained live before MTP preflight coverage");
        require(MtpRuntimeStorage::open_with_main_kv(
                    *rollback_context, *provisional_main_kv, rollback_geometry,
                    rollback_sources, memory_plan, error) == nullptr &&
                    error.find("matching main QSA KV and RoPE store") !=
                        std::string::npos &&
                    rollback_context->buffers().allocation_count() ==
                        allocations_before_runtime &&
                    rollback_context->buffers().allocated_size() ==
                        bytes_before_runtime,
                "MTP runtime allocated resources before rejecting a stale main KV");
        provisional_main_kv.reset();
        provisional_main_kv = KvStore::open(
            *rollback_context, memory_plan.context, memory_plan.kv, error);
        require(provisional_main_kv != nullptr &&
                    provisional_main_kv->allocation().length ==
                        memory_plan.kv_bytes,
                "open committed main KV for MTP shared-RoPE binding coverage: " + error);
        const std::size_t allocations_with_main_kv =
            rollback_context->buffers().allocation_count();
        const std::uint64_t bytes_with_main_kv =
            rollback_context->buffers().allocated_size();
        std::unique_ptr<MtpRuntimeStorage> provisional_runtime;
        BufferRef provisional_main_rope;
        BufferRef provisional_shared_rope;
        MtpKvViews provisional_kv;
        BufferRef provisional_expert;
        BufferRef provisional_dense;
        BufferRef provisional_bind_view;
        BufferRef provisional_params_slot;
        {
            auto outer = rollback_context->buffers().transaction();
            provisional_main_rope = provisional_main_kv->rope();
            provisional_runtime = MtpRuntimeStorage::open_with_main_kv(
                *rollback_context, *provisional_main_kv, rollback_geometry,
                rollback_sources, memory_plan, error);
            require(static_cast<bool>(provisional_main_rope),
                    "main KvStore did not expose its shared RoPE view");
            require(provisional_runtime != nullptr &&
                        provisional_runtime->is_valid(),
                    "open MTP runtime with the borrowed main RoPE view: " + error);
            provisional_shared_rope = provisional_runtime->shared_rope();
            require(provisional_shared_rope &&
                        provisional_shared_rope.handle == provisional_main_rope.handle &&
                        provisional_shared_rope.contents == provisional_main_rope.contents &&
                        provisional_shared_rope.gpu_address ==
                            provisional_main_rope.gpu_address &&
                        provisional_shared_rope.length == provisional_main_rope.length &&
                        provisional_shared_rope.offset == provisional_main_rope.offset &&
                        rollback_context->buffers().allocation_identity(
                            provisional_shared_rope) != 0 &&
                        rollback_context->buffers().allocation_identity(
                            provisional_shared_rope) ==
                            rollback_context->buffers().allocation_identity(
                                provisional_main_rope),
                    "MTP runtime did not borrow the exact main RoPE subview");
            const MtpBindLayout& bind_layout =
                provisional_runtime->bind_layout();
            require(bind_layout.allocation_bytes ==
                            memory_plan.mtp_bind_bytes &&
                        bind_layout.capacity == memory_plan.mtp_bind_capacity &&
                        bind_layout.draft_vocab_count == 106'299 &&
                        bind_layout.output_vocab_count == 106'299 &&
                        bind_layout.region(MtpBindRegionId::hl).bytes ==
                            106'299ull * sizeof(float) &&
                        bind_layout.region(MtpBindRegionId::draft_vocab_ids).bytes ==
                            106'299ull * sizeof(std::uint32_t) &&
                        bind_layout.reserved_bytes > 0,
                    "MTP runtime bind layout does not match the inspected draft vocabulary");
            require(provisional_runtime->layout().mapped_file_bytes ==
                            memory_plan.mtp_bytes &&
                        provisional_runtime->draft_vocab_ids().size() == 106'299,
                    "MTP runtime owner does not match the planned weights and bind layout");
            const MtpParamsLayout& params_layout =
                provisional_runtime->params_layout();
            require(params_layout.prefill_group_slots == 2 &&
                        params_layout.slot_count == 10 &&
                        params_layout.allocation_bytes ==
                            memory_plan.mtp_params_bytes,
                    "MTP parameter slots do not match the planned chunk");
            const BufferRef params_slot_0 =
                provisional_runtime->mtp_params_slot(0);
            const BufferRef params_slot_7 =
                provisional_runtime->mtp_params_slot(7);
            const BufferRef params_prefill_0 =
                provisional_runtime->mtp_params_slot(8);
            provisional_params_slot =
                provisional_runtime->mtp_params_slot(9);
            require(params_slot_0 && params_slot_7 && params_prefill_0 &&
                        provisional_params_slot &&
                        params_slot_0.length == kMtpParamsSlotBytes &&
                        params_slot_7.offset - params_slot_0.offset ==
                            7 * kMtpParamsSlotBytes &&
                        params_prefill_0.offset - params_slot_0.offset ==
                            8 * kMtpParamsSlotBytes &&
                        provisional_params_slot.offset - params_slot_0.offset ==
                            9 * kMtpParamsSlotBytes &&
                        !provisional_runtime->mtp_params_slot(10),
                    "MTP runtime exposes only bounded, correctly indexed parameter slots");
            const auto* params_bytes =
                static_cast<const std::uint8_t*>(params_slot_0.contents);
            require(params_bytes != nullptr && params_bytes[0] == 0 &&
                        params_bytes[kMtpParamsSlotBytes - 1] == 0 &&
                        static_cast<const std::uint8_t*>(
                            provisional_params_slot.contents)[0] == 0,
                    "MTP parameter slots begin zero-initialized");
            BufferRef projection_table =
                rollback_context->buffers().allocate(
                    "MTP projection binding test embedding table",
                    8u * N_EMBD * sizeof(std::uint16_t), error);
            BufferRef projection_residual =
                rollback_context->buffers().allocate(
                    "MTP projection binding test residual",
                    8u * HC_DIM * sizeof(float), error);
            require(projection_table && projection_residual,
                    "allocate MTP projection binding inputs: " + error);
            std::memset(projection_table.contents, 0, projection_table.length);
            std::memset(projection_residual.contents, 0,
                        projection_residual.length);
            MtpParams projection_params{};
            projection_params.T = 5u;
            projection_params.hid[0] = 8u;
            projection_params.hid[1] = 8u;
            projection_params.hid[2] = 4u;
            std::memcpy(provisional_params_slot.contents, &projection_params,
                        sizeof(projection_params));
            MtpInputProjectionSources projection_sources{
                9u,
                MtpInputTokenSource::params_tokens,
                {},
                EmbedRowsFormat::bf16,
                8u,
                rollback_context->buffers().read_only(projection_table),
                projection_residual,
            };
            MtpInputProjectionBuffers projection_buffers;
            require(provisional_runtime->bind_input_projection(
                        *rollback_context, projection_sources,
                        projection_buffers, error),
                    "bind MTP stage-1–5 runtime views: " + error);
            const BufferRef projection_e2 =
                provisional_runtime->bind_region(MtpBindRegionId::e2);
            const std::size_t projection_active_e2_bytes =
                5u * N_EMBD * sizeof(float);
            require(projection_buffers.params.contents() ==
                            provisional_params_slot.contents &&
                        projection_buffers.token_embedding_table.contents() ==
                            projection_table.contents &&
                        projection_buffers.residual.handle ==
                            projection_residual.handle &&
                        projection_buffers.embedding.handle ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::emb).handle &&
                        projection_buffers.embedding_norm.handle ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::en).handle &&
                        projection_buffers.xq.handle ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::xq).handle &&
                        projection_buffers.hidden_norm.handle ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::hn).handle &&
                        projection_buffers.h2.handle ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::h2).handle &&
                        projection_buffers.e2.handle == projection_e2.handle &&
                        projection_buffers.matvec_sink.handle ==
                            projection_e2.handle &&
                        projection_buffers.matvec_sink.offset ==
                            projection_e2.offset +
                                projection_active_e2_bytes &&
                        projection_buffers.matvec_sink.length >=
                            3u * N_EMBD * sizeof(float) &&
                        projection_buffers.fc_embedding_weights.contents() ==
                            provisional_runtime->dense_weight(
                                "fc_embedding.weight").contents &&
                        projection_buffers.embedding_norm_gamma.contents() ==
                            provisional_runtime->dense_weight(
                                "pre_fc_norm_embedding.weight").contents &&
                        projection_buffers.hidden_norm_gamma.contents() ==
                            provisional_runtime->dense_weight(
                                "pre_fc_norm_hidden.weight").contents &&
                        projection_buffers.fc_hidden_weights.contents() ==
                            provisional_runtime->dense_weight(
                                "fc_hidden.weight").contents,
                    "MTP stage-1–5 weights, parameter and T=5 scratch views are resolved");
            MtpAddStreamsBuffers t5_add_streams_buffers;
            MtpAttentionReadBuffers t5_attention_read_buffers;
            MtpAttentionKvProjectionBuffers t5_kv_projection_buffers;
            require(provisional_runtime->bind_add_streams(
                        *rollback_context, 9u, t5_add_streams_buffers,
                        error) &&
                        provisional_runtime->bind_attention_read(
                            *rollback_context, 9u,
                            t5_attention_read_buffers, error) &&
                        provisional_runtime->bind_attention_kv_projection(
                            *rollback_context, 9u,
                            t5_kv_projection_buffers, error),
                    "bind MTP stages 6–8 at T=5: " + error);
            const BufferRef attention_h2 =
                provisional_runtime->bind_region(MtpBindRegionId::h2);
            const BufferRef attention_bo =
                provisional_runtime->bind_region(MtpBindRegionId::bo);
            const auto same_view = [](const BufferRef& actual,
                                      const BufferRef& expected) {
                return actual && expected &&
                       actual.handle == expected.handle &&
                       actual.offset == expected.offset &&
                       actual.length == expected.length &&
                       actual.contents == expected.contents &&
                       actual.gpu_address == expected.gpu_address;
            };
            require(same_view(
                            t5_add_streams_buffers.output,
                            provisional_runtime->bind_region(
                                MtpBindRegionId::mtp_rin)) &&
                        same_view(t5_attention_read_buffers.residual,
                                  t5_add_streams_buffers.output) &&
                        same_view(t5_attention_read_buffers.dead_h2_scratch,
                                  attention_h2) &&
                        same_view(t5_attention_read_buffers.normalized,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::xn)) &&
                        same_view(t5_attention_read_buffers.rs,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::rs)) &&
                        same_view(t5_attention_read_buffers.lo,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::lo)) &&
                        same_view(t5_attention_read_buffers.inject_out,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::inj)) &&
                        t5_attention_read_buffers.norm.contents ==
                            provisional_runtime->dense_weight(
                                "attn_hyper_connection.hc_norm.weight").contents &&
                        t5_attention_read_buffers.hc_down.contents ==
                            provisional_runtime->dense_weight(
                                "attn_hyper_connection.input_mix_weight_down.weight").contents &&
                        t5_attention_read_buffers.hc_inject.contents ==
                            provisional_runtime->dense_weight(
                                "attn_hyper_connection.block_inject_weight.weight").contents &&
                        t5_attention_read_buffers.hc_up.contents ==
                            provisional_runtime->dense_weight(
                                "attn_hyper_connection.input_mix_weight_up.weight").contents &&
                        same_view(t5_attention_read_buffers.mixed,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::mixed)) &&
                        same_view(t5_kv_projection_buffers.mixed,
                                  t5_attention_read_buffers.mixed) &&
                        same_view(t5_kv_projection_buffers.xq,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::xq)) &&
                        t5_kv_projection_buffers.key_weights.contents() ==
                            provisional_runtime->dense_weight(
                                "self_attn.k_proj.weight").contents &&
                        t5_kv_projection_buffers.value_weights.contents() ==
                            provisional_runtime->dense_weight(
                                "self_attn.v_proj.weight").contents &&
                        same_view(t5_kv_projection_buffers.kcur,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::kcur)) &&
                        same_view(t5_kv_projection_buffers.vcur,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::vcur)) &&
                        same_view(t5_kv_projection_buffers.sink,
                                  attention_bo),
                    "MTP stages 6–8 resolve the planned weights and T=5 scratch views");
            auto t5_projection_command =
                rollback_context->make_command_buffer(
                    "MTP runtime-bound T=5 stages 1–8", error);
            require(t5_projection_command != nullptr &&
                        Encoder::encode(
                            *rollback_context, *t5_projection_command,
                            "MTP runtime-bound T=5 input-to-KV chain",
                            [&](Encoder& encoder,
                                std::string& encode_error) {
                                return encode_mtp_input_projection(
                                           *rollback_context, encoder,
                                           projection_buffers, encode_error) &&
                                       encode_mtp_add_streams(
                                           *rollback_context, encoder,
                                           t5_add_streams_buffers,
                                           encode_error) &&
                                       encode_mtp_attention_read(
                                           *rollback_context, encoder,
                                           t5_attention_read_buffers,
                                           encode_error) &&
                                       encode_mtp_attention_kv_projection(
                                           *rollback_context, encoder,
                                           t5_kv_projection_buffers,
                                           encode_error);
                            },
                            error) &&
                        t5_projection_command->commit(error) &&
                        t5_projection_command->wait_until_completed(error),
                    "dispatch MTP T=5 stages 1–8 with their planned scratch views: " +
                        error);

            projection_params = {};
            projection_params.T = 8u;
            projection_params.hid[0] = 8u;
            projection_params.hid[1] = 8u;
            projection_params.hid[2] = 8u;
            projection_params.hid[3] = 8u;
            std::memcpy(provisional_params_slot.contents, &projection_params,
                        sizeof(projection_params));
            require(provisional_runtime->bind_input_projection(
                        *rollback_context, projection_sources,
                        projection_buffers, error),
                    "bind T=8 MTP stage-1–5 runtime views: " + error);
            const BufferRef projection_bo =
                provisional_runtime->bind_region(MtpBindRegionId::bo);
            require(projection_buffers.matvec_sink.handle ==
                            projection_bo.handle &&
                        projection_buffers.matvec_sink.offset ==
                            projection_bo.offset &&
                        projection_buffers.matvec_sink.length ==
                            sizeof(float),
                    "T=8 MTP projection uses the later-dead bo word as its required sink");
            MtpAddStreamsBuffers t8_add_streams_buffers;
            MtpAttentionReadBuffers t8_attention_read_buffers;
            MtpAttentionKvProjectionBuffers t8_kv_projection_buffers;
            require(provisional_runtime->bind_add_streams(
                        *rollback_context, 9u, t8_add_streams_buffers,
                        error) &&
                        provisional_runtime->bind_attention_read(
                            *rollback_context, 9u,
                            t8_attention_read_buffers, error) &&
                        provisional_runtime->bind_attention_kv_projection(
                            *rollback_context, 9u,
                            t8_kv_projection_buffers, error) &&
                        same_view(
                            t8_add_streams_buffers.output,
                            provisional_runtime->bind_region(
                                MtpBindRegionId::mtp_rin)) &&
                        same_view(t8_attention_read_buffers.dead_h2_scratch,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::h2)) &&
                        same_view(t8_attention_read_buffers.mixed,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::mixed)) &&
                        same_view(t8_kv_projection_buffers.kcur,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::kcur)) &&
                        same_view(t8_kv_projection_buffers.vcur,
                                  provisional_runtime->bind_region(
                                      MtpBindRegionId::vcur)) &&
                        same_view(t8_kv_projection_buffers.sink,
                                  projection_bo),
                    "T=8 MTP stage-6–8 bindings use full planned active regions: " +
                        error);
            auto t8_projection_command =
                rollback_context->make_command_buffer(
                    "MTP runtime-bound T=8 stages 1–8", error);
            require(t8_projection_command != nullptr &&
                        Encoder::encode(
                            *rollback_context, *t8_projection_command,
                            "MTP runtime-bound T=8 input-to-KV chain",
                            [&](Encoder& encoder,
                                std::string& encode_error) {
                                return encode_mtp_input_projection(
                                           *rollback_context, encoder,
                                           projection_buffers, encode_error) &&
                                       encode_mtp_add_streams(
                                           *rollback_context, encoder,
                                           t8_add_streams_buffers,
                                           encode_error) &&
                                       encode_mtp_attention_read(
                                           *rollback_context, encoder,
                                           t8_attention_read_buffers,
                                           encode_error) &&
                                       encode_mtp_attention_kv_projection(
                                           *rollback_context, encoder,
                                           t8_kv_projection_buffers,
                                           encode_error);
                            },
                            error) &&
                        t8_projection_command->commit(error) &&
                        t8_projection_command->wait_until_completed(error),
                    "dispatch MTP stages 1–8 at the maximum T=8 active extent: " +
                        error);

            projection_params = {};
            projection_params.T = 1u;
            projection_params.hid[0] = 4u;
            std::memcpy(provisional_params_slot.contents, &projection_params,
                        sizeof(projection_params));
            projection_sources.token_source =
                MtpInputTokenSource::selected_token;
            const BufferRef selected_token_view =
                provisional_runtime->bind_region(MtpBindRegionId::out_ids)
                    .slice(0u, sizeof(std::uint32_t));
            projection_sources.selected_token =
                rollback_context->buffers().read_only(selected_token_view);
            const std::uint32_t stale_selected_token = 0xffffffffu;
            std::memcpy(selected_token_view.contents, &stale_selected_token,
                        sizeof(stale_selected_token));
            require(provisional_runtime->bind_input_projection(
                        *rollback_context, projection_sources,
                        projection_buffers, error) &&
                        projection_buffers.selected_token.contents() ==
                            selected_token_view.contents,
                    "MTP runtime binding preserves the separate GPU-produced T=1 token");
            MtpAddStreamsBuffers add_streams_buffers;
            require(provisional_runtime->bind_add_streams(
                        *rollback_context, 9u, add_streams_buffers, error) &&
                        add_streams_buffers.params.contents() ==
                            provisional_params_slot.contents &&
                        add_streams_buffers.h2.contents() ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::h2).contents &&
                        add_streams_buffers.e2.contents() ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::e2).contents &&
                        add_streams_buffers.output.handle ==
                            provisional_runtime->bind_region(
                                MtpBindRegionId::mtp_rin).handle,
                    "MTP runtime binding resolves stage-6 input and residual views");
            MtpAttentionReadBuffers attention_read_buffers;
            MtpAttentionKvProjectionBuffers kv_projection_buffers;
            require(provisional_runtime->bind_attention_read(
                        *rollback_context, 9u, attention_read_buffers,
                        error) &&
                        provisional_runtime->bind_attention_kv_projection(
                            *rollback_context, 9u, kv_projection_buffers,
                            error),
                    "bind MTP runtime views for stages 7–8: " + error);
            const ReadOnlyBufferRef projection_token_source =
                rollback_context->buffers().read_only(
                    provisional_params_slot.slice(
                        offsetof(MtpParams, tokens),
                        sizeof(std::uint32_t)));
            const std::array projection_token_output{
                EncoderBufferBinding{selected_token_view, 1u,
                                     sizeof(std::uint32_t),
                                     alignof(std::uint32_t)},
            };
            const std::array projection_token_input{
                EncoderReadOnlyBufferBinding{
                    projection_token_source, 0u, sizeof(std::uint32_t),
                    alignof(std::uint32_t)},
            };
            constexpr std::array<EncoderBytesBinding, 0> no_bytes{};
            constexpr std::array<EncoderThreadgroupMemory, 0> no_memory{};
            auto projection_command = rollback_context->make_command_buffer(
                "MTP runtime-bound input projection smoke", error);
            require(projection_command != nullptr &&
                        Encoder::encode(
                            *rollback_context, *projection_command,
                            "MTP runtime-bound GPU token producer and input projection",
                            [&](Encoder& encoder,
                                std::string& encode_error) {
                                return encoder.dispatch(
                                           "strata_selftest_copy_u32_one",
                                           {1u, 1u, 1u}, {1u, 1u, 1u},
                                           projection_token_output,
                                           projection_token_input, no_bytes,
                                           no_memory, encode_error) &&
                                       encode_mtp_input_projection(
                                           *rollback_context, encoder,
                                           projection_buffers, encode_error) &&
                                       encode_mtp_add_streams(
                                           *rollback_context, encoder,
                                           add_streams_buffers, encode_error) &&
                                       encode_mtp_attention_read(
                                           *rollback_context, encoder,
                                           attention_read_buffers,
                                           encode_error) &&
                                       encode_mtp_attention_kv_projection(
                                           *rollback_context, encoder,
                                           kv_projection_buffers,
                                           encode_error);
                            },
                            error) &&
                        projection_command->commit(error) &&
                        projection_command->wait_until_completed(error),
                    "dispatch MTP stages 1–8 after a GPU-produced selected token: " +
                        error);
            MtpInputProjectionSources invalid_projection_sources =
                projection_sources;
            invalid_projection_sources.params_slot = 10u;
            require(!provisional_runtime->bind_input_projection(
                        *rollback_context, invalid_projection_sources,
                        projection_buffers, error) &&
                        !error.empty() && !projection_buffers.params &&
                        !projection_buffers.token_embedding_table &&
                        !projection_buffers.fc_embedding_weights &&
                        !projection_buffers.embedding &&
                        !projection_buffers.matvec_sink,
                    "failed MTP projection binding clears every returned view");

            const BufferRef bind_base = provisional_runtime->bind_region(
                MtpBindRegionId::mtp_rin);
            require(bind_base &&
                        bind_base.length ==
                            bind_layout.region(MtpBindRegionId::mtp_rin).bytes,
                    "MTP runtime did not expose its first bounded bind region");
            for (std::size_t index = 0; index < kMtpBindRegionCount; ++index) {
                const auto id = static_cast<MtpBindRegionId>(index);
                const MtpBindRegion region = bind_layout.region(id);
                const BufferRef view = provisional_runtime->bind_region(id);
                if (region.bytes == 0) {
                    require(!view,
                            std::string(mtp_bind_region_name(id)) +
                                " unexpectedly has a device view");
                    continue;
                }
                require(view && rollback_context->buffers().owns(view) &&
                            view.handle == bind_base.handle &&
                            view.length == region.bytes &&
                            view.offset ==
                                bind_base.offset + region.offset &&
                            view.contents ==
                                static_cast<std::uint8_t*>(bind_base.contents) +
                                    region.offset &&
                            (bind_base.gpu_address == 0
                                 ? view.gpu_address == 0
                                 : view.gpu_address ==
                                       bind_base.gpu_address + region.offset),
                        std::string(mtp_bind_region_name(id)) +
                            " does not resolve to its bounded bind-arena view");
            }
            const BufferRef device_vocab_ids = provisional_runtime->bind_region(
                MtpBindRegionId::draft_vocab_ids);
            const auto& inspected_vocab_ids =
                provisional_runtime->draft_vocab_ids();
            require(device_vocab_ids &&
                        device_vocab_ids.length ==
                            inspected_vocab_ids.size() *
                                sizeof(std::uint32_t) &&
                        std::memcmp(device_vocab_ids.contents,
                                    inspected_vocab_ids.data(),
                                    device_vocab_ids.length) == 0,
                    "MTP device vocabulary map was not initialized from inspection");
            const BufferRef window_ids = provisional_runtime->bind_region(
                MtpBindRegionId::mtp_ids);
            require(window_ids &&
                        window_ids.length ==
                            static_cast<std::uint64_t>(
                                provisional_runtime->bind_layout().capacity) *
                                sizeof(std::int32_t),
                    "MTP window ID view does not match the planned capacity");
            const auto* initial_window_ids =
                static_cast<const std::int32_t*>(window_ids.contents);
            for (std::uint32_t index = 0;
                 index < provisional_runtime->bind_layout().capacity; ++index) {
                require(initial_window_ids[index] ==
                            static_cast<std::int32_t>(index),
                        "MTP window ID view was not initialized to identity");
            }
            provisional_bind_view = window_ids;
            provisional_expert = provisional_runtime->expert(0);
            provisional_dense =
                provisional_runtime->dense_weight("fc_embedding.weight");
            require(provisional_expert && provisional_dense &&
                        rollback_context->buffers().owns(provisional_expert) &&
                        rollback_context->buffers().owns(provisional_dense),
                    "MTP runtime owner did not expose its mapped weights");
            require(provisional_runtime->kv_views(provisional_kv, error) &&
                        provisional_kv.k && provisional_kv.v &&
                        rollback_context->buffers().owns(provisional_kv.k) &&
                        rollback_context->buffers().owns(provisional_kv.v),
                    "MTP runtime owner did not expose its planned K/V views: " + error);
            require(rollback_context->buffers().allocation_count() >
                        allocations_before_runtime &&
                        rollback_context->buffers().allocated_size() >
                            bytes_before_runtime,
                    "MTP runtime transaction did not register all planned resources");
        }
        require(rollback_context->buffers().allocation_count() ==
                        allocations_with_main_kv &&
                    rollback_context->buffers().allocated_size() ==
                        bytes_with_main_kv,
                "MTP rollback did not remove its allocations while preserving main KV");
        require(provisional_runtime != nullptr &&
                    !provisional_runtime->is_valid() &&
                    !provisional_runtime->shared_rope() &&
                    provisional_main_kv->allocation() &&
                    rollback_context->buffers().owns(provisional_main_rope) &&
                    rollback_context->buffers().owns(provisional_shared_rope) &&
                    !provisional_runtime->expert(0) &&
                    !provisional_runtime->dense_weight("fc_embedding.weight") &&
                    !provisional_runtime->bind_region(MtpBindRegionId::mtp_ids) &&
                    !provisional_runtime->mtp_params_slot(0),
                "MTP rollback invalidated the main KV or exposed stale runtime views");
        require(!provisional_expert && !provisional_dense && !provisional_bind_view &&
                    !provisional_params_slot &&
                    !provisional_kv.k && !provisional_kv.v &&
                    !rollback_context->buffers().owns(provisional_expert) &&
                    !rollback_context->buffers().owns(provisional_dense) &&
                    !rollback_context->buffers().owns(provisional_bind_view),
                "a previously returned MTP BufferRef stayed live after rollback");
        require(!provisional_runtime->kv_views(provisional_kv, error) &&
                    !provisional_kv.k && !provisional_kv.k_scale &&
                    !provisional_kv.v && !provisional_kv.v_scale,
                "rolled-back MTP runtime did not invalidate and clear its K/V views");

        PlanOptions no_prefill_options = plan_options;
        no_prefill_options.prefill_mode = PrefillMode::None;
        no_prefill_options.prefill_chunk = 0;
        PlanResult no_prefill_plan;
        require(make_plan(rollback_geometry, no_prefill_options,
                          no_prefill_plan, error) &&
                    no_prefill_plan.prefill_chunk == 0 &&
                    no_prefill_plan.mtp_params_bytes ==
                        9ull * kMtpParamsSlotBytes,
                "no-prefill MTP plan omitted its short-read parameter group");
        std::unique_ptr<MtpRuntimeStorage> no_prefill_runtime;
        BufferRef short_read_params_slot;
        {
            auto outer = rollback_context->buffers().transaction();
            no_prefill_runtime = MtpRuntimeStorage::open(
                *rollback_context, rollback_geometry, rollback_sources,
                no_prefill_plan, error);
            require(no_prefill_runtime != nullptr &&
                        no_prefill_runtime->is_valid() &&
                        !no_prefill_runtime->shared_rope() &&
                        no_prefill_runtime->params_layout().slot_count == 9 &&
                        no_prefill_runtime->params_layout().prefill_group_slots == 1,
                    "standalone no-prefill MTP runtime did not open unbound short-read storage: " +
                        error);
            short_read_params_slot =
                no_prefill_runtime->mtp_params_slot(8);
            require(short_read_params_slot &&
                        short_read_params_slot.length == kMtpParamsSlotBytes &&
                        !no_prefill_runtime->mtp_params_slot(9),
                    "short-read catch-up slot is not bounded to the no-prefill plan");
        }
        require(no_prefill_runtime != nullptr &&
                    !no_prefill_runtime->is_valid() &&
                    !no_prefill_runtime->mtp_params_slot(8) &&
                    !short_read_params_slot,
                "no-prefill parameter slot survived its outer transaction rollback");

        rollback_sources = {};
        rollback_window_sources = {};
        require(exclusive_lock_is_available(generation_lock_path),
                "invalid MTP runtime owner retained its generation lock");
        provisional_runtime.reset();
        provisional_main_kv.reset();
        rollback_context.reset();
        require(exclusive_lock_is_available(generation_lock_path),
                "rolling back the MTP runtime retained its generation lock");

        Geometry owner_geometry;
        WindowSources owner_window_sources;
        MtpRuntimeSources owner_mtp_sources;
        require(WeightStore::inspect(options, owner_geometry, error,
                                     &owner_window_sources, &owner_mtp_sources),
                "inspect model and MTP sources for aggregate-owner coverage: " + error);
        ContextOptions owner_context_options;
        auto owner_context = Context::open(owner_context_options, error);
        require(owner_context != nullptr,
                "open aggregate-owner Metal Context: " + error);
        PlanOptions owner_plan_options = plan_options;
        owner_plan_options.working_set_bytes =
            owner_context->device_facts().working_set_bytes;
        owner_plan_options.ram_bytes =
            owner_context->device_facts().physical_memory_bytes;
        owner_plan_options.available_bytes = owner_plan_options.ram_bytes;
        owner_plan_options.available_is_override = true;
        PlanResult owner_plan;
        require(make_plan(owner_geometry, owner_plan_options, owner_plan, error) &&
                    owner_plan.fits && owner_plan.mtp_bytes != 0 &&
                    owner_plan.session_state_bytes == 118'038'528u &&
                    owner_plan.window_scratch_bytes != 0,
                "make a live-device plan for aggregate MTP storage: " + error);

        const std::size_t allocations_before_owner =
            owner_context->buffers().allocation_count();
        const std::uint64_t bytes_before_owner =
            owner_context->buffers().allocated_size();
        ContextOptions scratch_limited_options;
        scratch_limited_options.pipelines = false;
        scratch_limited_options.max_buffer_length_cap =
            owner_plan.window_scratch_bytes - 1;
        auto scratch_limited_context =
            Context::open(scratch_limited_options, error);
        require(scratch_limited_context != nullptr &&
                    scratch_limited_context->device_facts().max_buffer_length ==
                        owner_plan.window_scratch_bytes - 1,
                "open Context capped just below the planned Window scratch size: " +
                    error);
        require(ModelStorage::open(*scratch_limited_context, native.string(),
                                   owner_geometry, owner_window_sources, owner_plan,
                                   error, &owner_mtp_sources) == nullptr &&
                    error.find("Window scratch exceeds") != std::string::npos &&
                    scratch_limited_context->buffers().allocation_count() == 0 &&
                    scratch_limited_context->buffers().allocated_size() == 0,
                "ModelStorage must reject an oversized Window scratch before allocating");
        scratch_limited_context.reset();

        ContextOptions limited_context_options;
        limited_context_options.pipelines = false;
        limited_context_options.max_buffer_length_cap = owner_plan.kv_bytes - 1;
        auto limited_context = Context::open(limited_context_options, error);
        require(limited_context != nullptr &&
                    limited_context->device_facts().max_buffer_length ==
                        owner_plan.kv_bytes - 1,
                "open a Context with a one-byte-smaller buffer cap: " + error);
        require(ModelStorage::open(*limited_context, native.string(), owner_geometry,
                                   owner_window_sources, owner_plan, error,
                                   &owner_mtp_sources) == nullptr &&
                    error == "main QSA KV state exceeds the Metal device's maxBufferLength" &&
                    limited_context->buffers().allocation_count() == 0 &&
                    limited_context->buffers().allocated_size() == 0,
                "ModelStorage must reject a valid speculative plan at its live maxBufferLength preflight");
        limited_context.reset();

        QsaKvLayout bounded_kv_layout;
        require(validate_main_qsa_kv_plan(owner_plan, owner_plan.kv_bytes,
                                          bounded_kv_layout, error) &&
                    bounded_kv_layout.allocation_bytes == owner_plan.kv_bytes,
                "valid speculative main KV plan must fit its exact device-buffer limit: " +
                    error);
        require(!validate_main_qsa_kv_plan(owner_plan, owner_plan.kv_bytes - 1,
                                           bounded_kv_layout, error) &&
                    error.find("maxBufferLength") != std::string::npos &&
                    bounded_kv_layout.allocation_bytes == 0 &&
                    owner_context->buffers().allocation_count() ==
                        allocations_before_owner &&
                    owner_context->buffers().allocated_size() == bytes_before_owner,
                "speculative main KV plan must fail a smaller single-buffer limit without allocation");

        PlanResult invalid_main_kv_plan = owner_plan;
        ++invalid_main_kv_plan.kv_bytes;
        ++invalid_main_kv_plan.total_bytes;
        require(ModelStorage::open(*owner_context, native.string(), owner_geometry,
                                   owner_window_sources, invalid_main_kv_plan, error,
                                   &owner_mtp_sources) == nullptr &&
                    error.find("main QSA KV bytes do not match the memory plan") !=
                        std::string::npos &&
                    owner_context->buffers().allocation_count() ==
                        allocations_before_owner &&
                    owner_context->buffers().allocated_size() == bytes_before_owner,
                "aggregate owner must reject forged main KV bytes before allocating");

        PlanResult invalid_owner_plan = owner_plan;
        ++invalid_owner_plan.mtp_params_bytes;
        require(ModelStorage::open(*owner_context, native.string(), owner_geometry,
                                   owner_window_sources, invalid_owner_plan, error,
                                   &owner_mtp_sources) == nullptr &&
                    error.find("MTP parameter bytes do not match") !=
                        std::string::npos &&
                    owner_context->buffers().allocation_count() ==
                        allocations_before_owner,
                "aggregate owner must validate MTP layout bytes before allocating");
        PlanResult invalid_scratch_plan = owner_plan;
        ++invalid_scratch_plan.window_scratch_bytes;
        require(ModelStorage::open(*owner_context, native.string(), owner_geometry,
                                   owner_window_sources, invalid_scratch_plan, error,
                                   &owner_mtp_sources) == nullptr &&
                    error.find("exact session and Window scratch bytes") !=
                        std::string::npos &&
                    owner_context->buffers().allocation_count() ==
                        allocations_before_owner,
                "aggregate owner must reject forged fixed-state subtotals before allocating");

        PlanResult underreported_fixed_plan = owner_plan;
        const std::uint64_t exact_fixed_subtotal =
            owner_plan.session_state_bytes + owner_plan.window_scratch_bytes +
            static_cast<std::uint64_t>(kTieredMaxLayers) * owner_geometry.n_expert *
                sizeof(float);
        underreported_fixed_plan.fixed_bytes = exact_fixed_subtotal - 1;
        underreported_fixed_plan.total_bytes -=
            owner_plan.fixed_bytes - underreported_fixed_plan.fixed_bytes;
        require(ModelStorage::open(*owner_context, native.string(), owner_geometry,
                                   owner_window_sources, underreported_fixed_plan, error,
                                   &owner_mtp_sources) == nullptr &&
                    error.find("exact session, Window scratch and MoE usage bytes exceed") !=
                        std::string::npos &&
                    owner_context->buffers().allocation_count() ==
                        allocations_before_owner,
                "aggregate owner must reject a fixed allowance smaller than its exact state subtotal");
        require(ModelStorage::open(*owner_context, native.string(), owner_geometry,
                                   owner_window_sources, owner_plan, error) == nullptr &&
                    error.find("inspected MTP runtime sources") != std::string::npos &&
                    owner_context->buffers().allocation_count() ==
                        allocations_before_owner,
                "aggregate owner must require inspected MTP sources before allocating");

        WindowSources invalid_model_sources;
        require(ModelStorage::open(*owner_context, native.string(), owner_geometry,
                                   invalid_model_sources, owner_plan, error,
                                   &owner_mtp_sources) == nullptr &&
                    owner_context->buffers().allocation_count() ==
                        allocations_before_owner &&
                    owner_context->buffers().allocated_size() == bytes_before_owner,
                "aggregate owner must roll back MTP allocations when model storage fails");
        const fs::path owner_generation_lock =
            owner_mtp_sources.generation_lock.path;
        owner_mtp_sources = {};
        owner_window_sources = {};
        require(exclusive_lock_is_available(owner_generation_lock.string()),
                "aggregate model-storage failure retained the MTP generation lock");

        require(WeightStore::inspect(options, owner_geometry, error,
                                     &owner_window_sources, &owner_mtp_sources),
                "reinspect model and MTP sources for aggregate-owner success: " + error);
        auto aggregate_storage = ModelStorage::open(
            *owner_context, native.string(), owner_geometry, owner_window_sources,
            owner_plan, error, &owner_mtp_sources);
        const BufferRef aggregate_main_rope =
            aggregate_storage != nullptr && aggregate_storage->kv_store() != nullptr
                ? aggregate_storage->kv_store()->rope()
                : BufferRef{};
        const BufferRef aggregate_mtp_rope =
            aggregate_storage != nullptr &&
                    aggregate_storage->mtp_runtime() != nullptr
                ? aggregate_storage->mtp_runtime()->shared_rope()
                : BufferRef{};
        require(aggregate_storage != nullptr &&
                    aggregate_storage->kv_store() != nullptr &&
                    aggregate_storage->kv_store()->allocation().length ==
                        owner_plan.kv_bytes &&
                    aggregate_main_rope && aggregate_mtp_rope &&
                    aggregate_mtp_rope.handle == aggregate_main_rope.handle &&
                    aggregate_mtp_rope.contents == aggregate_main_rope.contents &&
                    aggregate_mtp_rope.gpu_address ==
                        aggregate_main_rope.gpu_address &&
                    aggregate_mtp_rope.length == aggregate_main_rope.length &&
                    aggregate_mtp_rope.offset == aggregate_main_rope.offset &&
                    owner_context->buffers().allocation_identity(
                        aggregate_mtp_rope) != 0 &&
                    owner_context->buffers().allocation_identity(
                        aggregate_mtp_rope) ==
                        owner_context->buffers().allocation_identity(
                            aggregate_main_rope) &&
                    aggregate_storage->mtp_runtime() != nullptr &&
                    aggregate_storage->mtp_runtime()->is_valid() &&
                    aggregate_storage->mtp_runtime()->layout().mapped_file_bytes ==
                        owner_plan.mtp_bytes &&
                    aggregate_storage->window_scratch() != nullptr &&
                    aggregate_storage->window_scratch()->is_valid() &&
                    aggregate_storage->window_scratch()->layout().allocation_bytes ==
                        owner_plan.window_scratch_bytes &&
                    aggregate_storage->window_scratch()->layout().cell_capacity ==
                        aggregate_storage->kv_store()->layout().cell_capacity,
                "aggregate owner did not publish valid main KV, model, MTP and Window scratch storage: " +
                    error);
        BufferRef aggregate_residual;
        require(aggregate_storage->window_scratch()->view(
                    aggregate_storage->window_scratch()->layout().residual,
                    aggregate_residual, error) &&
                    owner_context->buffers().owns(aggregate_residual) &&
                    aggregate_residual.length ==
                        aggregate_storage->window_scratch()->layout().residual.bytes,
                "ModelStorage did not retain an exact bounded Window scratch view: " +
                    error);
        aggregate_storage.reset();
        owner_context.reset();
        owner_mtp_sources = {};
        owner_window_sources = {};
        require(exclusive_lock_is_available(owner_generation_lock.string()),
                "aggregate owner success retained its MTP generation lock after Context close");

        std::cout << "metal_mtp_storage_test: OK\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "metal_mtp_storage_test: " << exception.what() << '\n';
        return 1;
    }
}
