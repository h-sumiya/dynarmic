/* Process-local, asynchronous A64 optimized-code cache.
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Dynarmic::A64 {

struct SharedCodeCacheConfig {
    // Includes constructing, live and retired executable regions. A stalled
    // reader delays compilation rather than allowing this bound to grow.
    std::size_t capacity_bytes = 256u * 1024u * 1024u;
    std::size_t region_size = 8u * 1024u * 1024u;
    std::size_t max_pending_jobs = 1024;
    std::size_t worker_count = 1;
    // Power-of-two, fixed-size read-only-on-hit native dispatch index.
    std::size_t max_blocks = 524288;
    std::filesystem::path persistent_directory;
    std::uint64_t persistent_capacity_bytes = 512u * 1024u * 1024u;
};

struct SharedCodeCacheStatistics {
    std::uint64_t queued_jobs = 0, deduplicated_jobs = 0, dropped_jobs = 0;
    std::uint64_t compiled_blocks = 0, failed_jobs = 0, discarded_jobs = 0;
    std::uint64_t compilation_nanoseconds = 0;
    std::uint64_t live_blocks = 0, pending_jobs = 0;
    // Charged capacity includes construction reservations and remains charged
    // until worker-side physical destruction has finished.
    std::uint64_t allocated_bytes = 0, retired_bytes = 0;
    std::uint64_t reclaimed_bytes = 0, reclaimed_regions = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t queue_wait_nanoseconds = 0, queue_wait_max_nanoseconds = 0;
    std::uint64_t pending_high_water = 0;
    std::uint64_t link_cells = 0;
    std::uint64_t retirement_nanoseconds = 0, retirement_max_nanoseconds = 0;
    std::uint64_t reclamation_nanoseconds = 0, reclamation_max_nanoseconds = 0;
    std::uint64_t restored_blocks = 0, restored_regions = 0, persisted_regions = 0;
    std::uint64_t persistent_bytes = 0, persistence_failures = 0, restore_nanoseconds = 0;
    std::uint64_t index_files = 0, index_bytes = 0, payload_files = 0, payload_bytes = 0;
};

struct SharedCodeCacheAccess;

// Share this object only among Jits of ONE guest address space. Attaching a
// different code-generation contract throws invalid_argument. Callbacks, TLS
// and processor IDs remain per Jit. Workers consume copied instruction words;
// they never call an execution thread's callbacks or access guest code memory.
// Only optimized native regions are persisted; private baseline code is never
// serialized. Stable image identities and source words bind disk entries to
// the guest code, and relocation records reconnect each new host process.
class SharedCodeCache final {
public:
    explicit SharedCodeCache(SharedCodeCacheConfig config = {});
    ~SharedCodeCache();
    SharedCodeCache(const SharedCodeCache&) = delete;
    SharedCodeCache& operator=(const SharedCodeCache&) = delete;

    // Removes published targets and cancels stale in-flight snapshots before
    // returning. Run/Step maintains reader protection through callbacks and
    // lookups; a lookup tail-entered from shared code may advance its epoch.
    // Every invocation releases the reader on exit. This cache does not halt
    // attached Jits. The owner must stop and synchronize affected Jits before
    // changing executable guest bytes or releasing guest mappings.
    void InvalidateRange(std::uint64_t address, std::size_t length);
    void Clear();
    SharedCodeCacheStatistics GetStatistics() const;
    // The image and reader must remain valid until this cache is destroyed.
    // Registration follows relocation/protection and precedes guest execution.
    void RegisterImage(std::string identity, std::uint64_t address, std::size_t size,
                       std::function<std::optional<std::uint32_t>(std::uint64_t)> read_code, bool restore = true);
    void RestorePersistence();
    void FlushPersistence();

private:
    friend struct SharedCodeCacheAccess;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Dynarmic::A64
