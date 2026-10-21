/* Process-local, asynchronous A64 optimized-code cache.
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace Dynarmic::A64 {

struct SharedCodeCacheConfig {
    // Includes every live AND retired executable region. A stalled reader
    // delays compilation rather than allowing this bound to grow.
    std::size_t capacity_bytes = 256u * 1024u * 1024u;
    std::size_t region_size = 8u * 1024u * 1024u;
    std::size_t max_pending_jobs = 1024;
    std::size_t worker_count = 1;
    // Power-of-two, fixed-size read-only-on-hit native dispatch index.
    std::size_t max_blocks = 524288;
};

struct SharedCodeCacheStatistics {
    std::uint64_t queued_jobs = 0, deduplicated_jobs = 0, dropped_jobs = 0;
    std::uint64_t compiled_blocks = 0, failed_jobs = 0, discarded_jobs = 0;
    std::uint64_t compilation_nanoseconds = 0;
    std::uint64_t live_blocks = 0, pending_jobs = 0;
    std::uint64_t allocated_bytes = 0, retired_bytes = 0;
    std::uint64_t reclaimed_bytes = 0, reclaimed_regions = 0;
    std::uint64_t capacity_bytes = 0;
};

struct SharedCodeCacheAccess;

// Share this object only among Jits of ONE guest address space. Attaching a
// different code-generation contract throws invalid_argument. Callbacks, TLS
// and processor IDs remain per Jit. Workers consume copied instruction words;
// they never call an execution thread's callbacks or access guest code memory.
// No disk cache, serialization, or baseline sharing is involved.
class SharedCodeCache final {
public:
    explicit SharedCodeCache(SharedCodeCacheConfig config = {});
    ~SharedCodeCache();
    SharedCodeCache(const SharedCodeCache&) = delete;
    SharedCodeCache& operator=(const SharedCodeCache&) = delete;

    // Removes published targets and cancels stale in-flight snapshots before
    // returning. Execution already in progress reaches a dispatcher safepoint;
    // native bytes are retained until its last reader has passed that point.
    // Guest mapping lifetime is still the embedding application's contract.
    void InvalidateRange(std::uint64_t address, std::size_t length);
    void Clear();
    SharedCodeCacheStatistics GetStatistics() const;

private:
    friend struct SharedCodeCacheAccess;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Dynarmic::A64
