/* apk2exe Dynarmic cache diagnostics.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstdint>

namespace Dynarmic::A64 {

// Counters are per Jit and cumulative for its lifetime. ClearCache does not
// reset them. Obtain a snapshot on the owning thread, including from a callback;
// concurrent access while another thread runs the Jit is not supported.
struct CacheStatistics {
    bool available = false;
    std::uint64_t compiled_blocks = 0;
    std::uint64_t hot_samples = 0;
    std::uint64_t baseline_nanoseconds = 0;
    std::uint64_t emitted_bytes = 0;
    std::uint64_t full_clears = 0;    // Explicit full invalidations only.
    std::uint64_t full_cache_discarded_blocks = 0;
    std::uint64_t range_invalidation_batches = 0;
    std::uint64_t range_invalidated_blocks = 0;
    std::uint64_t fastmem_invalidated_blocks = 0;
    std::uint64_t frontend_nanoseconds = 0;
    std::uint64_t optimization_nanoseconds = 0;
    std::uint64_t emission_nanoseconds = 0;
    std::uint64_t live_blocks = 0;
    // Private bytes occupied by the permanent prelude and live segment spans,
    // including invalidated code awaiting segment reuse. This is allocator
    // occupancy, not resident RAM; shared regions have separate statistics.
    std::uint64_t cache_used_bytes = 0;
    std::uint64_t cache_capacity_bytes = 0;
    std::uint64_t cache_high_water_bytes = 0;
    std::uint64_t baseline_reclaimed_blocks = 0;
    std::uint64_t baseline_reclaimed_bytes = 0;
    std::uint64_t baseline_reclaimed_segments = 0;
    std::uint64_t baseline_reclamation_nanoseconds = 0;
    std::uint64_t baseline_reclamation_max_nanoseconds = 0;
    // Slow-path observations only. Native shared-index hits intentionally do
    // not increment a contended shared counter.
    std::uint64_t shared_cache_hits = 0;
    std::uint64_t shared_cache_requests = 0;
};

}  // namespace Dynarmic::A64
