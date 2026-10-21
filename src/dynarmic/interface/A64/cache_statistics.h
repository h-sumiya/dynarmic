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
    std::uint64_t optimized_blocks = 0;
    std::uint64_t hot_samples = 0;
    std::uint64_t baseline_nanoseconds = 0;
    std::uint64_t hot_optimization_nanoseconds = 0;
    std::uint64_t emitted_bytes = 0;
    std::uint64_t capacity_clears = 0;
    std::uint64_t full_clears = 0;  // Includes capacity_clears.
    std::uint64_t full_cache_discarded_blocks = 0;
    std::uint64_t range_invalidation_batches = 0;
    std::uint64_t range_invalidated_blocks = 0;
    std::uint64_t fastmem_invalidated_blocks = 0;
    std::uint64_t frontend_nanoseconds = 0;
    std::uint64_t optimization_nanoseconds = 0;
    std::uint64_t emission_nanoseconds = 0;
    std::uint64_t live_blocks = 0;
    // Used includes the Jit prelude and invalidated code not reclaimed until a
    // full clear. It measures the code allocator cursor, not resident RAM.
    std::uint64_t cache_used_bytes = 0;
    std::uint64_t cache_capacity_bytes = 0;
    std::uint64_t cache_high_water_bytes = 0;
};

}  // namespace Dynarmic::A64
