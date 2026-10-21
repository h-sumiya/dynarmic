// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Dynarmic::Backend::X64 {

struct BaselineReclamation {
    std::size_t reclaimed_blocks = 0;
    std::size_t reclaimed_bytes = 0;
    std::size_t reclaimed_segments = 0;
    // Only currently indexed descriptors are returned. An old allocation may
    // already have been invalidated and its location translated elsewhere.
    std::vector<std::uint64_t> evicted_locations;
};

struct BaselineCacheUsage {
    std::size_t capacity_bytes = 0;
    // Includes invalidated bodies until their individual segment is reused.
    std::size_t used_bytes = 0;
    std::size_t high_water_bytes = 0;
    std::size_t segment_count = 0;
    std::size_t segment_size = 0;
};

/// Metadata for a thread-private, bounded ring of native-code segments.
///
/// This class owns no executable memory and uses no synchronization. Its owner
/// calls Prepare only from the immutable dispatcher, with no native return
/// address into a translated baseline body. The reclaimer must disconnect all
/// references to the selected segment before Prepare returns its address.
/// Segments are reused independently; capacity pressure never clears the cache.
class A64BaselineCodeCache {
public:
    struct Allocation {
        std::uint64_t location;
        std::uintptr_t entrypoint;
    };

    struct Segment {
        std::uintptr_t begin = 0;
        std::size_t capacity = 0;
        std::size_t used = 0;
        std::vector<Allocation> allocations;
        // Flat storage avoids one heap allocation per translated block.
        std::vector<std::uint64_t> fault_rips;
    };

    bool Initialized() const { return !segments.empty(); }

    void Initialize(std::uintptr_t begin, std::size_t capacity, std::size_t requested_segment_size,
                    std::size_t minimum_segment_size = alignment) {
        if (Initialized())
            throw std::logic_error("baseline code cache is already initialized");
        if (begin > std::numeric_limits<std::uintptr_t>::max() - (alignment - 1))
            throw std::invalid_argument("baseline code cache address overflows");
        const auto aligned_begin = AlignUp(begin);
        const auto skipped = aligned_begin - begin;
        if (capacity < skipped)
            throw std::invalid_argument("baseline code cache has no aligned space");
        capacity = AlignDown(capacity - skipped);
        if (aligned_begin > std::numeric_limits<std::uintptr_t>::max() - capacity)
            throw std::invalid_argument("baseline code cache range overflows");
        if (minimum_segment_size > std::numeric_limits<std::size_t>::max() - (alignment - 1))
            throw std::invalid_argument("baseline code cache segment size overflows");
        minimum_segment_size = AlignUp(std::max(minimum_segment_size, alignment));
        const auto largest_segment = AlignDown(capacity / 2);
        if (minimum_segment_size > largest_segment)
            throw std::invalid_argument("baseline code cache requires at least two complete segments");
        const auto segment_size = std::clamp(AlignDown(requested_segment_size),
                                             minimum_segment_size, largest_segment);
        const auto count = capacity / segment_size;
        std::vector<Segment> initial(count);
        for (std::size_t index = 0; index < count; ++index) {
            initial[index].begin = aligned_begin + index * segment_size;
            initial[index].capacity = index + 1 == count ? capacity - index * segment_size : segment_size;
        }
        segments = std::move(initial);
        usage = {capacity, 0, 0, count, segment_size};
    }

    /// Reserve a conservative upper bound. Record supplies the actual end.
    /// Reclaim is called at most once, and only for a previously used segment.
    /// If it throws, allocator metadata still names the old segment.
    template<typename Reclaim>
    std::uintptr_t Prepare(std::size_t required_bytes, Reclaim&& reclaim) {
        if (!Initialized())
            throw std::logic_error("baseline code cache is not initialized");
        if (required_bytes == 0)
            throw std::invalid_argument("baseline block must not be empty");
        if (required_bytes > usage.segment_size)
            throw std::length_error("baseline block exceeds the configured segment size");
        auto& active = segments[current_segment];
        if (required_bytes <= active.capacity - active.used)
            return active.begin + active.used;

        const auto next_index = (current_segment + 1) % segments.size();
        auto& victim = segments[next_index];
        if (victim.used != 0)
            std::forward<Reclaim>(reclaim)(victim);
        usage.used_bytes -= victim.used;
        victim.used = 0;
        victim.allocations.clear();
        victim.fault_rips.clear();
        current_segment = next_index;
        return victim.begin;
    }

    /// Record the exact allocation and fault addresses after emission. This is
    /// deliberately independent of the descriptor index: invalidation can drop
    /// a descriptor long before the corresponding native bytes are reclaimed.
    void Record(std::uint64_t location, std::uintptr_t entrypoint, std::uintptr_t end,
                std::span<const std::uint64_t> fault_rips) {
        RecordWithFaults(location, entrypoint, end, fault_rips.size(),
                        [&](std::size_t index) { return fault_rips[index]; });
    }

    // Map fault offsets directly into the segment's flat vector. Emission need
    // not allocate a temporary vector for each block with fast memory accesses.
    template<typename FaultAt>
    void RecordWithFaults(std::uint64_t location, std::uintptr_t entrypoint, std::uintptr_t end,
                          std::size_t fault_count, FaultAt&& fault_at) {
        if (!Initialized())
            throw std::logic_error("baseline code cache is not initialized");
        auto& active = segments[current_segment];
        const auto previous_end = active.begin + active.used;
        if (entrypoint < previous_end || end <= entrypoint || end > active.begin + active.capacity)
            throw std::out_of_range("baseline emission crossed its reserved segment");
        // Reserve both vectors before changing logical state. Capacity remains
        // reusable when a segment wraps, avoiding steady-state heap churn.
        ReserveAdditional(active.allocations, 1);
        ReserveAdditional(active.fault_rips, fault_count);
        active.allocations.push_back({location, entrypoint});
        for (std::size_t index = 0; index < fault_count; ++index)
            active.fault_rips.push_back(fault_at(index));
        const auto new_used = static_cast<std::size_t>(end - active.begin);
        usage.used_bytes += new_used - active.used;
        active.used = new_used;
        usage.high_water_bytes = std::max(usage.high_water_bytes, usage.used_bytes);
    }

    /// For an explicit caller-requested full invalidation only. This is never
    /// called by Prepare or used as a capacity-management policy.
    void Reset() {
        for (auto& segment : segments) {
            segment.used = 0;
            segment.allocations.clear();
            segment.fault_rips.clear();
        }
        usage.used_bytes = 0;
        current_segment = 0;
    }

    BaselineCacheUsage Usage() const { return usage; }

private:
    static constexpr std::size_t alignment = 16;
    static constexpr std::size_t AlignDown(std::size_t value) { return value & ~(alignment - 1); }
    static constexpr std::size_t AlignUp(std::size_t value) { return (value + alignment - 1) & ~(alignment - 1); }

    template<typename T>
    static void ReserveAdditional(std::vector<T>& values, std::size_t count) {
        if (count > values.max_size() - values.size())
            throw std::length_error("baseline segment metadata exceeds vector capacity");
        const auto required = values.size() + count;
        if (required > values.capacity()) {
            const auto grown = values.capacity() > values.max_size() / 2 ? values.max_size() : values.capacity() * 2;
            values.reserve(std::max(required, std::max(std::size_t{64}, grown)));
        }
    }

    std::vector<Segment> segments;
    std::size_t current_segment = 0;
    BaselineCacheUsage usage;
};

}  // namespace Dynarmic::Backend::X64
