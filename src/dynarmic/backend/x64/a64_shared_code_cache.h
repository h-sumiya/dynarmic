/* SPDX-License-Identifier: 0BSD */
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/A64/shared_code_cache.h"

namespace Dynarmic::A64 {

struct alignas(64) SharedCacheReader {
    // Zero means outside native code. Written only at Run/Step boundaries and
    // permanent-prelude safepoints, never on a shared-code cache hit.
    std::atomic<std::uint64_t> epoch{0};
};

struct SharedCodeSnapshot {
    std::uint64_t location = 0;
    std::uint64_t revision = 0;
    std::array<std::optional<std::uint32_t>, 64> instructions;
    // Additional immutable words at non-contiguous direct successors. The
    // capture budget is fixed; workers never dereference guest memory.
    std::vector<std::pair<std::uint64_t, std::uint32_t>> successors;
};

// Prefix layout is consumed by GenBaselineReturn. Records are immutable once
// their pointer is release-published to the fixed dispatch index.
struct SharedPublishedBlock;
struct SharedGuestRange {
    std::uint64_t first;
    std::uint64_t last; // inclusive
    const SharedPublishedBlock* owner = nullptr; // finalized before publication
};

struct SharedPublishedBlock {
    static constexpr std::size_t maximum_guest_ranges = 128;
    std::uint64_t location;
    const void* entrypoint;
    std::uint64_t guest_start;
    std::uint64_t guest_end; // inclusive
    // Exact disjoint guest instruction ranges. Empty vectors retain the
    // contiguous guest_start/end compatibility form. Neither the vector nor
    // its addresses may change after it is indexed for publication.
    std::vector<SharedGuestRange> guest_ranges{};
};
static_assert(offsetof(SharedPublishedBlock, entrypoint) == 8);
static_assert(std::atomic<const SharedPublishedBlock*>::is_always_lock_free);
static_assert(sizeof(std::atomic<const SharedPublishedBlock*>) == sizeof(void*));
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t));

inline std::uint64_t SharedCodeHash(std::uint64_t location) {
    // Keep this in sync with the small assembly sequence in the dispatcher.
    return (location >> 2) ^ (location >> 19) ^ (location >> 37);
}

struct SharedCodeCacheAccess {
    static std::unique_ptr<SharedCacheReader> Attach(SharedCodeCache&, const UserConfig&);
    static void Detach(SharedCodeCache&, SharedCacheReader&);
    static std::uint64_t Pin(SharedCodeCache&, SharedCacheReader&);
    static void Unpin(SharedCacheReader&);
    static const void* DispatchTable(const SharedCodeCache&);
    static std::uint64_t DispatchMask(const SharedCodeCache&);
    static std::uint64_t Revision(const SharedCodeCache&);
    static const void* Lookup(const SharedCodeCache&, std::uint64_t location);
    static void Capture(SharedCodeSnapshot&, UserCallbacks&);
    // Nonblocking admission: if the scheduler is busy/full, simply try later.
    static bool Request(SharedCodeCache&, SharedCodeSnapshot);
};

} // namespace Dynarmic::A64
