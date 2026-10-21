/* This file is part of apk2exe's pinned Dynarmic patches.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <dynarmic/common/spin_lock.h>

namespace Dynarmic {

using VAddr = std::uint64_t;
using Vector = std::array<std::uint64_t, 2>;

/// x64 exclusive monitor. A lock protects a hash bucket, not the entire process.
/// Every naturally aligned 1/2/4/8/16-byte access fits in one 16-byte granule.
/// An epoch change invalidates every reservation for that bucket, including ABA
/// stores of the same value. Hash collisions only cause spurious store failure.
/// The expected-value CMPXCHG in the caller still detects non-exclusive writes.
class ExclusiveMonitor {
public:
    static constexpr unsigned GRANULE_SHIFT = 4;
    static constexpr unsigned BUCKET_SHIFT = 12;
    static constexpr std::size_t BUCKET_COUNT = std::size_t{1} << BUCKET_SHIFT;
    static constexpr std::size_t BUCKET_MASK = BUCKET_COUNT - 1;

    struct alignas(64) Bucket {
        SpinLock lock;
        std::uint64_t epoch = 1;
    };

    struct alignas(64) Reservation {
        VAddr address = 0;
        Vector value{};
        // Only the owning processor writes other fields. This field is atomic
        // because the exceptionally rare epoch-wrap path clears all slots.
        std::atomic<std::uint64_t> epoch{0};
        std::uint64_t bytes = 0;
    };

    explicit ExclusiveMonitor(std::size_t processor_count);
    std::size_t GetProcessorCount() const;

    static constexpr std::size_t BucketIndex(VAddr address) {
        const VAddr granule = address >> GRANULE_SHIFT;
        return static_cast<std::size_t>((granule ^ (granule >> BUCKET_SHIFT)) & BUCKET_MASK);
    }

    template<typename T, typename Function>
    T ReadAndMark(std::size_t processor_id, VAddr address, Function op) {
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(sizeof(T) <= 16 && (sizeof(T) & (sizeof(T) - 1)) == 0);
        Bucket& bucket = buckets[BucketIndex(address)];
        Guard guard{bucket};
        Reservation& reservation = reservations[processor_id];
        reservation.epoch.store(0, std::memory_order_relaxed);
        const T value = op();
        // The JIT checks alignment before calling us. Direct API callers with
        // an unaligned address can read but cannot acquire a reservation.
        if ((address & (sizeof(T) - 1)) == 0) {
            reservation.address = address;
            reservation.bytes = sizeof(T);
            std::memcpy(reservation.value.data(), &value, sizeof(T));
            reservation.epoch.store(bucket.epoch, std::memory_order_relaxed);
        }
        return value;
    }

    template<typename T, typename Function>
    bool DoExclusiveOperation(std::size_t processor_id, VAddr address, Function op) {
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(sizeof(T) <= 16 && (sizeof(T) & (sizeof(T) - 1)) == 0);
        Bucket& bucket = buckets[BucketIndex(address)];
        Guard guard{bucket};
        Reservation& reservation = reservations[processor_id];
        const auto expected_epoch = reservation.epoch.load(std::memory_order_relaxed);
        reservation.epoch.store(0, std::memory_order_relaxed);
        if (expected_epoch == 0 || expected_epoch != bucket.epoch ||
            reservation.address != address || reservation.bytes != sizeof(T) ||
            (address & (sizeof(T) - 1)) != 0) {
            return false;
        }
        // Clear reservations before invoking the fault-capable memory callback.
        // A failed CMPXCHG may invalidate extra reservations, which is allowed.
        if (++bucket.epoch == 0) {
            ResetEpochAfterWrap(bucket);
        }
        T expected;
        std::memcpy(&expected, reservation.value.data(), sizeof(T));
        return op(expected);
    }

    void Clear();
    void ClearProcessor(std::size_t processor_id);

    // Internal JIT entry point, called with bucket.lock held. Epoch wrap must
    // not resurrect a reservation from 2^64 operations ago. Other buckets may
    // run concurrently; clearing their atomic epoch can only cause a retry.
    void ResetEpochAfterWrap(Bucket& bucket);

private:
    class Guard {
    public:
        explicit Guard(Bucket& bucket_) : bucket{bucket_} { bucket.lock.Lock(); }
        ~Guard() { bucket.lock.Unlock(); }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    private:
        Bucket& bucket;
    };

    void LockAll();
    void UnlockAll();

    friend Bucket* GetExclusiveMonitorBuckets(ExclusiveMonitor*);
    friend Reservation* GetExclusiveMonitorReservation(ExclusiveMonitor*, std::size_t);

    std::array<Bucket, BUCKET_COUNT> buckets{};
    std::vector<Reservation> reservations;
};

static_assert(sizeof(ExclusiveMonitor::Bucket) == 64);
static_assert(sizeof(ExclusiveMonitor::Reservation) == 64);
static_assert(std::is_standard_layout_v<ExclusiveMonitor::Bucket>);
static_assert(std::is_standard_layout_v<ExclusiveMonitor::Reservation>);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t));

}  // namespace Dynarmic
