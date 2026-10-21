/* This file is part of apk2exe's pinned Dynarmic patches.
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/interface/exclusive_monitor.h"

#include <mcl/assert.hpp>

namespace Dynarmic {

ExclusiveMonitor::ExclusiveMonitor(std::size_t processor_count)
        : reservations(processor_count) {}

std::size_t ExclusiveMonitor::GetProcessorCount() const {
    return reservations.size();
}

void ExclusiveMonitor::LockAll() {
    // Normal operations acquire one bucket only. This fixed ordering makes
    // Clear/ClearProcessor safe even when another processor is still running.
    for (auto& bucket : buckets) {
        bucket.lock.Lock();
    }
}

void ExclusiveMonitor::UnlockAll() {
    for (auto it = buckets.rbegin(); it != buckets.rend(); ++it) {
        it->lock.Unlock();
    }
}

void ExclusiveMonitor::Clear() {
    LockAll();
    for (auto& reservation : reservations) {
        reservation.epoch.store(0, std::memory_order_relaxed);
    }
    UnlockAll();
}

void ExclusiveMonitor::ClearProcessor(std::size_t processor_id) {
    ASSERT(processor_id < reservations.size());
    LockAll();
    reservations[processor_id].epoch.store(0, std::memory_order_relaxed);
    UnlockAll();
}

void ExclusiveMonitor::ResetEpochAfterWrap(Bucket& bucket) {
    ASSERT(bucket.epoch == 0);
    for (auto& reservation : reservations) {
        reservation.epoch.store(0, std::memory_order_relaxed);
    }
    bucket.epoch = 1;
}

}  // namespace Dynarmic
