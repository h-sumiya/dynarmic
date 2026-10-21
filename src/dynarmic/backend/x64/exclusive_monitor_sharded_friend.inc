/* This file is part of apk2exe's pinned Dynarmic patches.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic {

inline ExclusiveMonitor::Bucket* GetExclusiveMonitorBuckets(ExclusiveMonitor* monitor) {
    return monitor->buckets.data();
}

inline ExclusiveMonitor::Reservation* GetExclusiveMonitorReservation(ExclusiveMonitor* monitor, std::size_t index) {
    return monitor->reservations.data() + index;
}

}  // namespace Dynarmic
