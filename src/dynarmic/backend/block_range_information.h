/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <set>
#include <span>

#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/backend/page_range_index.h"
#include "dynarmic/ir/location_descriptor.h"

#ifndef A2E_DYNARMIC_PAGE_RANGE_INDEX
#define A2E_DYNARMIC_PAGE_RANGE_INDEX 1
#endif

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
class BlockRangeInformation {
public:
    void AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location);
    void ClearCache();
    void RemoveLocations(const tsl::robin_set<IR::LocationDescriptor>& locations);
    void RemoveLocations(std::span<const u64> locations);
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);

private:
#if A2E_DYNARMIC_PAGE_RANGE_INDEX
    PageRangeIndex<ProgramCounterType,
                   tsl::robin_map<ProgramCounterType, std::vector<std::uint64_t>>,
                   tsl::robin_map<std::uint64_t, PageRangeRecord<ProgramCounterType>>> block_ranges;
#else
    boost::icl::interval_map<ProgramCounterType, std::set<IR::LocationDescriptor>> block_ranges;
#endif
};

}  // namespace Dynarmic::Backend
