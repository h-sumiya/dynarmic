/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
#if A2E_DYNARMIC_PAGE_RANGE_INDEX
    if (!boost::icl::is_empty(range))
        block_ranges.Add(boost::icl::first(range), boost::icl::last(range), location.Value());
#else
    block_ranges.add(std::make_pair(range, std::set<IR::LocationDescriptor>{location}));
#endif
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
#if A2E_DYNARMIC_PAGE_RANGE_INDEX
    block_ranges.Clear();
#else
    block_ranges.clear();
#endif
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::RemoveLocations([[maybe_unused]] const tsl::robin_set<IR::LocationDescriptor>& locations) {
#if A2E_DYNARMIC_PAGE_RANGE_INDEX
    for (const auto& location : locations)
        block_ranges.Remove(location.Value());
#endif
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    for (auto invalidate_interval : ranges) {
#if A2E_DYNARMIC_PAGE_RANGE_INDEX
        if (!boost::icl::is_empty(invalidate_interval)) {
            for (const auto descriptor : block_ranges.Invalidate(boost::icl::first(invalidate_interval), boost::icl::last(invalidate_interval)))
                erase_locations.insert(IR::LocationDescriptor{descriptor});
        }
#else
        auto pair = block_ranges.equal_range(invalidate_interval);
        for (auto it = pair.first; it != pair.second; ++it) {
            for (const auto& descriptor : it->second) {
                erase_locations.insert(descriptor);
            }
        }
#endif
    }
    return erase_locations;
}

template class BlockRangeInformation<u32>;
template class BlockRangeInformation<u64>;

}  // namespace Dynarmic::Backend
