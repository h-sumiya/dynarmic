// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT

BaselineReclamation A64EmitX64::PrepareBaselineEmission(size_t body_size) {
    ASSERT(!shared_optimized);
    // The independent compiler's scratch is bounded at 512 KiB. Keep at least
    // two segments large enough for its largest body plus entry/alignment.
    constexpr size_t minimum_segment_size = 512 * 1024 + 64;
    constexpr size_t entry_reserve = 32;
    if (body_size > std::numeric_limits<size_t>::max() - entry_reserve)
        throw std::length_error("baseline body size overflows its entry reservation");
    if (!baseline_cache.Initialized()) {
        const auto begin = reinterpret_cast<std::uintptr_t>(code.GetCodeBegin());
        const auto end = reinterpret_cast<std::uintptr_t>(code.getCode()) + code.GetTotalCodeSize();
        baseline_cache.Initialize(begin, end - begin, conf.baseline_cache_segment_size, minimum_segment_size);
    }

    BaselineReclamation result;
    const auto required = body_size + entry_reserve;
    const auto cursor = baseline_cache.Prepare(required, [&](const A64BaselineCodeCache::Segment& segment) {
        result.evicted_locations.reserve(segment.allocations.size());
        tsl::robin_set<IR::LocationDescriptor> locations;
        locations.reserve(segment.allocations.size());
        // Baseline-only emitters have no outgoing direct patches or inline
        // sites. Native lookup helpers stay executable throughout this batch;
        // avoid a W^X protection transition for each retired block.
        code.DisableWriting();
        for (const auto& allocation : segment.allocations) {
            const IR::LocationDescriptor location{allocation.location};
            const auto entrypoint = reinterpret_cast<CodePtr>(allocation.entrypoint);
            const auto current = block_descriptors.find(location);
            if (current != block_descriptors.end() && current->second.entrypoint == entrypoint) {
                locations.insert(location);
                result.evicted_locations.push_back(allocation.location);
                block_descriptors.erase(current);
            }
            if (fast_dispatch_table_lookup) {
                auto& dispatch = (*fast_dispatch_table_lookup)(allocation.location);
                if (dispatch.code_ptr == entrypoint)
                    dispatch = {};
            }
            // Retain no empty patch buckets for evicted descriptors.
            const auto patch = patch_information.find(location);
            if (patch != patch_information.end() && patch->second.jg.empty() && patch->second.jz.empty() &&
                patch->second.jmp.empty() && patch->second.mov_rcx.empty()) {
                patch_information.erase(patch);
            }
        }
        block_ranges.RemoveLocations(locations);
        for (const auto rip : segment.fault_rips)
            fastmem_patch_info.erase(rip);
        result.reclaimed_blocks = segment.allocations.size();
        result.reclaimed_bytes = segment.used;
        result.reclaimed_segments = 1;
    });
    code.SetCodePtr(reinterpret_cast<CodePtr>(cursor));
    code.EnsureMemoryCommitted(required);
    return result;
}
