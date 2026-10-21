/* apk2exe Dynarmic cache metadata index.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
struct PageRangeRecord {
    ProgramCounterType first;
    ProgramCounterType last;
};

// Keep records once per live descriptor and only an eight-byte descriptor in
// each page bucket. PageMap and BlockMap may be flat hash maps; the defaults
// allow the exact index implementation to be tested without JIT dependencies.
template<typename ProgramCounterType,
         typename PageMap = std::unordered_map<ProgramCounterType, std::vector<std::uint64_t>>,
         typename BlockMap = std::unordered_map<std::uint64_t, PageRangeRecord<ProgramCounterType>>>
class PageRangeIndex {
    static_assert(std::is_unsigned_v<ProgramCounterType>);

public:
    static constexpr unsigned page_bits = 12;
    // An unusually large translation must not allocate one bucket per guest
    // page, especially for a range extending to the top of an address space.
    static constexpr ProgramCounterType maximum_indexed_pages = 16;

    void Add(ProgramCounterType first, ProgramCounterType last, std::uint64_t descriptor) {
        if (first > last)
            return;
        // A descriptor denotes one live emitted block. Recompiling it replaces
        // its previous range, rather than retaining obsolete invalidation data.
        const auto previous = blocks.find(descriptor);
        if (previous != blocks.end()) {
            Replace(first, last, descriptor, previous->second);
            return;
        }
        try {
            if (IsWide(first, last)) {
                wide_blocks.push_back(descriptor);
            } else {
                ForPages(first, last, [&](ProgramCounterType page) {
                    pages[page].push_back(descriptor);
                });
            }
            // Publish the authoritative record last. A failed bucket/map
            // allocation must not leave a record whose page buckets are absent.
            blocks.emplace(descriptor, PageRangeRecord<ProgramCounterType>{first, last});
        } catch (...) {
            if (IsWide(first, last)) {
                wide_blocks.erase(std::remove(wide_blocks.begin(), wide_blocks.end(), descriptor), wide_blocks.end());
            } else {
                ForPages(first, last, [&](ProgramCounterType page) {
                    const auto found = pages.find(page);
                    if (found == pages.end())
                        return;
                    auto& bucket = pages.at(page);
                    bucket.erase(std::remove(bucket.begin(), bucket.end(), descriptor), bucket.end());
                    if (bucket.empty())
                        pages.erase(found);
                });
            }
            throw;
        }
    }

    // Intervals are closed. Page buckets identify candidates only; the byte
    // interval test below is always authoritative, including for shared pages.
    std::vector<std::uint64_t> FindOverlaps(ProgramCounterType first, ProgramCounterType last) const {
        std::vector<std::uint64_t> removed;
        if (first > last || blocks.empty())
            return removed;

        const auto collect = [&](const auto& descriptors) {
            for (const auto descriptor : descriptors) {
                const auto record = blocks.find(descriptor);
                if (record != blocks.end() && record->second.first <= last && first <= record->second.last)
                    removed.push_back(descriptor);
            }
        };

        const ProgramCounterType first_page = first >> page_bits;
        const ProgramCounterType last_page = last >> page_bits;
        // Never walk the whole virtual address space for a broad invalidation.
        // Iterating the smaller of the requested span and resident page table
        // also covers the full u32/u64 interval without an overflow-prone +1.
        if (last_page - first_page < pages.size()) {
            ForPages(first, last, [&](ProgramCounterType page) {
                const auto found = pages.find(page);
                if (found != pages.end())
                    collect(found->second);
            });
        } else {
            for (const auto& [page, descriptors] : pages) {
                if (first_page <= page && page <= last_page)
                    collect(descriptors);
            }
        }
        collect(wide_blocks);
        SortUnique(removed);

        return removed;
    }

    std::vector<std::uint64_t> Invalidate(ProgramCounterType first, ProgramCounterType last) {
        auto removed = FindOverlaps(first, last);
        RemoveMany(removed);
        return removed;
    }

    void Clear() {
        pages.clear();
        blocks.clear();
        wide_blocks.clear();
    }

    void Remove(std::uint64_t descriptor) { EraseOne(descriptor); }

    // Segment retirement removes many descriptors together. Gather affected
    // pages first, erase the records, then compact each page bucket once.
    // Calling EraseOne for every descriptor scans the same densely translated
    // page repeatedly and makes a single retirement quadratic in its blocks.
    template<typename DescriptorRange, typename Key>
    void RemoveMany(const DescriptorRange& descriptors, Key&& key) {
        std::vector<ProgramCounterType> touched_pages;
        bool touched_wide = false;
        for (const auto& value : descriptors) {
            const auto found = blocks.find(key(value));
            if (found == blocks.end())
                continue;
            const auto record = found->second;
            if (IsWide(record.first, record.last)) {
                touched_wide = true;
            } else {
                ForPages(record.first, record.last, [&](ProgramCounterType page) {
                    touched_pages.push_back(page);
                });
            }
        }
        // All allocations finish before changing the authoritative records.
        SortUnique(touched_pages);
        for (const auto& value : descriptors)
            blocks.erase(key(value));
        for (const auto page : touched_pages) {
            auto& bucket = pages.at(page);
            bucket.erase(std::remove_if(bucket.begin(), bucket.end(), [&](std::uint64_t descriptor) {
                return blocks.find(descriptor) == blocks.end();
            }), bucket.end());
            if (bucket.empty())
                pages.erase(page);
        }
        if (touched_wide) {
            wide_blocks.erase(std::remove_if(wide_blocks.begin(), wide_blocks.end(), [&](std::uint64_t descriptor) {
                return blocks.find(descriptor) == blocks.end();
            }), wide_blocks.end());
        }
    }

    template<typename DescriptorRange>
    void RemoveMany(const DescriptorRange& descriptors) {
        RemoveMany(descriptors, [](std::uint64_t descriptor) { return descriptor; });
    }

    std::size_t BlockCount() const { return blocks.size(); }
    std::size_t PageCount() const { return pages.size(); }
    std::size_t WideBlockCount() const { return wide_blocks.size(); }

private:
    void Replace(ProgramCounterType first, ProgramCounterType last,
                 std::uint64_t descriptor, PageRangeRecord<ProgramCounterType> previous) {
        const bool previous_wide = IsWide(previous.first, previous.last);
        const bool next_wide = IsWide(first, last);
        const auto in_previous_pages = [&](ProgramCounterType page) {
            return !previous_wide && (previous.first >> page_bits) <= page &&
                   page <= (previous.last >> page_bits);
        };
        // Add only new bucket memberships first. The old record and its
        // buckets stay intact if any allocation fails during replacement.
        try {
            if (next_wide && !previous_wide) {
                wide_blocks.push_back(descriptor);
            } else if (!next_wide) {
                ForPages(first, last, [&](ProgramCounterType page) {
                    if (!in_previous_pages(page))
                        pages[page].push_back(descriptor);
                });
            }
        } catch (...) {
            if (next_wide && !previous_wide) {
                wide_blocks.erase(std::remove(wide_blocks.begin(), wide_blocks.end(), descriptor), wide_blocks.end());
            } else if (!next_wide) {
                ForPages(first, last, [&](ProgramCounterType page) {
                    if (in_previous_pages(page))
                        return;
                    const auto found = pages.find(page);
                    if (found == pages.end())
                        return;
                    auto& bucket = pages.at(page);
                    bucket.erase(std::remove(bucket.begin(), bucket.end(), descriptor), bucket.end());
                    if (bucket.empty())
                        pages.erase(found);
                });
            }
            throw;
        }
        // Updating the existing scalar record and dropping old memberships
        // require no allocation. Readers of this owner-only index see one
        // authoritative range for the descriptor throughout the operation.
        blocks.at(descriptor) = {first, last};
        if (previous_wide && !next_wide) {
            wide_blocks.erase(std::remove(wide_blocks.begin(), wide_blocks.end(), descriptor), wide_blocks.end());
        } else if (!previous_wide) {
            ForPages(previous.first, previous.last, [&](ProgramCounterType page) {
                if (!next_wide && (first >> page_bits) <= page && page <= (last >> page_bits))
                    return;
                auto& bucket = pages.at(page);
                bucket.erase(std::remove(bucket.begin(), bucket.end(), descriptor), bucket.end());
                if (bucket.empty())
                    pages.erase(page);
            });
        }
    }

    template<typename Value>
    static void SortUnique(std::vector<Value>& values) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }

    static bool IsWide(ProgramCounterType first, ProgramCounterType last) {
        return (last >> page_bits) - (first >> page_bits) >= maximum_indexed_pages;
    }

    template<typename Function>
    static void ForPages(ProgramCounterType first, ProgramCounterType last, Function&& function) {
        const auto last_page = last >> page_bits;
        for (auto page = first >> page_bits;; ++page) {
            function(page);
            if (page == last_page)
                break;
        }
    }

    void EraseOne(std::uint64_t descriptor) {
        const auto found = blocks.find(descriptor);
        if (found == blocks.end())
            return;
        const auto record = found->second;
        if (IsWide(record.first, record.last)) {
            wide_blocks.erase(std::remove(wide_blocks.begin(), wide_blocks.end(), descriptor), wide_blocks.end());
        } else {
            ForPages(record.first, record.last, [&](ProgramCounterType page) {
                auto& descriptors = pages.at(page);
                descriptors.erase(std::remove(descriptors.begin(), descriptors.end(), descriptor), descriptors.end());
                if (descriptors.empty())
                    pages.erase(page);
            });
        }
        blocks.erase(descriptor);
    }

    PageMap pages;
    BlockMap blocks;
    std::vector<std::uint64_t> wide_blocks;
};

}  // namespace Dynarmic::Backend
