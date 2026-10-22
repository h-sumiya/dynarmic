/* SPDX-License-Identifier: 0BSD */

#include "dynarmic/backend/x64/a64_shared_code_cache.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/page_range_index.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/a64_runtime_callback.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::A64 {
using namespace Backend::X64;
namespace {

auto TranslationContract(const UserConfig& c) {
    // Deliberately exclude callbacks, processor_id, TLS addresses and local
    // cache sizes. Every value embedded by an optimized emitter is included.
    return std::tuple{
        c.global_monitor, c.optimizations, c.unsafe_optimizations,
        c.hook_data_cache_operations, c.hook_isb, c.hook_hint_instructions,
        c.cntfrq_el0, c.ctr_el0, c.dczid_el0,
        c.tpidr_el0 != nullptr, c.tpidrro_el0 != nullptr,
        c.page_table, c.page_table_address_space_bits, c.page_table_pointer_mask_bits,
        c.silently_mirror_page_table, c.absolute_offset_page_table,
        c.detect_misaligned_access_via_page_table,
        c.only_detect_misalignment_via_page_table_on_page_boundary,
        c.fastmem_pointer, c.optimize_zero_base_fastmem,
        c.fastmem_address_space_bits, c.silently_mirror_fastmem,
        c.fastmem_exclusive_access, c.define_unpredictable_behaviour,
        c.wall_clock_cntpct, c.check_halt_on_memory_access, c.memory_callbacks_preserve_state,
        c.enable_cycle_counting, c.very_verbose_debugging_output};
}

bool OptimizableTerminal(const IR::Terminal& terminal) {
    if (boost::get<IR::Term::Interpret>(&terminal) || boost::get<IR::Term::Invalid>(&terminal))
        return false;
    if (const auto* term = boost::get<IR::Term::If>(&terminal))
        return OptimizableTerminal(term->then_) && OptimizableTerminal(term->else_);
    if (const auto* term = boost::get<IR::Term::CheckBit>(&terminal))
        return OptimizableTerminal(term->then_) && OptimizableTerminal(term->else_);
    if (const auto* term = boost::get<IR::Term::CheckHalt>(&terminal))
        return OptimizableTerminal(term->else_);
    return true;
}

RunCodeCallbacks SharedRunCallbacks(const UserConfig& conf) {
    return {
        // Normal shared terminals always use the owner's permanent prelude.
        // Keep a valid, fail-closed callback for the unused generic run entry.
        std::make_unique<ArgCallback>(+[](void*) -> CodePtr { std::terminate(); }, 0),
        std::make_unique<ArgCallback>(MakeA64Callback<&UserCallbacks::AddTicks>(nullptr, true)),
        std::make_unique<ArgCallback>(MakeA64Callback<&UserCallbacks::GetTicksRemaining>(nullptr, true)),
        conf.enable_cycle_counting,
    };
}

UserConfig WorkerConfig(UserConfig conf) {
    conf.shared_code_cache.reset(); // Never keep an ownership cycle.
    conf.callbacks = nullptr;      // Workers may not call a dying Jit.
    conf.processor_id = 0;         // Shared emitter loads the actual reservation.
    conf.recompile_on_fastmem_failure = false;
    conf.recompile_on_exclusive_fastmem_failure = false;
    return conf;
}

struct IncomingLink {
    std::atomic<const void*>* cell;
    const void* fallback;
    IncomingLink* previous;
    IncomingLink* next;
    std::uint64_t target;
};

struct SharedRegion {
    explicit SharedRegion(const UserConfig& conf, std::size_t bytes)
        : code(SharedRunCallbacks(conf), JitStateInfo{layout}, bytes, [](BlockOfCode&) {}, true),
          emitter(code, conf, nullptr, true) {}

    A64JitState layout;
    BlockOfCode code;
    A64EmitX64 emitter;
    std::vector<std::unique_ptr<SharedPublishedBlock>> records;
    // The cells reside in this region's writable data pages. Deque element
    // addresses remain stable while appending. Intrusive per-target chains
    // let retirement unlink exactly this source's edges without scanning or
    // copying every other region that calls the same popular helper.
    std::deque<IncomingLink> links;
    bool sealed = false;
    bool building = false;
    bool retired = false;
    bool retire_requested = false;
    std::uint64_t retire_epoch = 0;
};

} // namespace

struct SharedCodeCache::Impl {
    struct QueuedJob {
        SharedCodeSnapshot snapshot;
        std::chrono::steady_clock::time_point enqueued;
    };
    struct Reclamation {
        std::unique_ptr<SharedRegion> region;
        std::chrono::steady_clock::time_point started;
    };
    explicit Impl(SharedCodeCacheConfig config_) : config(config_) {
        constexpr std::size_t mib = 1024 * 1024;
        if (config.region_size < 8 * mib || config.region_size > 512 * mib ||
            (config.region_size % (64 * 1024)) != 0 ||
            config.worker_count == 0 || config.worker_count > 4 ||
            config.capacity_bytes / config.region_size < config.worker_count + 1 ||
            config.max_pending_jobs == 0 || config.max_pending_jobs > 1048576 ||
            config.max_blocks < 256 || config.max_blocks > 4194304 ||
            !std::has_single_bit(config.max_blocks)) {
            throw std::invalid_argument("Invalid shared A64 cache budget, region, worker or index size");
        }
        index = std::make_unique<std::atomic<const SharedPublishedBlock*>[]>(config.max_blocks);
        for (std::size_t i = 0; i < config.max_blocks; ++i)
            index[i].store(nullptr, std::memory_order_relaxed);
        statistics.capacity_bytes = config.capacity_bytes;
        active.resize(config.worker_count, nullptr);
        replacement.resize(config.max_blocks / 4, 0);
        try {
            for (std::size_t i = 0; i < config.worker_count; ++i)
                workers.emplace_back([this, i] { Worker(i); });
        } catch (...) {
            { std::lock_guard lock(mutex); stopping = true; }
            changed.notify_all();
            for (auto& worker : workers)
                worker.join();
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            queue.clear();
        }
        changed.notify_all();
        for (auto& worker : workers)
            worker.join();
        // Every attached Jit owns its shared_ptr<SharedCodeCache>; therefore
        // there cannot be a native reader when this final owner is destroyed.
        ASSERT(readers.empty());
    }

    std::size_t Slot(std::uint64_t location) const {
        return (static_cast<std::size_t>(SharedCodeHash(location)) & (config.max_blocks / 4 - 1)) * 4;
    }

    static std::uint64_t RangeKey(const SharedGuestRange& range) {
        return reinterpret_cast<std::uintptr_t>(&range);
    }

    static const SharedPublishedBlock* RangeOwner(std::uint64_t key) {
        return reinterpret_cast<const SharedGuestRange*>(static_cast<std::uintptr_t>(key))->owner;
    }

    void PublishIndex(const SharedPublishedBlock* record) {
        const auto first = Slot(record->location);
        for (std::size_t i = 0; i < 4; ++i) {
            const auto* current = index[first + i].load(std::memory_order_relaxed);
            if (!current || current->location == record->location) {
                index[first + i].store(record, std::memory_order_release);
                return;
            }
        }
        auto& victim = replacement[first / 4];
        index[first + (victim++ & 3)].store(record, std::memory_order_release);
    }

    void UpdateIncoming(std::uint64_t location, const void* entrypoint) {
        const auto found = incoming.find(location);
        if (found == incoming.end())
            return;
        for (auto* link = found->second; link; link = link->next)
            link->cell->store(entrypoint ? entrypoint : link->fallback, std::memory_order_release);
    }

    void RegisterIncoming(SharedRegion& source, std::uint64_t location,
                          std::atomic<const void*>* cell, const void* fallback) {
        std::lock_guard lock(mutex);
        auto [found, inserted] = incoming.try_emplace(location, nullptr);
        try {
            source.links.push_back({cell, fallback, nullptr, found->second, location});
        } catch (...) {
            if (inserted)
                incoming.erase(found);
            throw;
        }
        // No allocation remains after exposing the new node to the target's
        // chain. A failed append leaves every prior head/link unchanged.
        auto* link = &source.links.back();
        if (found->second)
            found->second->previous = link;
        found->second = link;
        ++statistics.link_cells;
        const auto completed_target = completed.find(location);
        cell->store(completed_target == completed.end() ? fallback : completed_target->second->entrypoint,
                    std::memory_order_release);
    }

    void RemoveIncomingSource(SharedRegion& source) {
        for (auto& link : source.links) {
            if (link.previous) {
                link.previous->next = link.next;
            } else {
                const auto found = incoming.find(link.target);
                ASSERT(found != incoming.end() && found->second == &link);
                if (link.next)
                    found->second = link.next;
                else
                    incoming.erase(found);
            }
            if (link.next)
                link.next->previous = link.previous;
        }
        statistics.link_cells -= source.links.size();
    }

    void RemovePublished(SharedRegion& region) {
        // Clear already removed all published slots and link targets. Do not
        // scan historical per-region records again while retiring its slabs.
        if (completed.empty())
            return;
        try {
            std::vector<std::uint64_t> removed;
            removed.reserve(region.records.size());
            for (const auto& record : region.records) {
                const auto found = completed.find(record->location);
                if (found != completed.end() && found->second == record.get()) {
                    for (const auto& range : record->guest_ranges)
                        removed.push_back(RangeKey(range));
                }
            }
            published_ranges.RemoveMany(removed);
        } catch (const std::bad_alloc&) {
            // RemoveMany completes all allocations before changing records.
            // Retirement must still make progress under allocation pressure,
            // including after an invalidation marks a worker's active region.
            // The slower one-range path requires no allocation.
            for (const auto& record : region.records) {
                const auto found = completed.find(record->location);
                if (found != completed.end() && found->second == record.get()) {
                    for (const auto& range : record->guest_ranges)
                        published_ranges.Remove(RangeKey(range));
                }
            }
        }
        for (const auto& record : region.records) {
            const auto first = Slot(record->location);
            for (std::size_t i = 0; i < 4; ++i) {
                auto& slot = index[first + i];
                if (slot.load(std::memory_order_relaxed) == record.get())
                    slot.store(nullptr, std::memory_order_release);
            }
            const auto found = completed.find(record->location);
            if (found != completed.end() && found->second == record.get()) {
                UpdateIncoming(record->location, nullptr);
                completed.erase(found);
            }
        }
        statistics.live_blocks = completed.size();
    }

    void Retire(SharedRegion& region) {
        ASSERT(region.sealed && !region.building && !region.retired);
        const auto started = std::chrono::steady_clock::now();
        for (auto*& ptr : active)
            if (ptr == &region) ptr = nullptr;
        RemovePublished(region);
        // SC epochs close the reader-publication/collector scan race. Epoch
        // refresh occurs only at execution entry and C++ lookup safepoints.
        region.retire_epoch = generation.fetch_add(1, std::memory_order_seq_cst) + 1;
        region.retired = true;
        statistics.retired_bytes += config.region_size;
        ++retired_regions;
        const auto elapsed = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
        statistics.retirement_nanoseconds += elapsed;
        statistics.retirement_max_nanoseconds = std::max(statistics.retirement_max_nanoseconds, elapsed);
        changed.notify_all();
    }

    Reclamation TakeRetiredRegion() {
        if (retired_regions == 0)
            return {};
        for (auto it = regions.begin(); it != regions.end();) {
            auto& region = **it;
            if (!region.retired) { ++it; continue; }
            const bool protected_reader = std::any_of(readers.begin(), readers.end(), [&](const auto* reader) {
                const auto epoch = reader->epoch.load(std::memory_order_seq_cst);
                return epoch != 0 && epoch < region.retire_epoch;
            });
            if (protected_reader) { ++it; continue; }
            const auto started = std::chrono::steady_clock::now();
            RemoveIncomingSource(region);
            auto result = std::move(*it);
            regions.erase(it);
            // Still charge this physical allocation until the worker has
            // destroyed it. Another worker must not overcommit the budget.
            return {std::move(result), started};
        }
        return {};
    }

    void ReclaimRegion(Reclamation reclamation) {
        // Destruction includes metadata deletion, fault-registry removal and
        // unmapping executable pages. Only an optimizer worker does this, and
        // never while holding the scheduler mutex or an execution-thread Jit.
        reclamation.region.reset();
        const auto elapsed = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - reclamation.started).count());
        {
            std::lock_guard lock(mutex);
            statistics.allocated_bytes -= config.region_size;
            statistics.retired_bytes -= config.region_size;
            statistics.reclaimed_bytes += config.region_size;
            ++statistics.reclaimed_regions;
            --retired_regions;
            statistics.reclamation_nanoseconds += elapsed;
            statistics.reclamation_max_nanoseconds = std::max(statistics.reclamation_max_nanoseconds, elapsed);
        }
        changed.notify_all();
    }

    SharedRegion* RegionFor(std::size_t worker, std::unique_lock<std::mutex>& lock) {
        constexpr std::size_t emission_reserve = 1024 * 1024;
        auto*& current = active[worker];
        if (current && (current->sealed || current->code.SpaceRemaining() < emission_reserve)) {
            current->sealed = true;
            current = nullptr;
        }
        if (current)
            return current;
        if (statistics.allocated_bytes + config.region_size > config.capacity_bytes) {
            // Retiring more regions cannot make progress while an earlier
            // victim awaits a reader or a worker's physical destruction. Keep
            // the remaining useful publications alive instead of cascading
            // through the whole cache during that grace period.
            if (retired_regions != 0)
                return nullptr;
            // FIFO region retirement clears incoming mutable link cells and
            // removes publication before advancing the reader epoch. Existing
            // readers retain the victim until their next dispatch safepoint.
            const auto victim = std::find_if(regions.begin(), regions.end(), [](const auto& region) {
                return region->sealed && !region->building && !region->retired;
            });
            if (victim != regions.end())
                Retire(**victim);
        }
        if (statistics.allocated_bytes + config.region_size > config.capacity_bytes)
            return nullptr;
        // Charge the reservation before releasing the mutex. Other workers
        // must count an in-flight construction against the physical limit.
        // Construct its list node outside the lock too, so acceptance below
        // cannot fail after exposing the allocation or leaking its charge.
        std::list<std::unique_ptr<SharedRegion>> created;
        const auto construction_revision = revision.load(std::memory_order_relaxed);
        statistics.allocated_bytes += config.region_size;
        lock.unlock();
        try {
            // compile_config is set by the first Attach and then immutable.
            // Workers retain only WorkerConfig, never execution callbacks.
            created.push_back(std::make_unique<SharedRegion>(*compile_config, config.region_size));
            auto* source = created.front().get();
            source->emitter.SetSharedLinkResolver([this, source](u64 location, std::atomic<const void*>* cell, const void* fallback) {
                RegisterIncoming(*source, location, cell, fallback);
            });
        } catch (...) {
            // Even partial/prelude destruction may unmap native memory. Keep
            // that work outside the mutex, and keep its reservation charged
            // until every successfully constructed object is gone.
            created.clear();
            lock.lock();
            statistics.allocated_bytes -= config.region_size;
            changed.notify_all();
            throw;
        }
        lock.lock();
        if (stopping || construction_revision != revision.load(std::memory_order_relaxed)) {
            // A concurrent Clear must not leave an unpublished empty region
            // resident, and its canceled job must never execute afterward.
            lock.unlock();
            created.clear();
            lock.lock();
            statistics.allocated_bytes -= config.region_size;
            changed.notify_all();
            return nullptr;
        }
        regions.splice(regions.end(), created);
        current = regions.back().get();
        return current;
    }

    std::unique_ptr<SharedPublishedBlock> Compile(SharedRegion& region, const SharedCodeSnapshot& job) {
        const auto& conf = *compile_config;
        const LocationDescriptor location{IR::LocationDescriptor{job.location}};
        const auto first_pc = location.PC();
        std::vector<std::uint64_t> guest_words;
        guest_words.reserve(job.instructions.size() + job.successors.size());
        // Include the entry even for a translated instruction-fetch fault.
        guest_words.push_back(first_pc);
        const auto get_code = [&](std::uint64_t pc) -> std::optional<std::uint32_t> {
            if ((pc & 3) != 0)
                return std::nullopt;
            std::optional<std::uint32_t> instruction;
            if (pc >= first_pc && (pc - first_pc) / 4 < job.instructions.size()) {
                instruction = job.instructions[(pc - first_pc) / 4];
            } else {
                const auto found = std::lower_bound(job.successors.begin(), job.successors.end(), pc,
                    [](const auto& word, std::uint64_t address) { return word.first < address; });
                if (found != job.successors.end() && found->first == pc)
                    instruction = found->second;
            }
            if (instruction)
                guest_words.push_back(pc);
            return instruction;
        };
        IR::Block ir = TranslateTrace(location, get_code,
            {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct, conf.hook_hint_instructions},
            conf.HasOptimization(OptimizationFlag::GetSetElimination) && conf.HasOptimization(OptimizationFlag::ConstProp));
        if (OptimizableTerminal(ir.GetTerminal())) {
            Optimization::PolyfillPass(ir, {
                .sha256 = !region.code.HasHostFeature(HostFeature::SHA),
                .vector_multiply_widen = true,
            });
            Optimization::A64CallbackConfigPass(ir, conf);
            Optimization::NamingPass(ir);
            if (conf.HasOptimization(OptimizationFlag::GetSetElimination)) {
                Optimization::A64GetSetElimination(ir, conf.check_halt_on_memory_access, conf.memory_callbacks_preserve_state);
                Optimization::DeadCodeElimination(ir);
            }
            if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
                Optimization::ConstantPropagation(ir);
                Optimization::DeadCodeElimination(ir);
            }
            Optimization::IdentityRemovalPass(ir);
#ifndef NDEBUG
            Optimization::VerificationPass(ir);
#endif
            region.code.EnsureMemoryCommitted(1024 * 1024);
            const auto block = region.emitter.Emit(ir);
            auto result = std::make_unique<SharedPublishedBlock>(SharedPublishedBlock{
                job.location, block.entrypoint, first_pc, first_pc + 3});
            std::sort(guest_words.begin(), guest_words.end());
            guest_words.erase(std::unique(guest_words.begin(), guest_words.end()), guest_words.end());
            result->guest_ranges.reserve(guest_words.size());
            for (const auto pc : guest_words) {
                if (!result->guest_ranges.empty() && result->guest_ranges.back().last != std::numeric_limits<u64>::max() &&
                    result->guest_ranges.back().last + 1 == pc) {
                    result->guest_ranges.back().last = pc + 3;
                } else {
                    result->guest_ranges.push_back({pc, pc + 3});
                }
            }
            result->guest_start = result->guest_ranges.front().first;
            result->guest_end = result->guest_ranges.back().last;
            return result;
        }
        return nullptr;
    }

    void Worker(std::size_t worker) {
        struct Publication {
            std::uint64_t location = 0;
            std::uint64_t revision = 0;
            std::unique_ptr<SharedPublishedBlock> block;
            bool retry = false;
        };
        for (;;) {
            SharedCodeSnapshot job;
            SharedRegion* region = nullptr;
            Reclamation reclamation;
            bool have_job = false;
            {
                std::unique_lock lock(mutex);
                for (;;) {
                    if (stopping)
                        return;
                    reclamation = TakeRetiredRegion();
                    if (reclamation.region || !queue.empty())
                        break;
                    if (retired_regions != 0) {
                        // Unpin stays a read-mostly-path store, with no mutex,
                        // notification or shared RMW. Poll epochs only while
                        // there is actual retirement debt to collect.
                        changed.wait_for(lock, std::chrono::milliseconds(1));
                    } else {
                        changed.wait(lock, [&] { return stopping || !queue.empty() || retired_regions != 0; });
                    }
                }
                if (!reclamation.region) {
                    have_job = true;
                    const auto queued_for = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - queue.front().enqueued).count());
                    statistics.queue_wait_nanoseconds += queued_for;
                    statistics.queue_wait_max_nanoseconds = std::max(statistics.queue_wait_max_nanoseconds, queued_for);
                    job = std::move(queue.front().snapshot);
                    queue.pop_front();
                    if (job.revision != revision.load(std::memory_order_relaxed)) {
                        pending.erase(job.location);
                        ++statistics.discarded_jobs;
                        continue;
                    }
                    try {
                        region = RegionFor(worker, lock);
                    } catch (...) {
                        pending.erase(job.location);
                        ++statistics.failed_jobs;
                        continue;
                    }
                    // Region construction releases the mutex. Clear or stop
                    // may have canceled this snapshot while it was outside.
                    if (stopping)
                        return;
                    if (job.revision != revision.load(std::memory_order_relaxed)) {
                        pending.erase(job.location);
                        ++statistics.discarded_jobs;
                        continue;
                    }
                    if (!region) {
                        // RegionFor may just have retired one victim. Keep the
                        // admitted snapshot while reclaiming it outside the
                        // lock, then retry once without resnapshotting code.
                        reclamation = TakeRetiredRegion();
                        if (!reclamation.region) {
                            pending.erase(job.location);
                            ++statistics.dropped_jobs;
                            continue;
                        }
                    } else {
                        region->building = true;
                    }
                }
            }
            if (reclamation.region) {
                ReclaimRegion(std::move(reclamation));
                if (!have_job)
                    continue;
                std::unique_lock lock(mutex);
                if (stopping)
                    return;
                if (job.revision != revision.load(std::memory_order_relaxed)) {
                    pending.erase(job.location);
                    ++statistics.discarded_jobs;
                    continue;
                }
                try {
                    region = RegionFor(worker, lock);
                } catch (...) {
                    pending.erase(job.location);
                    ++statistics.failed_jobs;
                    continue;
                }
                if (stopping)
                    return;
                if (job.revision != revision.load(std::memory_order_relaxed)) {
                    pending.erase(job.location);
                    ++statistics.discarded_jobs;
                    continue;
                }
                if (!region) {
                    pending.erase(job.location);
                    ++statistics.dropped_jobs;
                    continue;
                }
                region->building = true;
            }

            const auto started = std::chrono::steady_clock::now();
            std::array<Publication, 32> publications;
            std::size_t count = 0;
            std::uint64_t compilation_nanoseconds = 0;
            bool emission_failed = false;
            for (;;) {
                auto& publication = publications[count++];
                publication.location = job.location;
                publication.revision = job.revision;
                const auto compile_started = std::chrono::steady_clock::now();
                try {
                    publication.block = Compile(*region, job);
                } catch (...) {
                    publication.retry = true;
                    emission_failed = true;
                }
                compilation_nanoseconds += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - compile_started).count());
                // Coalesce a partial instruction page within the batch deadline.
                // Foreground execution continues while the worker releases the lock.
                if (emission_failed || count == publications.size() ||
                    region->code.SpaceRemaining() < 1024 * 1024 ||
                    std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(1))
                    break;
                std::unique_lock lock(mutex);
                if (queue.empty() && !region->code.IsCodePageAligned()) {
                    changed.wait_until(lock, started + std::chrono::milliseconds(1), [&] {
                        return stopping || !queue.empty() || region->retire_requested ||
                               job.revision != revision.load(std::memory_order_relaxed);
                    });
                }
                if (stopping || queue.empty() || region->retire_requested ||
                    job.revision != revision.load(std::memory_order_relaxed))
                    break;
                const auto queued_for = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - queue.front().enqueued).count());
                statistics.queue_wait_nanoseconds += queued_for;
                statistics.queue_wait_max_nanoseconds = std::max(statistics.queue_wait_max_nanoseconds, queued_for);
                job = std::move(queue.front().snapshot);
                queue.pop_front();
            }
            const auto seal_started = std::chrono::steady_clock::now();
            try {
                region->code.DisableWriting();
            } catch (...) {
                emission_failed = true;
                for (std::size_t i = 0; i < count; ++i) {
                    publications[i].block.reset();
                    publications[i].retry = true;
                }
            }
            compilation_nanoseconds += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - seal_started).count());

            {
                std::lock_guard lock(mutex);
                region->building = false;
                if (emission_failed)
                    region->sealed = true;
                statistics.compilation_nanoseconds += compilation_nanoseconds;
                if (stopping)
                    return;
                for (std::size_t i = 0; i < count; ++i) {
                    auto& publication = publications[i];
                    auto& result = publication.block;
                    pending.erase(publication.location);
                    if (publication.revision != revision.load(std::memory_order_relaxed)) {
                        ++statistics.discarded_jobs;
                    } else if (!result) {
                        ++statistics.failed_jobs;
                        if (!publication.retry) {
                            try {
                                if (rejected.size() >= config.max_pending_jobs)
                                    rejected.clear();
                                rejected[publication.location] = publication.revision;
                            } catch (...) {}
                        }
                    } else {
                        const auto* published = result.get();
                        try {
                            if (result->guest_ranges.empty())
                                result->guest_ranges.push_back({result->guest_start, result->guest_end});
                            if (result->guest_ranges.size() > SharedPublishedBlock::maximum_guest_ranges)
                                throw std::length_error("shared translation has too many guest ranges");
                            for (auto& range : result->guest_ranges) {
                                if (range.first > range.last)
                                    throw std::invalid_argument("shared translation has an invalid guest range");
                                range.owner = published;
                            }
                            region->records.push_back(std::move(result));
                            try {
                                for (const auto& range : published->guest_ranges)
                                    published_ranges.Add(range.first, range.last, RangeKey(range));
                                if (!completed.emplace(publication.location, published).second)
                                    throw std::logic_error("duplicate shared translation publication");
                            } catch (...) {
                                // Remove uses no allocation. A partial index
                                // insertion or failed completed-map allocation
                                // cannot leave pointers into a freed vector.
                                for (const auto& range : published->guest_ranges)
                                    published_ranges.Remove(RangeKey(range));
                                region->records.pop_back();
                                throw;
                            }
                            // Finalized pages and immutable fault records precede
                            // the release publication of the executable address.
                            PublishIndex(published);
                            UpdateIncoming(publication.location, published->entrypoint);
                            statistics.live_blocks = completed.size();
                            ++statistics.compiled_blocks;
                        } catch (...) {
                            ++statistics.failed_jobs;
                            region->sealed = true;
                        }
                    }
                }
                if (region->retire_requested && !region->retired) {
                    region->sealed = true;
                    for (auto*& ptr : active)
                        if (ptr == region) ptr = nullptr;
                    Retire(*region);
                }
            }
        }
    }

    void Invalidate(std::uint64_t address, std::size_t length, bool all) {
        std::unique_lock lock(mutex);
        revision.fetch_add(1, std::memory_order_release);
        // All pending snapshots are conservatively invalidated. Published
        // unrelated blocks survive range invalidation.
        for (const auto& job : queue)
            pending.erase(job.snapshot.location);
        statistics.discarded_jobs += queue.size();
        queue.clear();
        rejected.clear();
        const auto remove_record = [&](const SharedPublishedBlock* record) {
            const auto first = Slot(record->location);
            for (std::size_t way = 0; way < 4; ++way) {
                auto& slot = index[first + way];
                if (slot.load(std::memory_order_relaxed) == record)
                    slot.store(nullptr, std::memory_order_release);
            }
            UpdateIncoming(record->location, nullptr);
        };
        if (all) {
            for (const auto& [location, record] : completed)
                remove_record(record);
            completed.clear();
            published_ranges.Clear();
        } else {
            std::vector<const SharedPublishedBlock*> affected;
            const auto find_range = [&](u64 first, u64 last) {
                for (const auto key : published_ranges.FindOverlaps(first, last)) {
                    const auto* record = RangeOwner(key);
                    const auto found = completed.find(record->location);
                    if (found != completed.end() && found->second == record)
                        affected.push_back(record);
                }
            };
            const u64 last = address + static_cast<u64>(length - 1);
            if (last >= address) {
                find_range(address, last);
            } else {
                find_range(address, std::numeric_limits<u64>::max());
                find_range(0, last);
            }
            std::sort(affected.begin(), affected.end(), [](const auto* a, const auto* b) {
                return a->location < b->location;
            });
            affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
            std::vector<std::uint64_t> range_keys;
            range_keys.reserve(affected.size());
            for (const auto* record : affected) {
                for (const auto& range : record->guest_ranges)
                    range_keys.push_back(RangeKey(range));
            }
            // Finish every allocation before unpublishing anything. Compact
            // each touched page once even if many traces share disjoint code.
            published_ranges.RemoveMany(range_keys);
            for (const auto* record : affected) {
                remove_record(record);
                completed.erase(record->location);
            }
        }
        statistics.live_blocks = completed.size();
        generation.fetch_add(1, std::memory_order_seq_cst);
        if (all) {
            for (auto& region : regions) {
                region->retire_requested = true;
                if (!region->building && !region->retired) {
                    region->sealed = true;
                    for (auto*& ptr : active)
                        if (ptr == region.get()) ptr = nullptr;
                    Retire(*region);
                }
            }
        }
        lock.unlock();
        changed.notify_all();
    }

    const SharedCodeCacheConfig config;
    std::unique_ptr<std::atomic<const SharedPublishedBlock*>[]> index;
    // Separate cache lines keep the dispatch read-mostly generation apart from
    // worker-owned queue/counter state.
    alignas(64) std::atomic<std::uint64_t> generation{1};
    alignas(64) std::atomic<std::uint64_t> revision{1};
    mutable std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false;
    std::optional<UserConfig> compile_config;
    std::vector<std::thread> workers;
    std::vector<SharedCacheReader*> readers;
    std::deque<QueuedJob> queue;
    std::unordered_map<std::uint64_t, std::uint64_t> pending;
    std::unordered_map<std::uint64_t, std::uint64_t> rejected;
    std::list<std::unique_ptr<SharedRegion>> regions;
    std::vector<SharedRegion*> active;
    std::vector<std::uint8_t> replacement;
    // Includes regions detached for worker destruction. It reaches zero only
    // after their physical memory has been released and the budget debited.
    std::size_t retired_regions = 0;
    // Bounded by resident regions, not by guest address-space history. A native
    // index collision does not cause duplicate compilation of resident code.
    std::unordered_map<std::uint64_t, const SharedPublishedBlock*> completed;
    Backend::PageRangeIndex<u64> published_ranges;
    std::unordered_map<std::uint64_t, IncomingLink*> incoming;
    SharedCodeCacheStatistics statistics;
};

SharedCodeCache::SharedCodeCache(SharedCodeCacheConfig config) : impl(std::make_unique<Impl>(config)) {}
SharedCodeCache::~SharedCodeCache() = default;
void SharedCodeCache::InvalidateRange(std::uint64_t address, std::size_t length) {
    if (length != 0)
        impl->Invalidate(address, length, false);
}
void SharedCodeCache::Clear() { impl->Invalidate(0, 0, true); }
SharedCodeCacheStatistics SharedCodeCache::GetStatistics() const {
    std::lock_guard lock(impl->mutex);
    auto result = impl->statistics;
    result.pending_jobs = impl->pending.size();
    return result;
}

std::unique_ptr<SharedCacheReader> SharedCodeCacheAccess::Attach(SharedCodeCache& cache, const UserConfig& conf) {
    auto& self = *cache.impl;
    std::lock_guard lock(self.mutex);
    if (self.compile_config && TranslationContract(*self.compile_config) != TranslationContract(conf))
        throw std::invalid_argument("Shared A64 cache attached to incompatible Jit configuration");
    if (!self.compile_config)
        self.compile_config = WorkerConfig(conf);
    auto reader = std::make_unique<SharedCacheReader>();
    self.readers.push_back(reader.get());
    return reader;
}
void SharedCodeCacheAccess::Detach(SharedCodeCache& cache, SharedCacheReader& reader) {
    reader.epoch.store(0, std::memory_order_seq_cst);
    {
        std::lock_guard lock(cache.impl->mutex);
        auto& readers = cache.impl->readers;
        readers.erase(std::remove(readers.begin(), readers.end(), &reader), readers.end());
    }
    cache.impl->changed.notify_all();
}
std::uint64_t SharedCodeCacheAccess::Pin(SharedCodeCache& cache, SharedCacheReader& reader) {
    for (;;) {
        const auto epoch = cache.impl->generation.load(std::memory_order_seq_cst);
        // A C++ dispatch within the same Run already protects this epoch.
        // Keep the SC publication handshake only for entry or epoch changes.
        if (reader.epoch.load(std::memory_order_relaxed) == epoch)
            return epoch;
        reader.epoch.store(epoch, std::memory_order_seq_cst);
        if (cache.impl->generation.load(std::memory_order_seq_cst) == epoch)
            return epoch;
    }
}
void SharedCodeCacheAccess::Unpin(SharedCacheReader& reader) {
    // Publishing quiescence needs to order prior code reads, not an SC RMW.
    // A collector that still sees the pinned epoch only postpones reclamation.
    reader.epoch.store(0, std::memory_order_release);
}
const void* SharedCodeCacheAccess::DispatchTable(const SharedCodeCache& cache) { return cache.impl->index.get(); }
std::uint64_t SharedCodeCacheAccess::DispatchMask(const SharedCodeCache& cache) { return cache.impl->config.max_blocks / 4 - 1; }
std::uint64_t SharedCodeCacheAccess::Revision(const SharedCodeCache& cache) { return cache.impl->revision.load(std::memory_order_acquire); }
const void* SharedCodeCacheAccess::Lookup(const SharedCodeCache& cache, std::uint64_t location) {
    auto& self = *cache.impl;
    const auto first = self.Slot(location);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto* block = self.index[first + i].load(std::memory_order_acquire);
        if (block && block->location == location)
            return block->entrypoint;
    }
    // Native-index eviction does not retire the underlying optimized code.
    // On a genuine C++ miss, reuse an existing publication immediately instead
    // of waiting for baseline sampling to rediscover it. Never wait behind a
    // worker: an unavailable scheduler still leaves execution on baseline.
    std::unique_lock lock(self.mutex, std::try_to_lock);
    if (!lock.owns_lock())
        return nullptr;
    const auto found = self.completed.find(location);
    if (found == self.completed.end())
        return nullptr;
    self.PublishIndex(found->second);
    return found->second->entrypoint;
}
void SharedCodeCacheAccess::Capture(SharedCodeSnapshot& snapshot, UserCallbacks& callbacks) {
    const LocationDescriptor location{IR::LocationDescriptor{snapshot.location}};
    const auto first_pc = location.PC();
    snapshot.instructions = {};
    snapshot.successors.clear();
    for (std::size_t i = 0; i < snapshot.instructions.size(); ++i) {
        const auto pc = first_pc + i * 4;
        if (pc < first_pc)
            break;
        snapshot.instructions[i] = callbacks.MemoryReadCode(pc);
        if (!snapshot.instructions[i])
            break;
    }
    const auto captured = [&](std::uint64_t pc) {
        if (pc >= first_pc && (pc - first_pc) / 4 < snapshot.instructions.size())
            return true;
        return std::any_of(snapshot.successors.begin(), snapshot.successors.end(),
            [=](const auto& word) { return word.first == pc; });
    };
    const auto capture_successor = [&](std::uint64_t pc, std::uint32_t instruction) {
        if ((instruction & 0x7c000000u) != 0x14000000u)
            return;
        const auto displacement = static_cast<std::int64_t>(static_cast<std::int32_t>(instruction << 6)) / 16;
        const auto target = location.SetPC(pc + displacement).PC();
        // Leave indirect targets outside the snapshot as dispatch edges. A
        // direct successor needs at most a short block; the whole capture has
        // an additional fixed 64-word budget regardless of branch fan-out.
        for (std::size_t i = 0; i < 32 && snapshot.successors.size() < 64; ++i) {
            const auto address = target + i * 4;
            if (address < target || captured(address))
                break;
            const auto word = callbacks.MemoryReadCode(address);
            if (!word)
                break;
            snapshot.successors.emplace_back(address, *word);
        }
    };
    for (std::size_t i = 0; i < snapshot.instructions.size(); ++i) {
        if (snapshot.instructions[i])
            capture_successor(first_pc + i * 4, *snapshot.instructions[i]);
    }
    for (std::size_t i = 0; i < snapshot.successors.size() && snapshot.successors.size() < 64; ++i) {
        // Copy before capture_successor can reallocate the vector.
        const auto word = snapshot.successors[i];
        capture_successor(word.first, word.second);
    }
    std::sort(snapshot.successors.begin(), snapshot.successors.end());
}

bool SharedCodeCacheAccess::Request(SharedCodeCache& cache, SharedCodeSnapshot job) {
    auto& self = *cache.impl;
    std::unique_lock lock(self.mutex, std::try_to_lock);
    if (!lock.owns_lock() || self.stopping)
        return false;
    if (job.revision != self.revision.load(std::memory_order_acquire)) {
        ++self.statistics.discarded_jobs;
        return false;
    }
    const auto waiting = self.pending.find(job.location);
    if (waiting != self.pending.end() && waiting->second != job.revision)
        return false;
    const auto ready = self.completed.find(job.location);
    if (ready != self.completed.end()) {
        self.PublishIndex(ready->second);
        ++self.statistics.deduplicated_jobs;
        return true;
    }
    if (waiting != self.pending.end()) {
        ++self.statistics.deduplicated_jobs;
        return true;
    }
    if (self.rejected.contains(job.location))
        return false;
    // Keep the speculative backlog to at most one full batch per worker.
    // A large configured maximum is a memory ceiling, not a request to spend
    // milliseconds compiling an obsolete FIFO before current hot work.
    const auto admission_limit = std::min(self.config.max_pending_jobs, self.config.worker_count * std::size_t{32});
    if (self.pending.size() >= admission_limit) {
        ++self.statistics.dropped_jobs;
        return false;
    }
    try {
        self.pending.emplace(job.location, job.revision);
        self.queue.push_back({std::move(job), std::chrono::steady_clock::now()});
    } catch (...) {
        self.pending.erase(job.location);
        ++self.statistics.dropped_jobs;
        return false;
    }
    ++self.statistics.queued_jobs;
    self.statistics.pending_high_water = std::max(self.statistics.pending_high_water,
        static_cast<std::uint64_t>(self.pending.size()));
    lock.unlock();
    self.changed.notify_one();
    return true;
}

} // namespace Dynarmic::A64
