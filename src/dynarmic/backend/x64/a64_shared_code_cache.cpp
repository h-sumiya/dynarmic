/* SPDX-License-Identifier: 0BSD */

#include "dynarmic/backend/x64/a64_shared_code_cache.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dynarmic/backend/x64/a64_emit_x64.h"
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
        c.wall_clock_cntpct, c.check_halt_on_memory_access,
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

struct SharedRegion {
    explicit SharedRegion(const UserConfig& conf, std::size_t bytes)
        : code(SharedRunCallbacks(conf), JitStateInfo{layout}, bytes, [](BlockOfCode&) {}, true),
          emitter(code, conf, nullptr, true) {}

    A64JitState layout;
    BlockOfCode code;
    A64EmitX64 emitter;
    std::vector<std::unique_ptr<SharedPublishedBlock>> records;
    bool sealed = false;
    bool building = false;
    bool retired = false;
    bool retire_requested = false;
    std::uint64_t retire_epoch = 0;
};

bool Overlaps(std::uint64_t first, std::uint64_t last, std::uint64_t address, std::size_t length) {
    if (length == 0)
        return false;
    const auto end = address + static_cast<std::uint64_t>(length - 1);
    if (end >= address)
        return first <= end && address <= last;
    return first <= end || address <= last;
}

} // namespace

struct SharedCodeCache::Impl {
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

    void RemovePublished(SharedRegion& region) {
        for (const auto& record : region.records) {
            const auto first = Slot(record->location);
            for (std::size_t i = 0; i < 4; ++i) {
                auto& slot = index[first + i];
                if (slot.load(std::memory_order_relaxed) == record.get())
                    slot.store(nullptr, std::memory_order_release);
            }
            const auto found = completed.find(record->location);
            if (found != completed.end() && found->second == record.get())
                completed.erase(found);
        }
        statistics.live_blocks = completed.size();
    }

    void Retire(SharedRegion& region) {
        ASSERT(region.sealed && !region.building && !region.retired);
        for (auto*& ptr : active)
            if (ptr == &region) ptr = nullptr;
        RemovePublished(region);
        // SC epochs close the reader-publication/collector scan race. Native
        // hot paths only load generation; no shared RMW is emitted there.
        region.retire_epoch = generation.fetch_add(1, std::memory_order_seq_cst) + 1;
        region.retired = true;
        statistics.retired_bytes += config.region_size;
    }

    void Collect() {
        for (auto it = regions.begin(); it != regions.end();) {
            auto& region = **it;
            if (!region.retired) { ++it; continue; }
            const bool protected_reader = std::any_of(readers.begin(), readers.end(), [&](const auto* reader) {
                const auto epoch = reader->epoch.load(std::memory_order_seq_cst);
                return epoch != 0 && epoch < region.retire_epoch;
            });
            if (protected_reader) { ++it; continue; }
            statistics.allocated_bytes -= config.region_size;
            statistics.retired_bytes -= config.region_size;
            statistics.reclaimed_bytes += config.region_size;
            ++statistics.reclaimed_regions;
            it = regions.erase(it);
        }
    }

    SharedRegion* RegionFor(std::size_t worker) {
        constexpr std::size_t emission_reserve = 1024 * 1024;
        auto*& current = active[worker];
        if (current && (current->sealed || current->code.SpaceRemaining() < emission_reserve)) {
            current->sealed = true;
            current = nullptr;
        }
        if (current)
            return current;
        Collect();
        if (statistics.allocated_bytes + config.region_size > config.capacity_bytes) {
            // FIFO region retirement is bounded work. All native links leave
            // through the permanent per-thread dispatcher, so no other region
            // embeds a pointer into this victim.
            const auto victim = std::find_if(regions.begin(), regions.end(), [](const auto& region) {
                return region->sealed && !region->building && !region->retired;
            });
            if (victim != regions.end())
                Retire(**victim);
            Collect();
        }
        if (statistics.allocated_bytes + config.region_size > config.capacity_bytes)
            return nullptr;
        auto region = std::make_unique<SharedRegion>(*compile_config, config.region_size);
        regions.push_back(std::move(region));
        current = regions.back().get();
        statistics.allocated_bytes += config.region_size;
        return current;
    }

    std::unique_ptr<SharedPublishedBlock> Compile(SharedRegion& region, const SharedCodeSnapshot& job) {
        const auto& conf = *compile_config;
        const LocationDescriptor location{IR::LocationDescriptor{job.location}};
        const auto first_pc = location.PC();
        const auto get_code = [&](std::uint64_t pc) -> std::optional<std::uint32_t> {
            if (pc < first_pc || ((pc - first_pc) & 3) != 0)
                return std::nullopt;
            const auto offset = (pc - first_pc) / 4;
            return offset < job.instructions.size() ? job.instructions[offset] : std::nullopt;
        };
        IR::Block ir = Translate(location, get_code,
            {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct, conf.hook_hint_instructions});
        if (OptimizableTerminal(ir.GetTerminal())) {
            Optimization::PolyfillPass(ir, {
                .sha256 = !region.code.HasHostFeature(HostFeature::SHA),
                .vector_multiply_widen = true,
            });
            Optimization::A64CallbackConfigPass(ir, conf);
            Optimization::NamingPass(ir);
            if (conf.HasOptimization(OptimizationFlag::GetSetElimination) && !conf.check_halt_on_memory_access) {
                Optimization::A64GetSetElimination(ir);
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
            const auto end_pc = LocationDescriptor{ir.EndLocation()}.PC();
            return std::make_unique<SharedPublishedBlock>(SharedPublishedBlock{
                job.location, block.entrypoint, first_pc, end_pc - 1});
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
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping)
                    return;
                job = std::move(queue.front());
                queue.pop_front();
                if (job.revision != revision.load(std::memory_order_relaxed)) {
                    pending.erase(job.location);
                    ++statistics.discarded_jobs;
                    continue;
                }
                try {
                    region = RegionFor(worker);
                } catch (...) {
                    pending.erase(job.location);
                    ++statistics.failed_jobs;
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
                // Pack already queued work into the same writable tail. A
                // page is sealed once for the batch, before any entry becomes
                // visible. A bounded 1 ms window also packs sparse producers;
                // execution never waits for this worker's publication.
                if (emission_failed || count == publications.size() ||
                    region->code.SpaceRemaining() < 1024 * 1024 ||
                    std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(1))
                    break;
                std::unique_lock lock(mutex);
                changed.wait_until(lock, started + std::chrono::milliseconds(1), [&] {
                    return stopping || !queue.empty() || region->retire_requested;
                });
                if (stopping || queue.empty() || region->retire_requested)
                    break;
                job = std::move(queue.front());
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
                            region->records.push_back(std::move(result));
                            try {
                                completed.insert_or_assign(publication.location, published);
                            } catch (...) {
                                region->records.pop_back();
                                throw;
                            }
                            // Finalized pages and immutable fault records precede
                            // the release publication of the executable address.
                            PublishIndex(published);
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
                Collect();
            }
        }
    }

    void Invalidate(std::uint64_t address, std::size_t length, bool all) {
        std::lock_guard lock(mutex);
        revision.fetch_add(1, std::memory_order_release);
        // All pending snapshots are conservatively invalidated. Published
        // unrelated blocks survive range invalidation.
        for (const auto& job : queue)
            pending.erase(job.location);
        statistics.discarded_jobs += queue.size();
        queue.clear();
        rejected.clear();
        for (std::size_t i = 0; i < config.max_blocks; ++i) {
            const auto* record = index[i].load(std::memory_order_relaxed);
            if (record && (all || Overlaps(record->guest_start, record->guest_end, address, length))) {
                index[i].store(nullptr, std::memory_order_release);
            }
        }
        for (auto it = completed.begin(); it != completed.end();) {
            const auto* record = it->second;
            if (all || Overlaps(record->guest_start, record->guest_end, address, length))
                it = completed.erase(it);
            else
                ++it;
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
        Collect();
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
    std::deque<SharedCodeSnapshot> queue;
    std::unordered_map<std::uint64_t, std::uint64_t> pending;
    std::unordered_map<std::uint64_t, std::uint64_t> rejected;
    std::list<std::unique_ptr<SharedRegion>> regions;
    std::vector<SharedRegion*> active;
    std::vector<std::uint8_t> replacement;
    // Bounded by resident regions, not by guest address-space history. A native
    // index collision does not cause duplicate compilation of resident code.
    std::unordered_map<std::uint64_t, const SharedPublishedBlock*> completed;
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
    std::lock_guard lock(cache.impl->mutex);
    auto& readers = cache.impl->readers;
    readers.erase(std::remove(readers.begin(), readers.end(), &reader), readers.end());
    cache.impl->Collect();
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
const std::atomic<std::uint64_t>* SharedCodeCacheAccess::Generation(const SharedCodeCache& cache) { return &cache.impl->generation; }
const void* SharedCodeCacheAccess::DispatchTable(const SharedCodeCache& cache) { return cache.impl->index.get(); }
std::uint64_t SharedCodeCacheAccess::DispatchMask(const SharedCodeCache& cache) { return cache.impl->config.max_blocks / 4 - 1; }
std::uint64_t SharedCodeCacheAccess::Revision(const SharedCodeCache& cache) { return cache.impl->revision.load(std::memory_order_acquire); }
const void* SharedCodeCacheAccess::Lookup(const SharedCodeCache& cache, std::uint64_t location) {
    const auto first = cache.impl->Slot(location);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto* block = cache.impl->index[first + i].load(std::memory_order_acquire);
        if (block && block->location == location)
            return block->entrypoint;
    }
    return nullptr;
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
    if (self.pending.size() >= self.config.max_pending_jobs) {
        ++self.statistics.dropped_jobs;
        return false;
    }
    try {
        self.pending.emplace(job.location, job.revision);
        self.queue.push_back(std::move(job));
    } catch (...) {
        self.pending.erase(job.location);
        ++self.statistics.dropped_jobs;
        return false;
    }
    ++self.statistics.queued_jobs;
    lock.unlock();
    self.changed.notify_one();
    return true;
}

} // namespace Dynarmic::A64
