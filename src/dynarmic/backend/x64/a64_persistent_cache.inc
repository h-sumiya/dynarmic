/* SPDX-License-Identifier: 0BSD */

struct MissingPersistentImage final : std::exception {};

std::uint64_t SharedCodeCache::Impl::PersistentContract() const {
    const auto& c = *compile_config;
    // Pointer presence and code-generation choices belong to the contract;
    // process addresses are supplied by relocation or the executing Jit.
    const std::array<std::uint64_t, 29> values{
        c.global_monitor != nullptr, static_cast<u64>(c.optimizations), c.unsafe_optimizations,
        c.hook_data_cache_operations, c.hook_isb, c.hook_hint_instructions,
        c.cntfrq_el0, c.ctr_el0, c.dczid_el0, c.tpidr_el0 != nullptr, c.tpidrro_el0 != nullptr,
        c.page_table != nullptr, c.page_table_address_space_bits, static_cast<u64>(c.page_table_pointer_mask_bits),
        c.silently_mirror_page_table, c.absolute_offset_page_table, c.detect_misaligned_access_via_page_table,
        c.only_detect_misalignment_via_page_table_on_page_boundary, c.fastmem_pointer.has_value(),
        c.fastmem_pointer == uintptr_t{0}, c.optimize_zero_base_fastmem, c.fastmem_address_space_bits,
        c.silently_mirror_fastmem, c.fastmem_exclusive_access, c.define_unpredictable_behaviour,
        c.wall_clock_cntpct, c.check_halt_on_memory_access, c.memory_callbacks_preserve_state,
        c.enable_cycle_counting};
    return Persistent::Hash({reinterpret_cast<const std::uint8_t*>(values.data()), sizeof(values)});
}

Persistent::Reference SharedCodeCache::Impl::GuestReference(Persistent::Region& saved,
        std::unordered_map<std::string, std::uint32_t>& image_indices,
        std::uint64_t value, std::uint64_t origin, bool location, bool pc) const {
    const auto address = location ? LocationDescriptor{IR::LocationDescriptor{value}}.PC() : value;
    const auto image_at = [&](u64 pc) -> const PersistentImage* {
        auto found = std::upper_bound(persistent_image_order.begin(), persistent_image_order.end(), pc,
            [&](u64 address, auto index) { return address < persistent_images[index].address; });
        if (found == persistent_image_order.begin()) return nullptr;
        const auto& image = persistent_images[*std::prev(found)];
        return pc - image.address < image.size ? &image : nullptr;
    };
    // An ADR/ADRP expression follows its origin image's relocation, including
    // addends outside that image. Dispatch PCs identify the destination image.
    const bool destination = location || pc;
    const auto* found = image_at(destination ? address : origin);
    if (!found) found = image_at(destination ? origin : address);
    if (!found) throw std::runtime_error("A64 guest address has no stable image");
    auto identity = image_indices.find(found->identity);
    std::uint32_t index;
    if (identity == image_indices.end()) {
        index = static_cast<std::uint32_t>(saved.images.size());
        saved.images.push_back(found->identity);
        image_indices.emplace(found->identity, index);
    } else index = identity->second;
    return {index, address - found->address,
            location ? value & ~LocationDescriptor::pc_mask : 0};
}

std::optional<Persistent::Region> SharedCodeCache::Impl::SnapshotPersistent(SharedRegion& source) {
    if (source.restored || source.retired || source.building || source.records.empty() ||
        source.saved_records == source.records.size() || !compile_config ||
        compile_config->very_verbose_debugging_output)
        return std::nullopt;
    if (persistent_image_order.size() != persistent_images.size()) {
        persistent_image_order.clear();
        persistent_image_order.reserve(persistent_images.size());
        for (std::size_t i = 0; i < persistent_images.size(); ++i) persistent_image_order.push_back(i);
        std::sort(persistent_image_order.begin(), persistent_image_order.end(), [&](auto left, auto right) {
            return persistent_images[left].address < persistent_images[right].address;
        });
    }
    Persistent::Region saved;
    std::unordered_map<std::string, std::uint32_t> image_indices;
    image_indices.reserve(persistent_images.size());
    saved.runtime_identity = Persistent::RuntimeIdentity();
    saved.contract = PersistentContract();
    saved.features = source.code.HostFeatures();
    saved.capacity = config.region_size;
    saved.prelude = source.code.PreludeSize();
    saved.data_offset = source.code.PersistentDataOffset();
    const auto base = reinterpret_cast<u64>(source.code.getCode());
    const auto body_end = source.code.getSize();
    const auto data = source.code.PersistentData();
    saved.data.assign(data.begin(), data.end());
    saved.body.assign(source.code.getCode() + saved.prelude, source.code.getCode() + body_end);
    std::vector<const BlockOfCode::Relocation*> relocations;
    relocations.reserve(source.code.Relocations().size());
    for (const auto& relocation : source.code.Relocations()) relocations.push_back(&relocation);
    std::stable_sort(relocations.begin(), relocations.end(), [](const auto* left, const auto* right) {
        return left->offset < right->offset;
    });
    const auto host_reference = [&](u64 address) {
        for (std::size_t index = 0; index < saved.hosts.size(); ++index) {
            const auto& known = saved.hosts[index];
            if (address >= known.base && address - known.base < known.size)
                return Persistent::Reference{static_cast<std::uint32_t>(index), address - known.base, 0};
        }
        auto host = Persistent::HostImageAt(address);
        const auto found = std::find_if(saved.hosts.begin(), saved.hosts.end(), [&](const auto& existing) {
            return existing.base == host.base;
        });
        std::uint32_t index;
        if (found == saved.hosts.end()) { index = static_cast<std::uint32_t>(saved.hosts.size()); saved.hosts.push_back(host); }
        else index = static_cast<std::uint32_t>(found - saved.hosts.begin());
        return Persistent::Reference{index, address - host.base, 0};
    };
    for (const auto& record : source.records) {
        const auto published = completed.find(record->location);
        if (published == completed.end() || published->second != record.get() || record->source_words.empty()) continue;
        const auto entry = reinterpret_cast<u64>(record->entrypoint) - base;
        if (!record->host_size || entry < saved.prelude || entry > body_end || record->host_size > body_end - entry) continue;
        const auto image_count = saved.images.size(), host_count = saved.hosts.size();
        const auto relocation_count = saved.relocations.size();
        try {
            Persistent::Block block;
            const auto origin = LocationDescriptor{IR::LocationDescriptor{record->location}}.PC();
            block.location = GuestReference(saved, image_indices, record->location, origin, true);
            block.entry = entry; block.size = record->host_size;
            for (const auto& [pc, word] : record->source_words)
                block.words.push_back({GuestReference(saved, image_indices, pc, pc, false), word});
            auto relocation_it = std::lower_bound(relocations.begin(), relocations.end(), entry,
                [](const auto* relocation, u64 offset) { return relocation->offset < offset; });
            for (; relocation_it != relocations.end() && (*relocation_it)->offset - entry < record->host_size; ++relocation_it) {
                const auto& relocation = **relocation_it;
                Persistent::Relocation item;
                item.offset = relocation.offset; item.kind = static_cast<std::uint8_t>(relocation.kind);
                switch (relocation.kind) {
                case BlockOfCode::RelocationKind::HostAddress:
                    item.target = host_reference(relocation.value); break;
                case BlockOfCode::RelocationKind::GuestAddress:
                case BlockOfCode::RelocationKind::GuestLocation:
                    item.target = GuestReference(saved, image_indices, relocation.value, relocation.origin,
                        relocation.kind == BlockOfCode::RelocationKind::GuestLocation); break;
                case BlockOfCode::RelocationKind::GuestPC:
                    item.target = GuestReference(saved, image_indices, relocation.value, relocation.origin, false, true); break;
                case BlockOfCode::RelocationKind::GlobalMonitor:
                case BlockOfCode::RelocationKind::ExclusiveBuckets:
                    break;
                }
                saved.relocations.push_back(item);
            }
            saved.blocks.push_back(std::move(block));
        } catch (...) {
            for (auto i = image_count; i < saved.images.size(); ++i) image_indices.erase(saved.images[i]);
            saved.images.resize(image_count); saved.hosts.resize(host_count);
            saved.relocations.resize(relocation_count);
        }
    }
    if (saved.blocks.empty()) return std::nullopt;
    std::sort(saved.blocks.begin(), saved.blocks.end(), [](const auto& left, const auto& right) {
        return left.entry < right.entry;
    });
    const auto owner = [&](u64 offset) -> const Persistent::Block* {
        auto block = std::upper_bound(saved.blocks.begin(), saved.blocks.end(), offset,
            [](u64 value, const auto& candidate) { return value < candidate.entry; });
        if (block == saved.blocks.begin()) return nullptr;
        --block;
        return offset - block->entry < block->size ? &*block : nullptr;
    };
    for (const auto& link : source.links) {
        const auto cell = reinterpret_cast<u64>(link.cell) - base;
        // Snapshot under mutex: all mutable cells are copied consistently,
        // then cleared so the file carries no live cross-region pointers.
        if (cell >= saved.data_offset && cell - saved.data_offset + sizeof(u64) <= saved.data.size())
            std::fill_n(saved.data.begin() + (cell - saved.data_offset), sizeof(u64), 0);
        const auto fallback = reinterpret_cast<u64>(link.fallback) - base;
        const auto* block = owner(fallback);
        if (!block) continue;
        const auto& identity = saved.images.at(block->location.image);
        const auto& image = persistent_images.at(persistent_image_index.at(identity));
        const auto origin = image.address + block->location.offset;
        saved.links.push_back({GuestReference(saved, image_indices, link.target, origin, true), cell, fallback});
    }
    for (const auto& fault : source.emitter.PersistentFaults())
        if (owner(fault.rip)) saved.faults.push_back({fault.rip, fault.resume, fault.callback});
    source.saved_records = source.records.size();
    return saved;
}

void SharedCodeCache::Impl::SavePersistent(bool force) {
    if (config.persistent_directory.empty()) return;
    try {
        // A v3 reader retains its opened file snapshot across atomic rename
        // and trimming. Serialization and disk writes never hold the restore
        // mutex, so executing guest threads cannot queue behind a save.
        std::lock_guard writer(persistence_save);
        std::size_t budget;
        {
            std::lock_guard lock(mutex);
            budget = regions.size();
        }
        std::unordered_set<std::uintptr_t> visited;
        visited.reserve(budget);
        bool wrote = false;
        while (budget--) {
            std::optional<Persistent::Region> saved;
            std::string filename;
            bool selected = false;
            {
                std::lock_guard lock(mutex);
                const auto started = std::chrono::steady_clock::now();
                for (auto& region : regions) {
                    if (region->restored || region->retired || region->building || region->records.empty() ||
                        region->saved_records == region->records.size() ||
                        visited.contains(reinterpret_cast<std::uintptr_t>(region.get())) ||
                        (!force && !region->sealed && started - region->last_checkpoint < std::chrono::seconds(5)))
                        continue;
                    visited.insert(reinterpret_cast<std::uintptr_t>(region.get()));
                    selected = true;
                    try {
                        saved = SnapshotPersistent(*region);
                        if (saved) {
                            if (region->save_file.empty()) region->save_file = Persistent::Key(*saved) + ".a64c";
                            filename = region->save_file;
                        }
                    } catch (...) { ++statistics.persistence_failures; }
                    // Unrepresentable records must not prevent reclamation.
                    // A captured snapshot owns every byte before retirement.
                    region->saved_records = region->records.size();
                    region->last_checkpoint = started;
                    break;
                }
                const auto elapsed = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count());
                statistics.snapshot_nanoseconds += elapsed;
                statistics.snapshot_max_nanoseconds = std::max(statistics.snapshot_max_nanoseconds, elapsed);
            }
            if (!selected) break;
            changed.notify_all();
            if (!saved) continue;
            const auto started = std::chrono::steady_clock::now();
            try {
                const auto path = config.persistent_directory / filename;
                Persistent::Write(path, *saved);
                const auto bytes = std::filesystem::file_size(path);
                std::lock_guard lock(mutex);
                ++statistics.persisted_regions; statistics.persistent_bytes += bytes;
                wrote = true;
            } catch (...) {
                std::lock_guard lock(mutex); ++statistics.persistence_failures;
            }
            const auto elapsed = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
            {
                std::lock_guard lock(mutex);
                statistics.persistent_write_nanoseconds += elapsed;
                statistics.persistent_write_max_nanoseconds = std::max(statistics.persistent_write_max_nanoseconds, elapsed);
            }
        }
        if (wrote) {
            const auto started = std::chrono::steady_clock::now();
            Persistent::Trim(config.persistent_directory, config.persistent_capacity_bytes);
            const auto elapsed = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
            std::lock_guard lock(mutex);
            statistics.persistent_write_nanoseconds += elapsed;
            statistics.persistent_write_max_nanoseconds = std::max(statistics.persistent_write_max_nanoseconds, elapsed);
        }
    } catch (...) {
        std::lock_guard lock(mutex); ++statistics.persistence_failures;
    }
}

void SharedCodeCache::Impl::PersistenceWorker() {
    for (;;) {
        std::unique_lock lock(mutex);
        changed.wait_for(lock, std::chrono::milliseconds(250), [&] { return stopping || persistence_requested; });
        if (stopping) return;
        persistence_requested = false;
        lock.unlock();
        SavePersistent(false);
    }
}

void SharedCodeCache::Impl::LoadPersistent() {
    if (config.persistent_directory.empty() || persistent_index_ready.load(std::memory_order_acquire)) return;
    const auto started = std::chrono::steady_clock::now();
    try {
        std::lock_guard io(persistence_io);
        if (persistent_index_ready.load(std::memory_order_relaxed)) return;
        u64 contract;
        {
            std::lock_guard lock(mutex);
            if (!compile_config || stopping || compile_config->very_verbose_debugging_output) return;
            contract = PersistentContract();
        }
        std::vector<PersistentIndexFile> files;
        std::unordered_map<std::string, std::vector<PersistentIndexLocation>> locations;
        u64 index_bytes = 0, failures = 0;
        std::error_code error;
        if (std::filesystem::is_directory(config.persistent_directory, error)) {
            for (const auto& file : std::filesystem::directory_iterator(config.persistent_directory)) {
                if (!file.is_regular_file() || file.path().extension() != ".a64c") continue;
                try {
                    auto index = Persistent::ReadIndex(file.path(), config.region_size * 8 + 16 * 1024 * 1024);
                    index_bytes += index.read_bytes;
                    if (index.runtime_identity != Persistent::RuntimeIdentity() || index.contract != contract ||
                        index.capacity != config.region_size) continue;
                    const auto ordinal = files.size();
                    for (std::size_t block = 0; block < index.blocks.size(); ++block) {
                        const auto& location = index.blocks[block].location;
                        locations[index.images.at(location.image)].push_back({location.offset, location.flags, ordinal, block});
                    }
                    files.push_back({file.path().filename().string(), std::move(index)});
                } catch (...) { ++failures; }
            }
        }
        for (auto& [image, entries] : locations) {
            std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
                return std::tie(left.offset, left.flags, left.file) < std::tie(right.offset, right.flags, right.file);
            });
        }
        persistent_index_files = std::move(files);
        persistent_locations = std::move(locations);
        {
            std::lock_guard lock(mutex);
            statistics.index_files += persistent_index_files.size();
            statistics.index_bytes += index_bytes;
            statistics.persistence_failures += failures;
            statistics.restore_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
        }
        persistent_index_nonempty.store(!persistent_locations.empty(), std::memory_order_release);
        persistent_index_ready.store(true, std::memory_order_release);
    } catch (...) {
        std::lock_guard lock(mutex); ++statistics.persistence_failures;
    }
}

void SharedCodeCache::Impl::RestorePersistentBlock(std::size_t file_ordinal, std::size_t block_ordinal) {
    if (config.persistent_directory.empty()) return;
    const auto started = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        std::lock_guard lock(mutex);
        const auto elapsed = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
        statistics.restore_nanoseconds += elapsed;
        statistics.restore_max_nanoseconds = std::max(statistics.restore_max_nanoseconds, elapsed);
    };
    try {
        std::lock_guard io(persistence_io);
        const auto waited = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
        {
            std::lock_guard lock(mutex);
            statistics.restore_wait_nanoseconds += waited;
            statistics.restore_wait_max_nanoseconds = std::max(statistics.restore_wait_max_nanoseconds, waited);
        }
        const auto& file = persistent_index_files.at(file_ordinal);
        const auto& filename = file.filename;
        const auto& saved = file.saved;
        if (restored_files.contains(filename)) return;
        const auto finish = [&] {
            restored_files.insert(filename);
            persistent_pending.erase(filename);
        };
        std::list<std::unique_ptr<SharedRegion>> created;
        SharedRegion* region = nullptr;
        SharedRegion* pinned = nullptr;
        bool charged = false;
        std::size_t image_count = 0;
        u64 start_revision = 0;
        SCOPE_EXIT {
            created.clear();
            std::lock_guard lock(mutex);
            if (pinned) {
                pinned->building = false;
                if (pinned->retire_requested && !pinned->retired) Retire(*pinned);
            }
            if (charged) {
                statistics.allocated_bytes -= config.region_size;
                changed.notify_all();
            }
        };
        try {
            auto found = persistent_pending.find(filename);
            UserConfig conf;
            bool retired = false;
            {
                std::lock_guard lock(mutex);
                if (!compile_config || stopping || compile_config->very_verbose_debugging_output) return;
                start_revision = revision.load(std::memory_order_acquire);
                image_count = persistent_images.size();
                if (found != persistent_pending.end()) {
                    if (found->second.processed.at(block_ordinal)) return;
                    const auto deferred = found->second.deferred_images.find(block_ordinal);
                    if (deferred != found->second.deferred_images.end() && deferred->second == image_count) return;
                }
                if (found != persistent_pending.end() && found->second.resident) {
                    const auto resident = std::find_if(regions.begin(), regions.end(), [&](const auto& candidate) {
                        return candidate->persistent_file == filename && !candidate->retired && !candidate->retire_requested;
                    });
                    if (resident == regions.end()) retired = true;
                    else {
                        region = resident->get();
                        ASSERT(!region->building);
                        region->building = true; pinned = region;
                    }
                } else {
                    // Admission precedes disk reads and source validation. A
                    // full cache must not repeatedly validate an unused file.
                    if (statistics.allocated_bytes + config.region_size > config.capacity_bytes) return;
                    statistics.allocated_bytes += config.region_size; charged = true;
                }
                conf = *compile_config;
            }
            if (retired) { finish(); return; }
            if (found == persistent_pending.end()) {
                PersistentPendingFile pending_file;
                for (const auto& host : saved.hosts) pending_file.hosts.push_back(Persistent::ResolveHostImage(host));
                pending_file.bindings.resize(saved.images.size());
                pending_file.processed.resize(saved.blocks.size());
                found = persistent_pending.emplace(filename, std::move(pending_file)).first;
            }
            auto& pending_file = found->second;
            {
                std::lock_guard lock(mutex);
                if (pending_file.bound_images != persistent_images.size()) {
                    for (std::size_t i = 0; i < saved.images.size(); ++i) {
                        if (pending_file.bindings[i]) continue;
                        const auto image = persistent_image_index.find(saved.images[i]);
                        if (image != persistent_image_index.end())
                            pending_file.bindings[i] = persistent_images[image->second];
                    }
                    pending_file.bound_images = persistent_images.size();
                }
                image_count = persistent_images.size();
            }
            const auto resolve = [&](const Persistent::Reference& ref, bool location) {
                if (ref.image >= pending_file.bindings.size()) throw std::runtime_error("Invalid persistent A64 image reference");
                const auto& image = pending_file.bindings[ref.image];
                if (!image) throw MissingPersistentImage{};
                const auto address = image->address + ref.offset;
                if (!location && ref.flags) throw std::runtime_error("Invalid persistent A64 address flags");
                return location ? (address & LocationDescriptor::pc_mask) | ref.flags : address;
            };
            auto payload = Persistent::ReadBlock(saved, block_ordinal);
            {
                std::lock_guard lock(mutex);
                statistics.payload_bytes += payload.read_bytes;
            }
            const auto& block = payload.block;
            auto record = std::make_unique<SharedPublishedBlock>();
            record->location = resolve(block.location, true); record->host_size = block.size;
            {
                std::lock_guard lock(mutex);
                if (completed.contains(record->location)) { pending_file.processed[block_ordinal] = true; return; }
            }
            for (const auto& word : block.words) {
                const auto address = resolve(word.address, false);
                const auto& image = *pending_file.bindings.at(word.address.image);
                if (image.size < 4 || address < image.address || address - image.address > image.size - 4 ||
                    image.read_code(address) != word.instruction) {
                    pending_file.processed[block_ordinal] = true;
                    return;
                }
                record->source_words.emplace_back(address, word.instruction);
            }
            std::vector<std::pair<u64, Persistent::Link>> links;
            std::unordered_set<u64> cells;
            for (const auto& link : payload.links) {
                if (!cells.insert(link.cell).second || pending_file.cells.contains(link.cell))
                    throw std::runtime_error("Duplicate persistent A64 link cell");
                links.emplace_back(resolve(link.location, true), link);
            }
            for (const auto& relocation : payload.relocations) {
                u64 value = 0;
                switch (static_cast<BlockOfCode::RelocationKind>(relocation.kind)) {
                case BlockOfCode::RelocationKind::HostAddress: {
                    const auto& host = pending_file.hosts.at(relocation.target.image);
                    if (relocation.target.flags || relocation.target.offset >= host.size)
                        throw std::runtime_error("Invalid persistent A64 helper offset");
                    value = host.base + relocation.target.offset; break;
                }
                case BlockOfCode::RelocationKind::GuestAddress:
                case BlockOfCode::RelocationKind::GuestPC:
                    value = resolve(relocation.target, false); break;
                case BlockOfCode::RelocationKind::GuestLocation:
                    value = resolve(relocation.target, true); break;
                case BlockOfCode::RelocationKind::GlobalMonitor:
                    value = reinterpret_cast<u64>(conf.global_monitor); break;
                case BlockOfCode::RelocationKind::ExclusiveBuckets:
#if A2E_DYNARMIC_SHARDED_MONITOR
                    value = reinterpret_cast<u64>(GetExclusiveMonitorBuckets(conf.global_monitor)); break;
#else
                    throw std::runtime_error("Persistent A64 monitor contract is unavailable");
#endif
                default: throw std::runtime_error("Unknown persistent A64 relocation");
                }
                std::memcpy(payload.body.data() + relocation.offset - block.entry, &value, sizeof(value));
            }
            for (const auto& [pc, word] : record->source_words) {
                if (!record->guest_ranges.empty() && record->guest_ranges.back().last + 1 == pc)
                    record->guest_ranges.back().last = pc + 3;
                else record->guest_ranges.push_back({pc, pc + 3});
            }
            if (record->guest_ranges.size() > SharedPublishedBlock::maximum_guest_ranges)
                throw std::runtime_error("Persistent A64 block has too many ranges");
            record->guest_start = record->guest_ranges.front().first;
            record->guest_end = record->guest_ranges.back().last;
            std::vector<A64EmitX64::PersistentFault> faults;
            faults.reserve(payload.faults.size());
            for (const auto& fault : payload.faults) faults.push_back({fault.rip, fault.resume, fault.callback});
            if (!region) {
                auto data = Persistent::ReadData(saved);
                {
                    std::lock_guard lock(mutex);
                    statistics.payload_bytes += data.size();
                    if (!pending_file.counted_file) {
                        ++statistics.payload_files; pending_file.counted_file = true;
                    }
                }
                created.push_back(std::make_unique<SharedRegion>(conf, config.region_size));
                region = created.front().get();
                if (region->code.HostFeatures() != saved.features || region->code.PreludeSize() != saved.prelude ||
                    region->code.PersistentDataOffset() != saved.data_offset)
                    throw std::runtime_error("Persistent A64 prelude contract differs");
                region->code.InitializePersistentBody(data, saved.body_end);
                region->sealed = true; region->restored = true;
                region->persistent_file = filename;
            }
            region->code.ImportPersistentBlock(block.entry, payload.body);
            const auto base = reinterpret_cast<u64>(region->code.getCode());
            record->entrypoint = reinterpret_cast<const void*>(base + block.entry);
            for (auto& range : record->guest_ranges) range.owner = record.get();
            if (region->records.size() == region->records.capacity())
                region->records.reserve(std::max<std::size_t>(16, region->records.capacity() * 2));
            {
                std::lock_guard lock(mutex);
                if (stopping || start_revision != revision.load(std::memory_order_acquire) || region->retire_requested) return;
                if (!created.empty()) {
                    region->building = true;
                    regions.splice(regions.end(), created); charged = false; pinned = region;
                    pending_file.resident = true;
                    ++statistics.restored_regions;
                }
                try {
                    region->emitter.RestorePersistentFaults(faults);
                    for (const auto& [target, link] : links) {
                        auto* cell = reinterpret_cast<std::atomic<const void*>*>(base + link.cell);
                        const auto* fallback = reinterpret_cast<const void*>(base + link.fallback);
                        std::construct_at(cell, fallback);
                        RegisterIncomingLocked(*region, target, cell, fallback);
                    }
                    const auto* published = record.get();
                    region->records.push_back(std::move(record));
                    if (!completed.contains(published->location)) {
                        for (const auto& range : published->guest_ranges)
                            published_ranges.Add(range.first, range.last, RangeKey(range));
                        completed.emplace(published->location, published);
                        PublishIndex(published); UpdateIncoming(published->location, published->entrypoint);
                        ++statistics.restored_blocks;
                        statistics.live_blocks = completed.size();
                    }
                    region->building = false; pinned = nullptr;
                } catch (...) {
                    for (const auto& existing : region->records)
                        for (const auto& range : existing->guest_ranges)
                            published_ranges.Remove(RangeKey(range));
                    region->retire_requested = true;
                    throw;
                }
            }
            pending_file.processed[block_ordinal] = true;
            pending_file.deferred_images.erase(block_ordinal);
            pending_file.cells.insert(cells.begin(), cells.end());
        } catch (const MissingPersistentImage&) {
            const auto found = persistent_pending.find(filename);
            if (found != persistent_pending.end()) found->second.deferred_images[block_ordinal] = image_count;
        } catch (...) {
            {
                std::lock_guard lock(mutex);
                if (pinned) pinned->retire_requested = true;
                ++statistics.persistence_failures;
            }
            finish();
        }
    } catch (...) {
        std::lock_guard lock(mutex); ++statistics.persistence_failures;
    }
}
