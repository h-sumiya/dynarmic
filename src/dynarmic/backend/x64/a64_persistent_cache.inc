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
        std::uint64_t value, std::uint64_t origin, bool location, bool pc) const {
    const auto address = location ? LocationDescriptor{IR::LocationDescriptor{value}}.PC() : value;
    const auto contains = [](const auto& image, auto pc) {
        return pc >= image.address && pc - image.address < image.size;
    };
    // An ADR/ADRP expression follows its origin image's relocation, including
    // addends outside that image. Dispatch PCs identify the destination image.
    const bool destination = location || pc;
    auto found = std::find_if(persistent_images.begin(), persistent_images.end(),
        [&](const auto& image) { return contains(image, destination ? address : origin); });
    if (found == persistent_images.end())
        found = std::find_if(persistent_images.begin(), persistent_images.end(),
            [&](const auto& image) { return contains(image, destination ? origin : address); });
    if (found == persistent_images.end()) throw std::runtime_error("A64 guest address has no stable image");
    auto identity = std::find(saved.images.begin(), saved.images.end(), found->identity);
    if (identity == saved.images.end()) {
        saved.images.push_back(found->identity);
        identity = std::prev(saved.images.end());
    }
    return {static_cast<std::uint32_t>(identity - saved.images.begin()), address - found->address,
            location ? value & ~LocationDescriptor::pc_mask : 0};
}

std::optional<Persistent::Region> SharedCodeCache::Impl::SnapshotPersistent(SharedRegion& source) {
    if (source.restored || source.retired || source.building || source.records.empty() ||
        source.saved_records == source.records.size() || !compile_config ||
        compile_config->very_verbose_debugging_output)
        return std::nullopt;
    Persistent::Region saved;
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
            block.location = GuestReference(saved, record->location, origin, true);
            block.entry = entry; block.size = record->host_size;
            for (const auto& [pc, word] : record->source_words)
                block.words.push_back({GuestReference(saved, pc, pc, false), word});
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
                    item.target = GuestReference(saved, relocation.value, relocation.origin,
                        relocation.kind == BlockOfCode::RelocationKind::GuestLocation); break;
                case BlockOfCode::RelocationKind::GuestPC:
                    item.target = GuestReference(saved, relocation.value, relocation.origin, false, true); break;
                case BlockOfCode::RelocationKind::GlobalMonitor:
                case BlockOfCode::RelocationKind::ExclusiveBuckets:
                    break;
                }
                saved.relocations.push_back(item);
            }
            saved.blocks.push_back(std::move(block));
        } catch (...) {
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
        const auto identity = saved.images.at(block->location.image);
        const auto image = std::find_if(persistent_images.begin(), persistent_images.end(),
            [&](const auto& candidate) { return candidate.identity == identity; });
        const auto origin = image->address + block->location.offset;
        saved.links.push_back({GuestReference(saved, link.target, origin, true), cell, fallback});
    }
    for (const auto& fault : source.emitter.PersistentFaults())
        if (owner(fault.rip)) saved.faults.push_back({fault.rip, fault.resume, fault.callback});
    source.saved_records = source.records.size();
    return saved;
}

void SharedCodeCache::Impl::SavePersistent() {
    if (config.persistent_directory.empty()) return;
    try {
        std::lock_guard io(persistence_io);
        std::vector<Persistent::Region> snapshots;
        {
            std::lock_guard lock(mutex);
            for (auto& region : regions) {
                try {
                    if (auto saved = SnapshotPersistent(*region)) snapshots.push_back(std::move(*saved));
                } catch (...) { ++statistics.persistence_failures; }
            }
        }
        for (const auto& saved : snapshots) {
            try {
                const auto path = config.persistent_directory / (Persistent::Key(saved) + ".a64c");
                Persistent::Write(path, saved);
                const auto bytes = std::filesystem::file_size(path);
                std::lock_guard lock(mutex);
                ++statistics.persisted_regions; statistics.persistent_bytes += bytes;
            } catch (...) {
                std::lock_guard lock(mutex); ++statistics.persistence_failures;
            }
        }
        if (!snapshots.empty()) Persistent::Trim(config.persistent_directory, config.persistent_capacity_bytes);
    } catch (...) {
        std::lock_guard lock(mutex); ++statistics.persistence_failures;
    }
}

void SharedCodeCache::Impl::PersistenceWorker() {
    for (;;) {
        std::unique_lock lock(mutex);
        if (changed.wait_for(lock, std::chrono::milliseconds(250), [&] { return stopping; })) return;
        lock.unlock();
        SavePersistent();
    }
}

void SharedCodeCache::Impl::LoadPersistent() {
    if (config.persistent_directory.empty()) return;
    const auto started = std::chrono::steady_clock::now();
    try {
        std::lock_guard io(persistence_io);
        std::vector<PersistentImage> images;
        UserConfig conf;
        u64 contract, start_revision;
        {
            std::lock_guard lock(mutex);
            if (!compile_config || persistent_images.empty() || stopping || compile_config->very_verbose_debugging_output) return;
            images = persistent_images; conf = *compile_config;
            contract = PersistentContract(); start_revision = revision.load(std::memory_order_acquire);
        }
        std::unordered_map<std::string_view, const PersistentImage*> image_index;
        for (const auto& image : images) image_index.emplace(image.identity, &image);
        const auto finish = [&](const std::string& filename) {
            restored_files.insert(filename);
            const auto found = persistent_pending.find(filename);
            if (found != persistent_pending.end()) {
                persistent_pending_bytes -= found->second.bytes;
                persistent_pending.erase(found);
            }
        };
        std::error_code error;
        std::vector<std::filesystem::path> files;
        for (const auto& [filename, pending_file] : persistent_pending)
            files.push_back(config.persistent_directory / filename);
        if (std::filesystem::is_directory(config.persistent_directory, error)) {
            for (const auto& file : std::filesystem::directory_iterator(config.persistent_directory)) {
                if (file.is_regular_file() && file.path().extension() == ".a64c" &&
                    !persistent_pending.contains(file.path().filename().string())) files.push_back(file.path());
            }
        }
        for (const auto& file : files) {
            const auto filename = file.filename().string();
            if (restored_files.contains(filename)) continue;
            std::list<std::unique_ptr<SharedRegion>> created;
            SharedRegion* pinned = nullptr;
            bool charged = false;
            SCOPE_EXIT {
                created.clear();
                std::lock_guard lock(mutex);
                if (pinned) {
                    pinned->building = false;
                    if (pinned->retire_requested && !pinned->retired) Retire(*pinned);
                }
                if (charged) statistics.allocated_bytes -= config.region_size;
            };
            try {
                auto found = persistent_pending.find(filename);
                if (found == persistent_pending.end()) {
                    const auto bytes = std::filesystem::file_size(file);
                    if (bytes > config.persistent_capacity_bytes - std::min(persistent_pending_bytes, config.persistent_capacity_bytes)) continue;
                    PersistentPendingFile pending_file;
                    auto& saved = pending_file.saved;
                    saved = Persistent::Read(file, config.region_size * 8 + 16 * 1024 * 1024);
                    if (saved.runtime_identity != Persistent::RuntimeIdentity() || saved.contract != contract ||
                        saved.capacity != config.region_size) {
                        finish(filename); continue;
                    }
                    if (saved.prelude > saved.capacity || saved.body.empty() || saved.body.size() > saved.capacity - saved.prelude ||
                        saved.data_offset > saved.prelude || saved.data.size() > saved.prelude - saved.data_offset)
                        throw std::runtime_error("Invalid persistent A64 region bounds");
                    std::sort(saved.blocks.begin(), saved.blocks.end(), [](const auto& left, const auto& right) {
                        return left.entry < right.entry;
                    });
                    u64 end = saved.prelude;
                    for (const auto& block : saved.blocks) {
                        if (!block.size || block.entry < end || block.entry - saved.prelude > saved.body.size() ||
                            block.size > saved.body.size() - (block.entry - saved.prelude) || block.words.empty() || block.words.size() > 4096)
                            throw std::runtime_error("Invalid persistent A64 block bounds");
                        end = block.entry + block.size;
                    }
                    for (const auto& host : saved.hosts) pending_file.hosts.push_back(Persistent::ResolveHostImage(host));
                    std::unordered_set<u64> link_cells;
                    for (const auto& link : saved.links) {
                        if (!link_cells.insert(link.cell).second)
                            throw std::runtime_error("Duplicate persistent A64 link cell");
                    }
                    pending_file.link_index.reserve(saved.links.size());
                    for (std::size_t i = 0; i < saved.links.size(); ++i) pending_file.link_index.push_back(i);
                    std::sort(pending_file.link_index.begin(), pending_file.link_index.end(), [&](auto left, auto right) {
                        return saved.links[left].fallback < saved.links[right].fallback;
                    });
                    std::stable_sort(saved.relocations.begin(), saved.relocations.end(), [](const auto& left, const auto& right) {
                        return left.offset < right.offset;
                    });
                    pending_file.processed.resize(saved.blocks.size());
                    pending_file.bytes = bytes;
                    found = persistent_pending.emplace(filename, std::move(pending_file)).first;
                    persistent_pending_bytes += bytes;
                }
                auto& pending_file = found->second;
                if (pending_file.tried_images == images.size()) continue;
                auto& saved = pending_file.saved;
                std::vector<const PersistentImage*> bindings;
                for (const auto& identity : saved.images) {
                    const auto image = image_index.find(identity);
                    bindings.push_back(image == image_index.end() ? nullptr : image->second);
                }
                const auto resolve = [&](const Persistent::Reference& ref, bool location) {
                    if (ref.image >= bindings.size()) throw std::runtime_error("Invalid persistent A64 image reference");
                    if (!bindings[ref.image]) throw MissingPersistentImage{};
                    const auto address = bindings[ref.image]->address + ref.offset;
                    if (!location && ref.flags) throw std::runtime_error("Invalid persistent A64 address flags");
                    return location ? (address & LocationDescriptor::pc_mask) | ref.flags : address;
                };
                std::vector<std::unique_ptr<SharedPublishedBlock>> records;
                std::vector<std::size_t> ready;
                std::vector<std::pair<u64, u64>> selected;
                std::vector<BlockOfCode::PersistentPatch> patches;
                bool deferred = false;
                for (std::size_t i = 0; i < saved.blocks.size(); ++i) {
                    if (pending_file.processed[i]) continue;
                    const auto& block = saved.blocks[i];
                    const auto patch_count = patches.size();
                    try {
                        auto record = std::make_unique<SharedPublishedBlock>();
                        record->location = resolve(block.location, true); record->host_size = block.size;
                        {
                            std::lock_guard lock(mutex);
                            if (completed.contains(record->location)) { pending_file.processed[i] = true; continue; }
                        }
                        bool matches = true;
                        for (const auto& word : block.words) {
                            const auto address = resolve(word.address, false);
                            const auto& image = *bindings.at(word.address.image);
                            if (image.size < 4 || address < image.address || address - image.address > image.size - 4 ||
                                image.read_code(address) != word.instruction) { matches = false; break; }
                            record->source_words.emplace_back(address, word.instruction);
                        }
                        if (!matches) { pending_file.processed[i] = true; continue; }
                        const auto& link_index = pending_file.link_index;
                        auto link_it = std::lower_bound(link_index.begin(), link_index.end(), block.entry,
                            [&](auto link, u64 offset) { return saved.links[link].fallback < offset; });
                        for (; link_it != link_index.end() && saved.links[*link_it].fallback - block.entry < block.size; ++link_it)
                            (void)resolve(saved.links[*link_it].location, true);
                        auto relocation_it = std::lower_bound(saved.relocations.begin(), saved.relocations.end(), block.entry,
                            [](const auto& relocation, u64 offset) { return relocation.offset < offset; });
                        for (; relocation_it != saved.relocations.end() && relocation_it->offset - block.entry < block.size; ++relocation_it) {
                            const auto& relocation = *relocation_it;
                            if (block.size - (relocation.offset - block.entry) < sizeof(u64))
                                throw std::runtime_error("Invalid persistent A64 relocation bounds");
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
                            patches.push_back({static_cast<std::size_t>(relocation.offset), value});
                        }
                        record->entrypoint = reinterpret_cast<const void*>(block.entry);
                        for (const auto& [pc, word] : record->source_words) {
                            if (!record->guest_ranges.empty() && record->guest_ranges.back().last + 1 == pc)
                                record->guest_ranges.back().last = pc + 3;
                            else record->guest_ranges.push_back({pc, pc + 3});
                        }
                        if (record->guest_ranges.size() > SharedPublishedBlock::maximum_guest_ranges)
                            throw std::runtime_error("Persistent A64 block has too many ranges");
                        record->guest_start = record->guest_ranges.front().first;
                        record->guest_end = record->guest_ranges.back().last;
                        selected.emplace_back(block.entry, block.entry + block.size);
                        ready.push_back(i);
                        records.push_back(std::move(record));
                    } catch (const MissingPersistentImage&) { patches.resize(patch_count); deferred = true; }
                }
                if (records.empty()) {
                    if (!deferred) finish(filename);
                    else pending_file.tried_images = images.size();
                    continue;
                }
                // Validated host bodies do not overlap, and selected preserves
                // their entry order. Only this batch's unpublished bodies match.
                const auto selected_offset = [&](u64 offset) {
                    auto range = std::upper_bound(selected.begin(), selected.end(), offset,
                        [](u64 value, const auto& candidate) { return value < candidate.first; });
                    return range != selected.begin() && offset < std::prev(range)->second;
                };
                std::vector<Persistent::Link> links;
                for (auto link : saved.links) {
                    if (!selected_offset(link.fallback)) continue;
                    if (link.cell < saved.data_offset || link.cell - saved.data_offset > saved.data.size() ||
                        sizeof(u64) > saved.data.size() - (link.cell - saved.data_offset) || link.cell % alignof(std::atomic<const void*>))
                        throw std::runtime_error("Invalid persistent A64 link cell");
                    link.location = {0, resolve(link.location, true), 0};
                    links.push_back(link);
                }
                std::vector<A64EmitX64::PersistentFault> faults;
                const auto code_end = saved.prelude + saved.body.size();
                for (const auto& fault : saved.faults) {
                    if (!selected_offset(fault.rip)) continue;
                    if (fault.resume >= code_end || fault.callback >= code_end || fault.callback < saved.data_offset + saved.data.size())
                        throw std::runtime_error("Invalid persistent A64 fastmem target");
                    faults.push_back({fault.rip, fault.resume, fault.callback});
                }
                SharedRegion* region = nullptr;
                {
                    std::lock_guard lock(mutex);
                    if (stopping || start_revision != revision.load(std::memory_order_acquire)) return;
                    if (pending_file.resident) {
                        const auto resident = std::find_if(regions.begin(), regions.end(), [&](const auto& candidate) {
                            return candidate->persistent_file == filename && !candidate->retired;
                        });
                        if (resident != regions.end() && !(*resident)->retire_requested) {
                            region = resident->get();
                            ASSERT(!region->building);
                            region->building = true; pinned = region;
                        }
                    } else {
                        if (statistics.allocated_bytes + config.region_size > config.capacity_bytes) continue;
                        statistics.allocated_bytes += config.region_size; charged = true;
                    }
                }
                if (pending_file.resident && !region) { finish(filename); continue; }
                if (!region) {
                    created.push_back(std::make_unique<SharedRegion>(conf, config.region_size));
                    region = created.front().get();
                    if (region->code.HostFeatures() != saved.features || region->code.PreludeSize() != saved.prelude ||
                        region->code.PersistentDataOffset() != saved.data_offset)
                        throw std::runtime_error("Persistent A64 prelude contract differs");
                    for (const auto& patch : patches)
                        std::memcpy(saved.body.data() + patch.offset - saved.prelude, &patch.value, sizeof(patch.value));
                    region->code.RestorePersistentBody(saved.data, saved.body);
                    region->code.DisableWriting();
                    region->sealed = true; region->restored = true;
                    region->persistent_file = filename;
                } else {
                    region->code.PatchPersistentCode(patches);
                }
                const auto base = reinterpret_cast<u64>(region->code.getCode());
                for (auto& record : records) {
                    record->entrypoint = reinterpret_cast<const void*>(base + reinterpret_cast<u64>(record->entrypoint));
                    for (auto& range : record->guest_ranges) range.owner = record.get();
                }
                region->records.reserve(region->records.size() + records.size());
                {
                    std::lock_guard lock(mutex);
                    if (stopping || start_revision != revision.load(std::memory_order_acquire) || region->retire_requested) continue;
                    if (!created.empty()) {
                        region->building = true;
                        regions.splice(regions.end(), created); charged = false; pinned = region;
                        pending_file.resident = true;
                        ++statistics.restored_regions;
                    }
                    try {
                        // New metadata precedes publication; existing code,
                        // cells and fault records are never copied or replaced.
                        region->emitter.RestorePersistentFaults(faults);
                        for (const auto& link : links) {
                            auto* cell = reinterpret_cast<std::atomic<const void*>*>(base + link.cell);
                            const auto* fallback = reinterpret_cast<const void*>(base + link.fallback);
                            std::construct_at(cell, fallback);
                            RegisterIncomingLocked(*region, link.location.offset, cell, fallback);
                        }
                        std::uint64_t published_count = 0;
                        for (auto& record : records) {
                            const auto* published = record.get();
                            region->records.push_back(std::move(record));
                            if (completed.contains(published->location)) continue;
                            for (const auto& range : published->guest_ranges)
                                published_ranges.Add(range.first, range.last, RangeKey(range));
                            completed.emplace(published->location, published);
                            PublishIndex(published); UpdateIncoming(published->location, published->entrypoint);
                            ++published_count;
                        }
                        statistics.live_blocks = completed.size();
                        statistics.restored_blocks += published_count;
                        region->building = false; pinned = nullptr;
                    } catch (...) {
                        for (const auto& record : region->records)
                            for (const auto& range : record->guest_ranges)
                                published_ranges.Remove(RangeKey(range));
                        region->retire_requested = true;
                        throw;
                    }
                }
                for (const auto index : ready) pending_file.processed[index] = true;
                if (!deferred) finish(filename);
                else pending_file.tried_images = images.size();
            } catch (...) {
                {
                    std::lock_guard lock(mutex);
                    if (pinned) pinned->retire_requested = true;
                    ++statistics.persistence_failures;
                }
                finish(filename);
            }
        }
    } catch (...) {
        std::lock_guard lock(mutex); ++statistics.persistence_failures;
    }
    {
        std::lock_guard lock(mutex);
        statistics.restore_nanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
    }
    changed.notify_all();
}
