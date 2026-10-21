    // Only the owning execution thread changes these fields. Shared machine
    // code reads them through r15, never through the compiling thread's state.
    const void* shared_dispatch = nullptr;
    const void* shared_indirect_lookup = nullptr;
    const std::atomic<u64>* shared_generation = nullptr;
    u64 shared_observed_generation = 0;
    const void* shared_dispatch_table = nullptr;
    u64 shared_dispatch_mask = 0;
    // Private, bounded dispatch hints. The generation is tested before using
    // a code or data pointer, so entries can survive region reclamation
    // without keeping a region alive or requiring a table-wide clear.
    struct SharedDispatchCacheEntry {
        u64 location = 0;
        const void* entrypoint = nullptr;
        u64 generation = 0;
        u64 replacement = 0; // Two-way indirect sites rotate this only on a miss.
    };
    static_assert(sizeof(SharedDispatchCacheEntry) == 32);
    static constexpr size_t SharedDispatchCacheSize = 1024;
    static constexpr size_t SharedDispatchCacheByteMask =
        (SharedDispatchCacheSize - 1) * sizeof(SharedDispatchCacheEntry);
    alignas(64) std::array<SharedDispatchCacheEntry, SharedDispatchCacheSize> shared_dispatch_cache{};
    u32 shared_rsb_ptr = 0;
    std::array<u64, RSBSize> shared_rsb_locations{};
    std::array<const std::atomic<const void*>*, RSBSize> shared_rsb_cells{};
    std::array<u64, RSBSize> shared_rsb_generations{};
    void* runtime_exclusive_reservation = nullptr;
    // A shared result must never be copied into the private dispatch table:
    // that table can outlive the shared region's retirement epoch.
    u8 last_lookup_is_shared = 0;
