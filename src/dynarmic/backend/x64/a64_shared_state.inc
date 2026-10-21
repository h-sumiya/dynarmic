    // Only the owning execution thread changes these fields. Shared machine
    // code reads them through r15, never through the compiling thread's state.
    const void* shared_dispatch = nullptr;
    const void* shared_indirect_lookup = nullptr;
    u64 shared_observed_generation = 0;
    const void* shared_dispatch_table = nullptr;
    u64 shared_dispatch_mask = 0;
    // Private, bounded dispatch hints. Epoch changes at execution-entry or
    // lookup safepoints clear the table; generation checks reject stale code
    // and data pointers before use.
    struct SharedDispatchCacheEntry {
        u64 location = 0;
        const void* entrypoint = nullptr;
        u64 generation = 0;
        u64 replacement = 0; // Two-way indirect sites rotate this only on a miss.
    };
    static_assert(sizeof(SharedDispatchCacheEntry) == 32);
    static constexpr size_t SharedDispatchCacheSize = 8192;
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
    void ResetSharedDispatchHints() {
        shared_dispatch_cache.fill({});
        shared_rsb_ptr = 0;
        shared_rsb_generations.fill(0);
    }
