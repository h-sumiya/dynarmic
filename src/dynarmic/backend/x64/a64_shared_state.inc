    // Only the owning execution thread changes these fields. Shared machine
    // code reads them through r15, never through the compiling thread's state.
    const void* shared_dispatch = nullptr;
    const std::atomic<u64>* shared_generation = nullptr;
    u64 shared_observed_generation = 0;
    const void* shared_dispatch_table = nullptr;
    u64 shared_dispatch_mask = 0;
    void* runtime_exclusive_reservation = nullptr;
    // A shared result must never be copied into the private dispatch table:
    // that table can outlive the shared region's retirement epoch.
    u8 last_lookup_is_shared = 0;
