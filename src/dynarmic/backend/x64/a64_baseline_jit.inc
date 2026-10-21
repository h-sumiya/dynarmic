    std::unordered_map<u64, u8> hot_samples;
    u32 hot_sample_sequence = 1;

    void InitializeBaselineState() {
        InitializeA64BaselineState(jit_state);
        jit_state.runtime_callbacks = conf.callbacks;
        jit_state.runtime_config = &conf;
        jit_state.runtime_tpidr_el0 = conf.tpidr_el0;
        jit_state.runtime_tpidrro_el0 = conf.tpidrro_el0;
        jit_state.baseline_return = emitter.GetBaselineReturn();
        jit_state.baseline_hot_return = emitter.GetBaselineHotReturn();
        jit_state.shared_dispatch = emitter.GetBaselineReturn();
        if (shared_reader) {
            jit_state.shared_generation = SharedCodeCacheAccess::Generation(*conf.shared_code_cache);
            jit_state.shared_dispatch_table = SharedCodeCacheAccess::DispatchTable(*conf.shared_code_cache);
            jit_state.shared_dispatch_mask = SharedCodeCacheAccess::DispatchMask(*conf.shared_code_cache);
        }
#if A2E_DYNARMIC_SHARDED_MONITOR
        if (conf.global_monitor)
            jit_state.runtime_exclusive_reservation = GetExclusiveMonitorReservation(conf.global_monitor, conf.processor_id);
#endif
    }

    CodePtr GetBlock(IR::LocationDescriptor current_location) {
        // Called before entering native code or by a tail-call out of a body
        // into the permanent private prelude. No native return address can
        // point into a reclaimable region at this safepoint.
        if (shared_reader)
            jit_state.shared_observed_generation = SharedCodeCacheAccess::Pin(*conf.shared_code_cache, *shared_reader);
        jit_state.last_lookup_is_shared = 0;
        const A64::LocationDescriptor location{current_location};
        const bool sampled = jit_state.baseline_hot_sample != 0;
        jit_state.baseline_hot_sample = 0;
        if (sampled) {
            hot_sample_sequence = hot_sample_sequence * 1664525u + 1013904223u;
            jit_state.baseline_hot_countdown = 64 + (hot_sample_sequence >> 26);
            ++cache_statistics.hot_samples;
        }

        if (shared_reader && !location.SingleStepping()) {
            if (const auto optimized = SharedCodeCacheAccess::Lookup(*conf.shared_code_cache, current_location.Value())) {
                jit_state.last_lookup_is_shared = 1;
                ++cache_statistics.shared_cache_hits;
                return optimized;
            }
            if (sampled && conf.enable_baseline_hot_optimization) {
                auto& hits = hot_samples[current_location.Value()];
                if (++hits >= 2) {
                    // Retry periodically: an admitted job can subsequently be
                    // cancelled by invalidation or capacity backpressure.
                    hits = 0;
                    SharedCodeSnapshot snapshot;
                    snapshot.location = current_location.Value();
                    snapshot.revision = SharedCodeCacheAccess::Revision(*conf.shared_code_cache);
                    for (std::size_t i = 0; i < snapshot.instructions.size(); ++i) {
                        const u64 pc = location.PC() + i * 4;
                        if (pc < location.PC())
                            break;
                        snapshot.instructions[i] = conf.callbacks->MemoryReadCode(pc);
                        if (!snapshot.instructions[i])
                            break;
                    }
                    if (SharedCodeCacheAccess::Request(*conf.shared_code_cache, std::move(snapshot)))
                        ++cache_statistics.shared_cache_requests;
                }
            }
        }

        if (const auto cached = emitter.GetBasicBlock(current_location))
            return cached->entrypoint;

        using Clock = std::chrono::steady_clock;
        const auto started = conf.enable_cache_statistics ? Clock::now() : Clock::time_point{};
        const auto get_code = [this](u64 pc) { return conf.callbacks->MemoryReadCode(pc); };
        const auto block = CompileA64Baseline(location.PC(), get_code, conf, 32, location.SingleStepping());
        ASSERT(block.has_value());
        const auto translated = conf.enable_cache_statistics ? Clock::now() : Clock::time_point{};
        const auto reclaimed = emitter.PrepareBaselineEmission(block->code.size());
        if (reclaimed.reclaimed_segments != 0) {
            jit_state.ResetRSB();
            for (const auto key : reclaimed.evicted_locations)
                hot_samples.erase(key);
            cache_statistics.baseline_reclaimed_blocks += reclaimed.reclaimed_blocks;
            cache_statistics.baseline_reclaimed_bytes += reclaimed.reclaimed_bytes;
            cache_statistics.baseline_reclaimed_segments += reclaimed.reclaimed_segments;
        }
        const auto result = emitter.EmitBaseline(location, *block);
        ++cache_statistics.compiled_blocks;
        cache_statistics.emitted_bytes += result.size;
        if (conf.enable_cache_statistics) {
            const auto published = Clock::now();
            const auto ns = [](auto duration) {
                return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
            };
            cache_statistics.frontend_nanoseconds += ns(translated - started);
            cache_statistics.emission_nanoseconds += ns(published - translated);
            cache_statistics.baseline_nanoseconds += ns(published - started);
        }
        return result.entrypoint;
    }
