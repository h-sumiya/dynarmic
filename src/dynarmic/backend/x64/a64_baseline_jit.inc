    struct HotBlockSamples {
        std::chrono::steady_clock::time_point first_observation{};
        std::chrono::steady_clock::time_point next_request{};
        u32 observations = 0;
        u8 retry_shift = 0;
    };
    std::unordered_map<u64, HotBlockSamples> hot_samples;
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
        jit_state.shared_indirect_lookup = emitter.GetSharedIndirectLookup();
        if (shared_reader) {
            jit_state.shared_dispatch_table = SharedCodeCacheAccess::DispatchTable(*conf.shared_code_cache);
            jit_state.shared_dispatch_mask = SharedCodeCacheAccess::DispatchMask(*conf.shared_code_cache);
        }
#if A2E_DYNARMIC_SHARDED_MONITOR
        if (conf.global_monitor)
            jit_state.runtime_exclusive_reservation = GetExclusiveMonitorReservation(conf.global_monitor, conf.processor_id);
#endif
    }

    CodePtr GetBlock(IR::LocationDescriptor current_location) {
        // Every caller is outside shared native code, with return addresses
        // only in the permanent private prelude. Refresh the epoch here and
        // discard old hints before any new lookup can expose a reclaimed cell.
        RefreshSharedCacheReader();
        jit_state.last_lookup_is_shared = 0;
        const A64::LocationDescriptor location{current_location};
        const bool sampled = jit_state.baseline_hot_sample != 0;
        jit_state.baseline_hot_sample = 0;
        if (sampled) {
            hot_sample_sequence = hot_sample_sequence * 1664525u + 1013904223u;
            // Sparse jittered sampling keeps short loops out of the C++
            // dispatcher and limits callbacks for short-lived code.
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
                using Clock = std::chrono::steady_clock;
                const auto now = Clock::now();
                auto& heat = hot_samples[current_location.Value()];
                if (heat.observations == 0)
                    heat.first_observation = now;
                if (heat.observations != std::numeric_limits<u32>::max())
                    ++heat.observations;
                // Repetition within a microsecond-long one-shot invocation
                // is poor evidence for asynchronous compilation: the worker
                // usually finishes after that code has stopped executing.
                // Require repeated observations spanning time, then retry
                // with backoff if publication/capacity is not ready yet.
                if (heat.observations >= 2 && now >= heat.next_request &&
                    now - heat.first_observation >= std::chrono::microseconds(50)) {
                    SharedCodeSnapshot snapshot;
                    snapshot.location = current_location.Value();
                    snapshot.revision = SharedCodeCacheAccess::Revision(*conf.shared_code_cache);
                    SharedCodeCacheAccess::Capture(snapshot, *conf.callbacks);
                    if (SharedCodeCacheAccess::Request(*conf.shared_code_cache, std::move(snapshot))) {
                        ++cache_statistics.shared_cache_requests;
                        heat.retry_shift = 0;
                    } else if (heat.retry_shift < 6) {
                        ++heat.retry_shift;
                    }
                    // Snapshot guest words only once per retry interval, not
                    // on every sparse sample while the same job is pending.
                    heat.next_request = now + std::chrono::milliseconds(u64{1} << heat.retry_shift);
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
            if (conf.enable_cache_statistics) {
                const auto elapsed = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - translated).count());
                cache_statistics.baseline_reclamation_nanoseconds += elapsed;
                cache_statistics.baseline_reclamation_max_nanoseconds = std::max(
                    cache_statistics.baseline_reclamation_max_nanoseconds, elapsed);
            }
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
