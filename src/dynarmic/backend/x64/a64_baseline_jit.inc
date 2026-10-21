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
    }

    static bool OptimizableTerminal(const IR::Terminal& terminal) {
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

    CodePtr OptimizeHotBlock(IR::LocationDescriptor location, CodePtr baseline) {
        using Clock = std::chrono::steady_clock;
        const auto started = conf.enable_cache_statistics ? Clock::now() : Clock::time_point{};
        const auto get_code = [this](u64 pc) { return conf.callbacks->MemoryReadCode(pc); };
        IR::Block ir = A64::Translate(A64::LocationDescriptor{location}, get_code,
                                     {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});
        // The independent baseline also handles instructions which this IR
        // frontend cannot translate. Keep those blocks on their valid path.
        if (!OptimizableTerminal(ir.GetTerminal())) {
            if (conf.enable_cache_statistics)
                cache_statistics.hot_optimization_nanoseconds += static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count());
            return baseline;
        }
        const auto translated = conf.enable_cache_statistics ? Clock::now() : Clock::time_point{};
        Optimization::PolyfillPass(ir, polyfill_options);
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
        const auto optimized = conf.enable_cache_statistics ? Clock::now() : Clock::time_point{};
        const auto bytes_before = block_of_code.SpaceRemaining();
        const auto result = emitter.PromoteBaseline(ir, baseline).entrypoint;
        ++cache_statistics.compiled_blocks;
        ++cache_statistics.optimized_blocks;
        cache_statistics.emitted_bytes += bytes_before - block_of_code.SpaceRemaining();
        cache_statistics.cache_high_water_bytes = std::max(cache_statistics.cache_high_water_bytes,
            static_cast<u64>(block_of_code.GetTotalCodeSize() - block_of_code.SpaceRemaining()));
        if (conf.enable_cache_statistics) {
            const auto published = Clock::now();
            const auto ns = [](auto duration) {
                return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
            };
            cache_statistics.frontend_nanoseconds += ns(translated - started);
            cache_statistics.optimization_nanoseconds += ns(optimized - translated);
            cache_statistics.emission_nanoseconds += ns(published - optimized);
            cache_statistics.hot_optimization_nanoseconds += ns(published - started);
        }
        return result;
    }

    CodePtr GetBlock(IR::LocationDescriptor current_location) {
        bool promote = false;
        const bool sampled = jit_state.baseline_hot_sample != 0;
        jit_state.baseline_hot_sample = 0;
        if (sampled) {
            // Jitter prevents loops whose period matches the sampling interval
            // from repeatedly selecting the same block.
            hot_sample_sequence = hot_sample_sequence * 1664525u + 1013904223u;
            jit_state.baseline_hot_countdown = 64 + (hot_sample_sequence >> 26);
            ++cache_statistics.hot_samples;
        }
        auto cached = emitter.GetBasicBlock(current_location);
        if (cached) {
            if (sampled && conf.enable_baseline_hot_optimization &&
                !A64::LocationDescriptor{current_location}.SingleStepping()) {
                auto& hits = hot_samples[current_location.Value()];
                if (hits != 255 && ++hits >= 2) {
                    hits = 255;
                    promote = true;
                }
            }
            if (!promote)
                return cached->entrypoint;
        }

        constexpr size_t reserve = 1 * 1024 * 1024;
        if (block_of_code.SpaceRemaining() < reserve) {
            ++cache_statistics.capacity_clears;
            invalidate_entire_cache = true;
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
            cached.reset();
        }
        block_of_code.EnsureMemoryCommitted(reserve);
        if (promote && cached)
            return OptimizeHotBlock(current_location, cached->entrypoint);

        hot_samples.erase(current_location.Value());
        using Clock = std::chrono::steady_clock;
        const auto started = conf.enable_cache_statistics ? Clock::now() : Clock::time_point{};
        const auto bytes_before = block_of_code.SpaceRemaining();
        const A64::LocationDescriptor location{current_location};
        const auto get_code = [this](u64 pc) { return conf.callbacks->MemoryReadCode(pc); };
        const auto block = CompileA64Baseline(location.PC(), get_code, conf, 32, location.SingleStepping());
        ASSERT(block.has_value());
        const auto translated = conf.enable_cache_statistics ? Clock::now() : Clock::time_point{};
        const auto result = emitter.EmitBaseline(location, *block).entrypoint;
        ++cache_statistics.compiled_blocks;
        cache_statistics.emitted_bytes += bytes_before - block_of_code.SpaceRemaining();
        cache_statistics.cache_high_water_bytes = std::max(cache_statistics.cache_high_water_bytes,
            static_cast<u64>(block_of_code.GetTotalCodeSize() - block_of_code.SpaceRemaining()));
        if (conf.enable_cache_statistics) {
            const auto published = Clock::now();
            const auto ns = [](auto duration) {
                return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
            };
            cache_statistics.frontend_nanoseconds += ns(translated - started);
            cache_statistics.emission_nanoseconds += ns(published - translated);
            cache_statistics.baseline_nanoseconds += ns(published - started);
        }
        return result;
    }
