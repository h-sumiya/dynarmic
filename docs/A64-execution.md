# A64 execution on x86-64

Each JIT owns its executable arena, block index, fault records and dispatch
tables. Cold blocks use the independent direct compiler with a 32-instruction
limit. A fixed probe samples baseline entries and self-edges every 64–127
executions. The second sample of a PC/FPCR location invokes the optimizing
frontend on the owning thread, with a 64-instruction limit. Unsupported
optimizing terminals keep their baseline block.

The baseline uses four GPR and four SIMD slots, with precompiled semantic
helpers for complex instructions. Recovery veneers preserve the exact guest
state at memory faults. Code invalidation requests a halt before the owner
changes its cache. Translation code and dispatch caches are thread-local;
exclusive reservations are shared between guest threads.

## Configuration

`A64::UserConfig` controls hot optimization (`enable_baseline_hot_optimization`),
phase timing (`enable_cache_statistics`), zero-base fastmem
(`optimize_zero_base_fastmem`, enabled) and the optional two-entry indirect
branch cache (`inline_dispatch_cache`, disabled). The branch cache requires
FastDispatch and switches a site to global dispatch after 64 misses.

Zero-base fastmem uses direct addresses in both compilers. Other fastmem
layouts retain the base-register path or memory callbacks. `GetRegisterData()`
exposes the materialized GPR bank for callbacks such as integer host leaf calls;
the view is accessed only by the owning execution thread.

The CMake options `A2E_DYNARMIC_SHARDED_MONITOR`, `A2E_DYNARMIC_PAGE_RANGE_INDEX`
and `A2E_DYNARMIC_LINEAR_IDENTITIES` are enabled by default. Their definitions
are public target properties so consumers use the same interface layout.

## Source layout

- `backend/x64/a64_baseline_*`: direct translation, semantic helpers, state and JIT integration.
- `backend/x64/a64_emit_x64.*`: optimizing emission and dispatch.
- `backend/page_range_index.h`: page-based invalidation lookup.
- `interface/exclusive_monitor_sharded.h`: bucketed exclusive reservations.
- `interface/A64/cache_statistics.h`: cache and compilation statistics.

Build this checkout directly, or link the `dynarmic` target through
`add_subdirectory`. Initialize the Git submodules before configuring.
