# A64 execution on x86-64

Each JIT owns its register state, baseline code segments, block index and
dispatch tables. Cold blocks use the independent direct compiler with a
32-instruction limit. A probe samples baseline entries and self-edges every
64–127 executions. Repeated samples enqueue a snapshot of up to 64 guest
instructions for asynchronous optimization.

JITs in the same guest address space use an `A64::SharedCodeCache`. Its bounded
queue deduplicates requests; execution continues in baseline code while workers
compile. Shared blocks read callbacks, TLS and exclusive reservations from the
executing JIT's state. Attached JITs must agree on the translation configuration.
Unrelated guest address spaces must use separate shared caches.

The baseline compiler uses four GPR and four SIMD slots, with precompiled
semantic helpers for complex instructions. Recovery veneers preserve the exact
guest state at memory faults. Baseline capacity pressure reclaims one segment
and its metadata at a time. Shared code uses independent regions and immutable
entries in a four-way dispatch index. Self-edges branch directly; other edges
return through the executing JIT's permanent dispatcher.

Workers pack queued blocks into batches of up to 32 blocks or 1 ms of work,
seal the instruction pages read-execute, and then publish the batch. The bounded
window also packs requests from sparse producers. Published pages are never rewritten. Reader
epochs protect retired code and fault records until execution leaves the shared
region, including callback return paths. Retired regions count against the
capacity limit; insufficient space leaves execution on the baseline path.

## Configuration

`A64::UserConfig` controls hot optimization (`enable_baseline_hot_optimization`),
phase timing (`enable_cache_statistics`), private arena and segment sizes
(`code_cache_size`, `baseline_cache_segment_size`), and the shared cache
(`shared_code_cache`). `SharedCodeCacheConfig` sets shared capacity, region size,
worker count, queue size and dispatch-index capacity.

Zero-base fastmem (`optimize_zero_base_fastmem`) uses direct addresses in both
compilers. Other fastmem layouts use a base register or memory callbacks.
`GetRegisterData()` exposes the materialized GPR bank to callbacks; the view is
accessed only by the owning execution thread.

Code changes require synchronized guest writes and invalidation. Engine owners
that invalidate the shared cache once for all their JITs set
`shared_cache_invalidation_managed`; other callers use JIT invalidation to
invalidate both tiers. Capacity reclamation does not clear the entire cache.

The CMake options `A2E_DYNARMIC_SHARDED_MONITOR`, `A2E_DYNARMIC_PAGE_RANGE_INDEX`
and `A2E_DYNARMIC_LINEAR_IDENTITIES` are enabled by default. Their definitions
are public target properties so consumers use the same interface layout.

## Source layout

- `backend/x64/a64_baseline_*`: direct translation, semantic helpers and private segments.
- `backend/x64/a64_shared_code_cache.*`: workers, publication and region reclamation.
- `backend/x64/a64_emit_x64.*`: optimizing emission and dispatch.
- `backend/page_range_index.h`: page-based invalidation lookup.
- `interface/exclusive_monitor_sharded.h`: bucketed exclusive reservations.
- `interface/A64/cache_statistics.h`: per-JIT cache and compilation statistics.
- `interface/A64/shared_code_cache.h`: shared cache configuration and statistics.

Build this checkout directly, or link the `dynarmic` target through
`add_subdirectory`. Initialize the Git submodules before configuring.
