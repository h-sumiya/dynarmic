# A64 execution on x86-64

## Ownership and capacities

Each JIT owns its architectural state, baseline executable segments, baseline
block index and private dispatch hints. Cold blocks use the independent direct
compiler with a 32-instruction limit. Baseline code is never shared with another
JIT or saved to storage. Baseline capacity pressure reclaims one segment and its
metadata at a time; it does not clear the entire cache.

JITs in the same guest address space can attach to one `A64::SharedCodeCache`.
It owns optimizing workers, immutable published code, writable link cells and a
separate executable-region budget. Shared blocks obtain callbacks, TLS, memory
configuration and exclusive reservations from the executing JIT's state.
Attached JITs must agree on all translation-affecting configuration. Unrelated
guest address spaces must use separate shared caches. Optimized code is not
persisted to storage.

Construction reservations and retired regions, including a region being
physically destroyed by a worker, remain charged to the shared budget.
If an old reader prevents collection,
execution continues in baseline code; neither an optimizer nor an execution
thread is allowed to exceed the budget to force a promotion. Do not interpret
`capacity_bytes` as a bound on every allocation: private JIT state, publication
metadata, fixed dispatch indices, queued snapshots and native allocator overhead
are separate from the executable-region budget.

## Admission and asynchronous compilation

A thread-private countdown samples baseline entries and self-edges. Its initial
sample is after 96 visits; subsequent intervals are jittered between 64 and
127 visits. A location must be observed at least twice over at least 50 us
before a snapshot is considered. This avoids queueing an optimizer for every
short-lived entry in a broad code scan. Sampling and retry state belong to the
executing JIT and are removed with its baseline block metadata.

Admission uses a nonblocking scheduler-lock attempt. A busy scheduler, full
queue or lack of reclaimable capacity leaves the baseline running. Successful
or already-pending submissions retry no more often than every 1 ms; failed
admissions back off up to 64 ms. The effective pending limit is the smaller of
the configured limit and 32 requests per worker. The pending and resident-code
maps deduplicate requests even when the fixed native dispatch index has a
collision. These checks and counters are outside the normal native hit path.

The execution thread copies up to 64 contiguous instruction words and at most
64 additional words from bounded direct-successor windows. The worker reads
only this immutable snapshot, never live guest memory or execution-thread
callbacks. It forms a trace of at most 64 executed guest instructions, following
captured unconditional branches, calls and resolved returns. A visited target,
conditional or unresolved branch, system-state transition, snapshot boundary
or instruction budget ends the trace. A missing continuation dispatches to the
correct PC; it does not synthesize an instruction-fetch fault for code that was
merely outside the capture window. Single-step remains one guest instruction.

Workers perform the IR optimization passes and pack blocks into batches of at
most 32 blocks within a 1 ms deadline from the first compilation. An incomplete
instruction page can wait for queued work until that deadline while foreground
execution continues. Complete pages publish without that wait. With W^X support
enabled, instruction pages are sealed
read-execute before publication. Subsequent publications and link updates never
rewrite those instruction bytes. Writable link cells are in the region's
separate data pool.

## Native dispatch

A shared direct edge uses a source-owned atomic link cell. Publishing a target
updates its incoming cells; invalidation or retirement restores their fallback
addresses. The normal edge checks halt state and the optional cycle budget,
then jumps through the cell. Self-edges can jump directly to
the loop body. A source's fallback belongs to that source and tail-jumps to the
executing JIT's permanent prelude, leaving no return address in a retired region.

Dynamic branches use a bounded 256 KiB table of 8192 thread-private hints. Each
site checks two entries, including the complete PC/FPCR descriptor and current
generation. A miss tail-jumps to a permanent native stub that probes the shared
index and the private baseline dispatch table before needing a C++ lookup.
Cold and unsupported targets can therefore continue directly in private code.
The last dynamic PC value may stay in a host register until the branch; a slow
or halt exit materializes it. A block that can change FPCR through a callback or
explicit write uses the runtime FPCR when constructing the target descriptor.

An eight-entry private return stack caches return-link cells. Its descriptor
and generation are checked before any source-cell pointer is dereferenced.
The generic shared dispatch index has four ways. Private hints are only hints:
collisions, invalidation and table eviction always fall back to authoritative
lookup. Native hits require no shared lock, shared reference-count update or
shared hit counter.

If the native index has evicted an otherwise resident target, the C++ miss
path makes a nonblocking attempt to find its existing publication and refresh
the index immediately. A busy scheduler leaves the baseline running; normal
native and index hits do not acquire that lock.

## Optimizing generated code

Bounded traces remove internal branch dispatch and allow constant propagation
and register get/set elimination across the joined instructions. Arithmetic
selection uses x86 signed 32-bit immediates when the full 64-bit value is
representable. The register allocator spills displaced live values to the stack.

A conservative pure self-loop path keeps up to two guest GPRs in host registers.
External entries load them, self-edges retain them, and every exit materializes
them. Eligible final ADD/SUB flag producers can retain packed NZCV and use the
live native condition flags at the terminal. Memory access, callbacks,
architectural observers, incompatible flag reads, verbose tracing and
single-step exclude this path. Cycle-counted blocks use the ordinary condition
materialization path. Other blocks retain the general emitter.

When `check_halt_on_memory_access` is enabled, register get/set elimination
preserves architectural state at memory accesses and other observation points.
Faulting reads do not commit their destination register. Ordinary callback
returns fall through their memory-abort checks; the exceptional PC update and
exit are emitted in a cold tail. The baseline compiler
uses four GPR and four SIMD slots and semantic helpers for complex instructions;
recovery veneers preserve exact guest state at memory faults in both tiers.

## Invalidation and reclamation

Code writes require synchronization and invalidation by the guest memory owner.
The snapshot revision is checked before a worker publishes; invalidation drops
obsolete pending work. A page index records the exact disjoint guest ranges
read for a shared trace. An inlined callee is part of the trace's dependencies;
the unrelated address gap between caller and callee is not. Invalidating any
range removes all of the publication's range records. Batch removal compacts
each affected page bucket once.

Publication removal restores incoming link cells and removes native index
entries before advancing the reclamation generation. Each Run/Step pins its
reader before the first lookup. A C++ lookup reached by tail-leaving shared code
can refresh that epoch: its native return address is in the permanent prelude.
An epoch change clears private shared hints and return-stack entries before
lookup. Callbacks retain their reader protection because they can return into
shared code. Native edges do not reload the global generation. A retired region
and its fault records remain alive until every reader that could have observed
it has reached such a lookup or returned. A long invocation without lookups can
postpone reclamation; capacity pressure continues in baseline code within the
budget. Every Run/Step exit, including exceptions, releases its reader pin.

Shared-cache invalidation removes publications but does not halt attached JITs.
The embedding owner must stop and synchronize affected JITs before changing
guest executable bytes or releasing their mappings. Engine invalidation halts
its JITs and drains their private invalidations at the invocation boundary.

Only optimizer workers physically destroy retired regions. They detach at most
one eligible region under the scheduler mutex, destroy its executable pages and
metadata outside the mutex, and then debit the budget. Invalidation and reader
detachment request this collection without synchronously unmapping the region.
New-region construction also happens outside the scheduler mutex, after its
capacity reservation is charged. A concurrent revision change or shutdown
discards the unexposed region outside the mutex and releases the reservation
only after destruction. Page allocation and permanent-prelude generation do
not hold up foreground invalidation or statistics through the scheduler lock.
Workers poll reader epochs at 1 ms intervals only while there is retirement
debt; an idle cache otherwise uses its condition variable without polling.
Capacity pressure does not retire further useful regions while a previous
victim is still awaiting its grace period or physical destruction.

Engine owners that invalidate the shared cache once for all JITs set
`shared_cache_invalidation_managed`. Other callers use JIT invalidation to
invalidate both tiers. Destroying a shared cache joins its workers; it is a
shutdown operation, not a nonblocking runtime operation.

## Configuration and observation

`A64::UserConfig` controls hot optimization (`enable_baseline_hot_optimization`),
phase timing (`enable_cache_statistics`), private arena and segment sizes
(`code_cache_size`, `baseline_cache_segment_size`), and the shared cache
(`shared_code_cache`). `SharedCodeCacheConfig` controls shared capacity, region
size, worker count, the configured pending limit and native-index capacity.

Per-JIT statistics include baseline compilation and reclamation durations.
Shared statistics include queue wait, pending high water, compilation,
retirement and physical reclamation totals/maxima. These measurements are taken
on slow paths and are intended to distinguish work admission, code generation,
logical retirement and physical memory release. C++ `shared_hits` does not
count native direct links or native dispatch-hint hits; it cannot be used as a
ratio of optimized execution time. Compare useful guest work, wall time and
latency distributions under matched thread counts, warm-up and capacities.

Zero-base fastmem (`optimize_zero_base_fastmem`) uses direct addresses in both
compilers. Other layouts use a base register or memory callbacks.
`GetRegisterData()` exposes the materialized GPR bank to callbacks; access is
restricted to the owning execution thread.

The CMake options `A2E_DYNARMIC_SHARDED_MONITOR`, `A2E_DYNARMIC_PAGE_RANGE_INDEX`
and `A2E_DYNARMIC_LINEAR_IDENTITIES` are enabled by default. Their definitions
are public target properties so consumers use the same interface layout.

## Source layout

- `backend/x64/a64_baseline_*`: direct translation, admission probes and private segments.
- `backend/x64/a64_shared_code_cache.*`: snapshots, workers, publication and reclamation.
- `backend/x64/a64_emit_x64.*`: optimizing emission and shared dispatch.
- `backend/x64/a64_shared_loop.inc`: conservative self-loop register/flag retention.
- `backend/page_range_index.h`: page-based invalidation and batch removal.
- `interface/exclusive_monitor_sharded.h`: bucketed exclusive reservations.
- `interface/A64/cache_statistics.h`: per-JIT cache and compilation statistics.
- `interface/A64/shared_code_cache.h`: shared configuration and statistics.

Build this checkout directly, or link the `dynarmic` target through
`add_subdirectory`. Initialize the Git submodules before configuring.
