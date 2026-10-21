// Direct blocks use a private segmented arena and a thread-owned descriptor
// index. Its immutable dispatcher is outside every reclaimable segment.
#include "dynarmic/backend/x64/a64_baseline_cache.inc"

void A64EmitX64::GenBaselineReturn() {
    code.align();
    baseline_return = code.getCurr<const void*>();
    code.cmp(code.dword[code.r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    if (conf.enable_cycle_counting) {
        code.cmp(code.qword[code.rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
        code.jle(code.GetForceReturnFromRunCodeAddress());
    }
    if (!shared_optimized && conf.shared_code_cache && conf.enable_baseline_hot_optimization) {
        Xbyak::Label shared_index;
        // The reader remains pinned across native edges. C++ lookups refresh
        // its epoch and reject stale private hints before choosing a target.
        code.mov(code.r8, code.qword[code.r15 + offsetof(A64JitState, shared_observed_generation)]);

        code.mov(code.rbx, code.qword[code.r15 + offsetof(A64JitState, pc)]);
        code.mov(code.rcx, A64::LocationDescriptor::pc_mask);
        code.and_(code.rbx, code.rcx);
        code.mov(code.edx, code.dword[code.r15 + offsetof(A64JitState, fpcr)]);
        code.and_(code.edx, A64::LocationDescriptor::fpcr_mask);
        code.shl(code.rdx, A64::LocationDescriptor::fpcr_shift);
        code.or_(code.rbx, code.rdx);
        // Repeated dynamic branches normally hit private data in L1. The
        // owner writes only on a miss; shared dispatch data stays read-only.
        // A generation mismatch rejects the raw pointer before it is loaded.
        EmitSharedDispatchCacheIndex(code.r9, code.rbx, code.rax);
        code.cmp(code.qword[code.r9 + offsetof(A64JitState::SharedDispatchCacheEntry, generation)], code.r8);
        code.jne(shared_index, code.T_NEAR);
        code.cmp(code.qword[code.r9 + offsetof(A64JitState::SharedDispatchCacheEntry, location)], code.rbx);
        code.jne(shared_index, code.T_NEAR);
        code.jmp(code.qword[code.r9 + offsetof(A64JitState::SharedDispatchCacheEntry, entrypoint)]);
        code.L(shared_index);
        // Same hash as SharedCodeHash, with no locks, refcounts or writes.
        code.mov(code.rax, code.rbx);
        code.shr(code.rax, 2);
        code.mov(code.rcx, code.rbx);
        code.shr(code.rcx, 19);
        code.xor_(code.rax, code.rcx);
        code.mov(code.rcx, code.rbx);
        code.shr(code.rcx, 37);
        code.xor_(code.rax, code.rcx);
        code.and_(code.rax, code.qword[code.r15 + offsetof(A64JitState, shared_dispatch_mask)]);
        code.shl(code.rax, 5); // Four pointer slots per read-only set.
        code.mov(code.rdx, code.qword[code.r15 + offsetof(A64JitState, shared_dispatch_table)]);
        code.add(code.rdx, code.rax);
        for (int way = 0; way < 4; ++way) {
            Xbyak::Label next;
            code.mov(code.rax, code.qword[code.rdx + way * 8]);
            code.test(code.rax, code.rax);
            code.jz(next, code.T_SHORT);
            code.cmp(code.qword[code.rax], code.rbx);
            code.jne(next, code.T_SHORT);
            code.mov(code.r10, code.qword[code.rax + 8]);
            code.mov(code.qword[code.r9 + offsetof(A64JitState::SharedDispatchCacheEntry, location)], code.rbx);
            code.mov(code.qword[code.r9 + offsetof(A64JitState::SharedDispatchCacheEntry, entrypoint)], code.r10);
            code.mov(code.qword[code.r9 + offsetof(A64JitState::SharedDispatchCacheEntry, generation)], code.r8);
            code.jmp(code.r10);
            code.L(next);
        }
    }
    code.jmp(terminal_handler_fast_dispatch_hint ? terminal_handler_fast_dispatch_hint
                                               : code.GetReturnFromRunCodeAddress());

    code.align();
    baseline_hot_return = code.getCurr<const void*>();
    code.cmp(code.dword[code.r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    code.LookupBlock();
    code.jmp(code.rax);

    code.align();
    shared_indirect_lookup = code.getCurr<const void*>();
    // Entry tail-jumps into this permanent prelude with RBX naming a private
    // cache slot. No shared return frame remains, so a C++ lookup can advance
    // the reader epoch and allow the source region to be reclaimed.
    code.cmp(code.dword[code.r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    if (conf.enable_cycle_counting) {
        code.cmp(code.qword[code.rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
        code.jle(code.GetForceReturnFromRunCodeAddress());
    }
    // RBX retains the source site's private cache slot; RBP retains the
    // descriptor across any C++ lookup. Both are ABI callee-saved registers.
    code.mov(code.rbp, code.qword[code.r15 + offsetof(A64JitState, pc)]);
    code.mov(code.rcx, A64::LocationDescriptor::pc_mask);
    code.and_(code.rbp, code.rcx);
    code.mov(code.ecx, code.dword[code.r15 + offsetof(A64JitState, fpcr)]);
    code.and_(code.ecx, A64::LocationDescriptor::fpcr_mask);
    code.shl(code.rcx, A64::LocationDescriptor::fpcr_shift);
    code.or_(code.rbp, code.rcx);
    Xbyak::Label lookup, fill_shared;
    code.mov(code.rax, code.rbp);
    code.shr(code.rax, 2);
    code.mov(code.rcx, code.rbp);
    code.shr(code.rcx, 19);
    code.xor_(code.rax, code.rcx);
    code.mov(code.rcx, code.rbp);
    code.shr(code.rcx, 37);
    code.xor_(code.rax, code.rcx);
    code.and_(code.rax, code.qword[code.r15 + offsetof(A64JitState, shared_dispatch_mask)]);
    code.shl(code.rax, 5);
    code.mov(code.rdx, code.qword[code.r15 + offsetof(A64JitState, shared_dispatch_table)]);
    code.add(code.rdx, code.rax);
    for (int way = 0; way < 4; ++way) {
        Xbyak::Label next;
        code.mov(code.rax, code.qword[code.rdx + way * 8]);
        code.test(code.rax, code.rax);
        code.jz(next, code.T_SHORT);
        code.cmp(code.qword[code.rax], code.rbp);
        code.jne(next, code.T_SHORT);
        code.mov(code.rax, code.qword[code.rax + 8]);
        code.jmp(fill_shared, code.T_NEAR);
        code.L(next);
    }
    const auto baseline_slot = [&] {
        code.mov(code.r12, reinterpret_cast<u64>(fast_dispatch_table.get()));
        code.mov(code.rdx, code.rbp);
        if (code.HasHostFeature(HostFeature::SSE42))
            code.crc32(code.rdx, code.r12);
        code.and_(code.edx, fast_dispatch_table_mask);
        code.add(code.rdx, code.r12);
    };
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        // A cold or unsupported target continues directly in its private
        // baseline block. Never make a frequently used baseline target pay a
        // C++ lookup merely because its caller has already been optimized.
        baseline_slot();
        code.cmp(code.qword[code.rdx + offsetof(FastDispatchEntry, location_descriptor)], code.rbp);
        code.jne(lookup, code.T_NEAR);
        code.jmp(code.qword[code.rdx + offsetof(FastDispatchEntry, code_ptr)]);
    }
    code.L(lookup);
    code.LookupBlock();
    code.cmp(code.byte[code.r15 + offsetof(A64JitState, last_lookup_is_shared)], 0);
    code.jne(fill_shared, code.T_NEAR);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        baseline_slot();
        code.mov(code.qword[code.rdx + offsetof(FastDispatchEntry, location_descriptor)], code.rbp);
        code.mov(code.qword[code.rdx + offsetof(FastDispatchEntry, code_ptr)], code.rax);
    }
    code.jmp(code.rax);
    code.L(fill_shared);
    code.mov(code.rdx, code.qword[code.r15 + offsetof(A64JitState, shared_observed_generation)]);
    code.mov(code.qword[code.rbx + offsetof(A64JitState::SharedDispatchCacheEntry, location)], code.rbp);
    code.mov(code.qword[code.rbx + offsetof(A64JitState::SharedDispatchCacheEntry, entrypoint)], code.rax);
    code.mov(code.qword[code.rbx + offsetof(A64JitState::SharedDispatchCacheEntry, generation)], code.rdx);
    code.jmp(code.rax);
}

A64EmitX64::BlockDescriptor A64EmitX64::EmitBaseline(
    A64::LocationDescriptor location, const A64BaselineBlock& block) {
    ASSERT(!shared_optimized && baseline_cache.Initialized());
    code.EnableWriting();
    SCOPE_EXIT { code.DisableWriting(); };
    code.align();
    const u8* entry = code.getCurr();
    // Materialize the selected entry PC before the independent baseline body.
    code.mov(code.rax, location.PC());
    code.mov(code.qword[code.r15 + offsetof(A64JitState, pc)], code.rax);
    const u8* body = code.getCurr();
    code.db(block.code.data(), block.code.size());
    baseline_cache.RecordWithFaults(location.UniqueHash(), reinterpret_cast<std::uintptr_t>(entry),
        code.getCurr<std::uintptr_t>(), block.faults.size(), [&](size_t index) {
            return reinterpret_cast<u64>(body + block.faults[index].fault_offset);
        });
    for (unsigned n = 0; n < block.faults.size(); ++n) {
        const auto& fault = block.faults[n];
        const auto rip = reinterpret_cast<u64>(body + fault.fault_offset);
        // ret_rip=0 resumes at the recovery veneer without a synthetic CALL.
        fastmem_patch_info.emplace(rip,
            FastmemPatchInfo{0, reinterpret_cast<u64>(body + fault.recovery_offset),
                             {location, n}, false});
    }
    block_ranges.AddRange(boost::icl::discrete_interval<u64>::closed(
        location.PC(), location.PC() + block.guest_size - 1), location);
    return RegisterBlock(location, entry, code.getCurr() - entry);
}
