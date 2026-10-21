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
        Xbyak::Label local, refresh;
        // A generation change is handled only here, after tail-leaving the
        // shared body. Refreshing an epoch from a callback would be unsafe:
        // its native return address may still name the shared region.
        code.mov(code.rax, code.qword[code.r15 + offsetof(A64JitState, shared_generation)]);
        code.mov(code.rdx, code.qword[code.rax]);
        code.cmp(code.rdx, code.qword[code.r15 + offsetof(A64JitState, shared_observed_generation)]);
        code.jne(refresh, code.T_NEAR);

        code.mov(code.rbx, code.qword[code.r15 + offsetof(A64JitState, pc)]);
        code.mov(code.rcx, A64::LocationDescriptor::pc_mask);
        code.and_(code.rbx, code.rcx);
        code.mov(code.edx, code.dword[code.r15 + offsetof(A64JitState, fpcr)]);
        code.and_(code.edx, A64::LocationDescriptor::fpcr_mask);
        code.shl(code.rdx, A64::LocationDescriptor::fpcr_shift);
        code.or_(code.rbx, code.rdx);
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
            code.jmp(code.qword[code.rax + 8]);
            code.L(next);
        }
        code.jmp(local, code.T_NEAR);
        code.L(refresh);
        code.LookupBlock();
        code.jmp(code.rax);
        code.L(local);
    }
    code.jmp(terminal_handler_fast_dispatch_hint ? terminal_handler_fast_dispatch_hint
                                               : code.GetReturnFromRunCodeAddress());

    code.align();
    baseline_hot_return = code.getCurr<const void*>();
    code.cmp(code.dword[code.r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    code.LookupBlock();
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
