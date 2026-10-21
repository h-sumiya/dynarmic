// Direct blocks use the ordinary per-thread code arena and descriptor index.
void A64EmitX64::GenBaselineReturn() {
    code.align();
    baseline_return = code.getCurr<const void*>();
    code.cmp(code.dword[code.r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    if (conf.enable_cycle_counting) {
        code.cmp(code.qword[code.rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
        code.jle(code.GetForceReturnFromRunCodeAddress());
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

A64EmitX64::BlockDescriptor A64EmitX64::PromoteBaseline(IR::Block& block, CodePtr baseline) {
    const tsl::robin_set<IR::LocationDescriptor> locations{block.Location()};
    block_ranges.RemoveLocations(locations);
    EmitX64::InvalidateBasicBlocks(locations);
    const auto result = Emit(block);
    // Only the owning thread changes its code, while executing the prelude.
    // RSB entries which still name this baseline entry follow the new target.
    code.EnableWriting();
    const auto saved = code.getCurr();
    code.SetCodePtr(baseline);
    code.jmp(result.entrypoint, Xbyak::CodeGenerator::T_NEAR);
    code.SetCodePtr(saved);
    code.DisableWriting();
    return result;
}

A64EmitX64::BlockDescriptor A64EmitX64::EmitBaseline(
    A64::LocationDescriptor location, const A64BaselineBlock& block) {
    code.EnableWriting();
    SCOPE_EXIT { code.DisableWriting(); };
    code.align();
    const u8* entry = code.getCurr();
    // Optimized direct edges materialize registers but can omit the PC store.
    // Publish it here before entering the position-independent baseline body.
    code.mov(code.rax, location.PC());
    code.mov(code.qword[code.r15 + offsetof(A64JitState, pc)], code.rax);
    const u8* body = code.getCurr();
    code.db(block.code.data(), block.code.size());
    for (unsigned n = 0; n < block.faults.size(); ++n) {
        const auto& fault = block.faults[n];
        // ret_rip=0 resumes at the recovery veneer without a synthetic CALL.
        fastmem_patch_info.emplace(reinterpret_cast<u64>(body + fault.fault_offset),
            FastmemPatchInfo{0, reinterpret_cast<u64>(body + fault.recovery_offset),
                             {location, n}, false});
    }
    block_ranges.AddRange(boost::icl::discrete_interval<u64>::closed(
        location.PC(), location.PC() + block.guest_size - 1), location);
    return RegisterBlock(location, entry, code.getCurr() - entry);
}
