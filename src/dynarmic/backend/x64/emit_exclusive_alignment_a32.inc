// A2E_EXCLUSIVE_ALIGNMENT_CHECK_V1
void A32EmitX64::EmitExclusiveAlignmentCheck(A32EmitContext& ctx, IR::Inst* inst, Xbyak::Reg64 vaddr, std::size_t bitsize) {
    if (bitsize == 8) {
        return;
    }
    const A32::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};
    [[maybe_unused]] const auto block_location = ctx.Location();
    SharedLabel unaligned = GenSharedLabel();
    code.test(vaddr.cvt32(), static_cast<u32>(bitsize / 8 - 1));
    code.jnz(*unaligned, code.T_NEAR);
    ctx.deferred_emits.emplace_back([=, this] {
        code.L(*unaligned);
        // There is no monitor lock to release: this guard runs before lock.
        code.sub(rsp, 8);  // The ABI save helper expects entry after CALL.
        ABI_PushCallerSaveRegistersAndAdjustStack(code);
        Devirtualize<&A32::UserCallbacks::ExceptionRaised>(conf.callbacks).EmitCall(code, [&](RegList param) {
            code.mov(param[0], current_location.PC());
            code.mov(param[1], static_cast<u64>(A32::Exception::AlignmentFault));
        });
        ABI_PopCallerSaveRegistersAndAdjustStack(code);
        code.add(rsp, 8);
        code.mov(code.byte[r15 + offsetof(A32JitState, exclusive_state)], u8(0));
        code.lock();
        code.or_(dword[r15 + offsetof(A32JitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
        EmitSetUpperLocationDescriptor(current_location, block_location);
        code.mov(dword[r15 + offsetof(A32JitState, Reg) + sizeof(u32) * 15], current_location.PC());
        code.ForceReturnFromRunCode();
    });
}
