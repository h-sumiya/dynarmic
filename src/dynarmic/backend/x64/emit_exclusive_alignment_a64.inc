// A2E_EXCLUSIVE_ALIGNMENT_CHECK_V1
void A64EmitX64::EmitExclusiveAlignmentCheck(A64EmitContext& ctx, IR::Inst* inst, Xbyak::Reg64 vaddr, std::size_t bitsize) {
    if (bitsize == 8) {
        return;
    }
    const A64::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};
    [[maybe_unused]] const auto block_location = ctx.Location();
    SharedLabel unaligned = GenSharedLabel();
    code.test(vaddr.cvt32(), static_cast<u32>(bitsize / 8 - 1));
    code.jnz(*unaligned, code.T_NEAR);
    ctx.deferred_emits.emplace_back([=, this] {
        code.L(*unaligned);
        // There is no monitor lock to release: this guard runs before lock.
        code.sub(rsp, 8);  // The ABI save helper expects entry after CALL.
        ABI_PushCallerSaveRegistersAndAdjustStack(code);
        GetUserCallback<&A64::UserCallbacks::ExceptionRaised>().EmitCall(code, [&](RegList param) {
            code.mov(param[0], current_location.PC());
            code.mov(param[1], static_cast<u64>(A64::Exception::AlignmentFault));
        });
        ABI_PopCallerSaveRegistersAndAdjustStack(code);
        code.add(rsp, 8);
        code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
        code.lock();
        code.or_(dword[r15 + offsetof(A64JitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
        code.mov(rax, current_location.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ForceReturnFromRunCode();
    });
}
