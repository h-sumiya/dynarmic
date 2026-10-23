// Deferred architectural state is local to shared optimized blocks. Metadata
// uses after each access keep the pre-access SSA values alive through regalloc.
void A64EmitX64::PrepareDeferredState(IR::Block& block) {
    if (!shared_optimized || !conf.fastmem_pointer || conf.page_table
        || !conf.check_halt_on_memory_access || !conf.memory_callbacks_preserve_state
        || conf.recompile_on_fastmem_failure || !exception_handler.SupportsFastmem())
        return;
    ASSERT(do_not_fastmem.empty());
    struct Pending {
        IR::Opcode opcode = IR::Opcode::Void;
        IR::Value value;
    };
    std::array<Pending, 32> pending{};
    const auto flush = [&](IR::Block::iterator before) {
        for (size_t i = 0; i < pending.size(); ++i) {
            auto& state = pending[i];
            if (state.opcode == IR::Opcode::Void)
                continue;
            if (i == 31)
                block.PrependNewInst(before, state.opcode, {state.value});
            else
                block.PrependNewInst(before, state.opcode, {IR::Value(static_cast<A64::Reg>(i)), state.value});
            state = {};
        }
    };
    for (auto it = block.begin(); it != block.end();) {
        auto current = it++;
        const auto opcode = current->GetOpcode();
        if (opcode == IR::Opcode::A64SetW || opcode == IR::Opcode::A64SetX || opcode == IR::Opcode::A64SetSP) {
            const size_t index = opcode == IR::Opcode::A64SetSP ? 31 : A64::RegNumber(current->GetArg(0).GetA64RegRef());
            pending[index] = {opcode, current->GetArg(opcode == IR::Opcode::A64SetSP ? 0 : 1)};
            current->Invalidate();
            continue;
        }
        bool eligible = false;
        switch (opcode) {
        case IR::Opcode::A64ReadMemory8:
        case IR::Opcode::A64ReadMemory16:
        case IR::Opcode::A64ReadMemory32:
        case IR::Opcode::A64ReadMemory64:
            eligible = current->GetArg(2).GetAccType() == IR::AccType::NORMAL;
            break;
        case IR::Opcode::A64WriteMemory8:
        case IR::Opcode::A64WriteMemory16:
        case IR::Opcode::A64WriteMemory32:
        case IR::Opcode::A64WriteMemory64:
            eligible = current->GetArg(3).GetAccType() == IR::AccType::NORMAL;
            break;
        default:
            break;
        }
        if (eligible) {
            for (size_t i = 0; i < pending.size(); ++i) {
                const auto& state = pending[i];
                if (state.opcode == IR::Opcode::Void)
                    continue;
                const auto marker = state.opcode == IR::Opcode::A64SetW
                    ? IR::Opcode::A64DeferredStateW : IR::Opcode::A64DeferredStateX;
                block.PrependNewInst(it, marker, {IR::Value(static_cast<u8>(i)), state.value});
            }
        } else if (current->IsMemoryReadOrWrite() || current->ReadsFromCoreRegister()
                   || current->MayHaveSideEffects() || opcode == IR::Opcode::A64GetCNTPCT) {
            flush(current);
        }
    }
    flush(block.end());
    Optimization::DeadCodeElimination(block);
}

void A64EmitX64::EmitA64DeferredStateW(A64EmitContext& ctx, IR::Inst* inst) {
    [[maybe_unused]] const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
}

void A64EmitX64::EmitA64DeferredStateX(A64EmitContext& ctx, IR::Inst* inst) {
    [[maybe_unused]] const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
}

std::vector<A64EmitX64::DeferredRegister> A64EmitX64::CaptureDeferredState(A64EmitContext& ctx, IR::Inst* inst) {
    std::vector<DeferredRegister> result;
    for (auto next = std::next(IR::Block::iterator{*inst}); next != ctx.block.end(); ++next) {
        const auto opcode = next->GetOpcode();
        if (opcode != IR::Opcode::A64DeferredStateW && opcode != IR::Opcode::A64DeferredStateX)
            break;
        result.push_back({next->GetArg(0).GetU8(), opcode == IR::Opcode::A64DeferredStateW,
            ctx.reg_alloc.SnapshotValue(next->GetArg(1))});
    }
    return result;
}

std::pair<u64, u64> A64EmitX64::EmitDeferredMemoryFallback(A64EmitContext& ctx, IR::Inst* inst,
    const void* wrapped, Xbyak::Label& end, const std::vector<DeferredRegister>& state) {
    const auto veneer = GenSharedLabel();
    if (state.empty())
        code.call(wrapped);
    else
        code.call(*veneer);
    const auto resume = reinterpret_cast<u64>(code.getCurr());
    EmitCheckMemoryAbort(ctx, inst, &end);
    code.jmp(end, code.T_NEAR);
    if (state.empty())
        return {resume, reinterpret_cast<u64>(wrapped)};

    code.L(*veneer);
    const auto callback = reinterpret_cast<u64>(code.getCurr());
    // Both a real CALL and the fault handler's FakeCall push one return
    // address. Preserve RAX and flags while reconstructing architectural state.
    code.push(rax);
    for (const auto& reg : state) {
        const auto& source = reg.value;
        if (source.immediate) {
            if (reg.word)
                code.mov(eax, static_cast<u32>(source.value));
            else
                code.mov(rax, source.value);
        } else if (HostLocIsGPR(source.location)) {
            if (source.location == HostLoc::RAX) {
                if (reg.word)
                    code.mov(eax, dword[rsp]);
                else
                    code.mov(rax, qword[rsp]);
            } else if (reg.word) {
                code.mov(eax, HostLocToReg64(source.location).cvt32());
            } else {
                code.mov(rax, HostLocToReg64(source.location));
            }
        } else if (HostLocIsXMM(source.location)) {
            if (reg.word)
                code.movd(eax, HostLocToXmm(source.location));
            else
                code.movq(rax, HostLocToXmm(source.location));
        } else {
            ASSERT(HostLocIsSpill(source.location));
            const auto offset = source.spill_offset + 2 * sizeof(u64);
            if (reg.word)
                code.mov(eax, dword[rsp + offset]);
            else
                code.mov(rax, qword[rsp + offset]);
        }
        const auto offset = reg.index == 31 ? offsetof(A64JitState, sp)
            : offsetof(A64JitState, reg) + reg.index * sizeof(u64);
        code.mov(qword[r15 + offset], rax);
    }
    code.mov(rax, A64::LocationDescriptor{IR::LocationDescriptor{inst->GetArg(0).GetU64()}}.PC());
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.pop(rax);
    code.jmp(wrapped);
    return {resume, callback};
}
