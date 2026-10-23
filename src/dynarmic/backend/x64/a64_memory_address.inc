/* SPDX-License-Identifier: 0BSD */

void A64EmitX64::PrepareSharedMemoryAddresses(IR::Block& block) {
    shared_memory_addresses.clear();
    shared_memory_cached_values.clear();
    shared_memory_cached_accesses.clear();
    if (!shared_optimized || conf.very_verbose_debugging_output ||
        !conf.optimize_zero_base_fastmem || conf.fastmem_pointer != uintptr_t{0} ||
        conf.fastmem_address_space_bits != 64 || conf.page_table ||
        conf.recompile_on_fastmem_failure || !do_not_fastmem.empty() ||
        !conf.memory_callbacks_preserve_state || !exception_handler.SupportsFastmem())
        return;
    std::map<const IR::Inst*, std::vector<const IR::Inst*>> cached_consumers;
    std::set<const IR::Inst*> ordinary_memory;
    for (const auto& memory : block) {
        switch (memory.GetOpcode()) {
        case IR::Opcode::A64ReadMemory8:
        case IR::Opcode::A64ReadMemory16:
        case IR::Opcode::A64ReadMemory32:
        case IR::Opcode::A64ReadMemory64:
        case IR::Opcode::A64ReadMemory128:
        case IR::Opcode::A64WriteMemory8:
        case IR::Opcode::A64WriteMemory16:
        case IR::Opcode::A64WriteMemory32:
        case IR::Opcode::A64WriteMemory64:
        case IR::Opcode::A64WriteMemory128:
            break;
        default:
            continue;
        }
        const auto access_type = memory.GetArg(memory.IsMemoryWrite() ? 3 : 2).GetAccType();
        if (access_type != IR::AccType::NORMAL && access_type != IR::AccType::VEC)
            continue;
        ordinary_memory.insert(&memory);
        if (memory.GetArg(1).IsImmediate())
            continue;
        auto* address = memory.GetArg(1).GetInst();
        auto* base = address;
        if (address->GetOpcode() == IR::Opcode::Add64) {
            if (address->UseCount() != 1 || !address->GetArg(2).IsImmediate() ||
                address->GetArg(2).GetU1() ||
                address->GetArg(0).IsImmediate() == address->GetArg(1).IsImmediate())
                continue;
            const auto immediate_index = address->GetArg(0).IsImmediate() ? 0 : 1;
            const auto value = address->GetArg(immediate_index).GetU64();
            const auto displacement = static_cast<s32>(value);
            if (value != static_cast<u64>(static_cast<s64>(displacement)) || displacement == 0)
                continue;
            // Only this memory operation observes the address. Its base can
            // share the producer's allocation while the operand adds disp32.
            shared_memory_addresses.emplace(address, displacement);
            base = address->GetArg(1 - immediate_index).GetInst();
        }
        if ((UseCachedSP() && base->GetOpcode() == IR::Opcode::A64GetSP) ||
            (UseCachedFP() && base->GetOpcode() == IR::Opcode::A64GetX &&
             base->GetArg(0).GetA64RegRef() == A64::Reg::R29))
            cached_consumers[base].push_back(&memory);
    }
    for (const auto& [base, consumers] : cached_consumers) {
        // Each direct access or single-use folded Add consumes this base once.
        // A matching count excludes copies, arithmetic and every other use.
        if (consumers.size() != base->UseCount())
            continue;
        // Ordinary memory callbacks preserve cached registers. The original
        // value remains usable until its architectural register is overwritten.
        const bool is_sp = base->GetOpcode() == IR::Opcode::A64GetSP;
        const HostLoc host = is_sp ? HostLoc::R14 : HostLoc::R13;
        bool found_base = false;
        for (const auto& inst : block) {
            if (&inst == base) {
                found_base = true;
                continue;
            }
            if (&inst == consumers.back())
                break;
            if (!found_base || ordinary_memory.contains(&inst))
                continue;
            const auto opcode = inst.GetOpcode();
            const bool writes_base = is_sp ? opcode == IR::Opcode::A64SetSP :
                ((opcode == IR::Opcode::A64SetW || opcode == IR::Opcode::A64SetX) &&
                 inst.GetArg(0).GetA64RegRef() == A64::Reg::R29);
            if (writes_base) {
                found_base = false;
                break;
            }
            switch (inst.GetOpcode()) {
            case IR::Opcode::A64SetW:
            case IR::Opcode::A64SetX:
            case IR::Opcode::A64SetSP:
            case IR::Opcode::A64SetS:
            case IR::Opcode::A64SetD:
            case IR::Opcode::A64SetQ:
            case IR::Opcode::A64SetNZCV:
            case IR::Opcode::A64SetNZCVRaw:
                continue;
            default:
                break;
            }
            if (inst.MayHaveSideEffects() || inst.IsMemoryReadOrWrite() ||
                inst.CausesCPUException() || inst.GetOpcode() == IR::Opcode::A64GetCNTPCT) {
                found_base = false;
                break;
            }
        }
        if (found_base) {
            shared_memory_cached_values.emplace(base, host);
            for (const auto* memory : consumers)
                shared_memory_cached_accesses.emplace(memory, host);
        }
    }
}

bool A64EmitX64::EmitSharedMemoryAddress(A64EmitContext& ctx, IR::Inst* inst) {
    if (!shared_memory_addresses.contains(inst))
        return false;
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.DefineValue(inst, args[args[0].IsImmediate() ? 1 : 0]);
    return true;
}

Xbyak::Reg64 A64EmitX64::UseSharedMemoryAddress(A64EmitContext& ctx, IR::Inst* inst, Argument& address) {
    if (const auto found = shared_memory_cached_accesses.find(inst); found != shared_memory_cached_accesses.end()) {
        ctx.reg_alloc.Use(address, found->second);
        return HostLocToReg64(found->second);
    }
    return ctx.reg_alloc.UseGpr(address);
}

s32 A64EmitX64::SharedMemoryDisplacement(IR::Inst* inst) const {
    if (inst->GetArg(1).IsImmediate())
        return 0;
    const auto found = shared_memory_addresses.find(inst->GetArg(1).GetInst());
    return found == shared_memory_addresses.end() ? 0 : found->second;
}

const void* A64EmitX64::EmitSharedOffsetMemoryCallback(const void* callback,
    Xbyak::Reg64 base, s32 displacement) {
    if (displacement == 0)
        return callback;
    Xbyak::Label continuation;
    code.jmp(continuation, code.T_NEAR);
    const void* entry = code.getCurr();
    // The base is live at the fault, and may have other SSA aliases. Preserve
    // it around the wrapper while supplying the full architectural address.
    // The push plus nested call retain the wrapper's original stack alignment.
    code.push(base);
    code.lea(base, ptr[base + displacement]);
    code.call(callback);
    code.pop(base);
    code.ret();
    code.L(continuation);
    return entry;
}
