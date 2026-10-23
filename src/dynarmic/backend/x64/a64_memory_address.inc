/* SPDX-License-Identifier: 0BSD */

void A64EmitX64::PrepareSharedMemoryAddresses(IR::Block& block) {
    shared_memory_addresses.clear();
    if (!shared_optimized || conf.very_verbose_debugging_output ||
        !conf.optimize_zero_base_fastmem || conf.fastmem_pointer != uintptr_t{0} ||
        conf.fastmem_address_space_bits != 64 || conf.page_table ||
        conf.recompile_on_fastmem_failure || !do_not_fastmem.empty() ||
        !conf.memory_callbacks_preserve_state || !exception_handler.SupportsFastmem())
        return;
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
        if ((access_type != IR::AccType::NORMAL && access_type != IR::AccType::VEC) ||
            memory.GetArg(1).IsImmediate())
            continue;
        auto* address = memory.GetArg(1).GetInst();
        if (address->GetOpcode() != IR::Opcode::Add64 || address->UseCount() != 1 ||
            !address->GetArg(2).IsImmediate() || address->GetArg(2).GetU1())
            continue;
        if (address->GetArg(0).IsImmediate() == address->GetArg(1).IsImmediate())
            continue;
        const auto value = address->GetArg(address->GetArg(0).IsImmediate() ? 0 : 1).GetU64();
        const auto displacement = static_cast<s32>(value);
        if (value != static_cast<u64>(static_cast<s64>(displacement)) || displacement == 0)
            continue;
        // Only this memory operation observes the address. Its base can share
        // the producer's allocation, while the x86 memory operand adds disp32.
        shared_memory_addresses.emplace(address, displacement);
    }
}

bool A64EmitX64::EmitSharedMemoryAddress(A64EmitContext& ctx, IR::Inst* inst) {
    if (!shared_memory_addresses.contains(inst))
        return false;
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.DefineValue(inst, args[args[0].IsImmediate() ? 1 : 0]);
    return true;
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
