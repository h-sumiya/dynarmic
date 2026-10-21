/* SPDX-License-Identifier: 0BSD */

namespace {

bool SharedLoopTerminal(const IR::Terminal& terminal, IR::LocationDescriptor location, bool& self_edge) {
    if (const auto* link = boost::get<IR::Term::LinkBlock>(&terminal)) {
        self_edge |= link->next == location;
        return true;
    }
    if (const auto* link = boost::get<IR::Term::LinkBlockFast>(&terminal)) {
        self_edge |= link->next == location;
        return true;
    }
    if (const auto* branch = boost::get<IR::Term::If>(&terminal))
        return SharedLoopTerminal(branch->then_, location, self_edge) &&
               SharedLoopTerminal(branch->else_, location, self_edge);
    if (const auto* branch = boost::get<IR::Term::CheckBit>(&terminal))
        return SharedLoopTerminal(branch->then_, location, self_edge) &&
               SharedLoopTerminal(branch->else_, location, self_edge);
    // A CheckHalt's early exit bypasses the normal shared-dispatch helper.
    // Keep that and indirect/callback terminals on the general emitter path.
    return false;
}

bool SharedLoopInstruction(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::A64GetW:
    case IR::Opcode::A64GetX:
    case IR::Opcode::A64SetW:
    case IR::Opcode::A64SetX:
    case IR::Opcode::A64GetCFlag:
    case IR::Opcode::A64GetNZCVRaw:
    case IR::Opcode::A64SetNZCV:
    case IR::Opcode::A64SetNZCVRaw:
    case IR::Opcode::A64SetCheckBit:
        return true;
    default:
        break;
    }
    // A fault, callback, or observer of architectural state would require a
    // materialization point inside the body. Do not delay its state stores.
    if (inst.MayHaveSideEffects() || inst.IsMemoryReadOrWrite() ||
        inst.ReadsFromCoreRegister() || inst.ReadsFromFPCR() || inst.ReadsFromFPSR())
        return false;
    switch (inst.GetOpcode()) {
#define OPCODE(...)
#define A32OPC(name, ...) case IR::Opcode::A32##name:
#define A64OPC(name, ...) case IR::Opcode::A64##name:
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
        return false;
    default:
        return true;
    }
}

bool SharedLoopNativeFlags(IR::Block& block) {
    const auto terminal = block.GetTerminal();
    const auto* branch = boost::get<IR::Term::If>(&terminal);
    if (!branch)
        return false;
    const auto direct = [](const IR::Terminal& terminal) {
        return boost::get<IR::Term::LinkBlock>(&terminal) ||
               boost::get<IR::Term::LinkBlockFast>(&terminal);
    };
    if (!direct(branch->then_) || !direct(branch->else_))
        return false;

    IR::Inst* setter = nullptr;
    for (auto& inst : block) {
        // Delaying the architectural flag store is safe only when no
        // instruction in the body reads it and there is one final writer.
        if (inst.ReadsFromCPSR())
            return false;
        if (inst.WritesToCPSR()) {
            if (setter || inst.GetOpcode() != IR::Opcode::A64SetNZCV)
                return false;
            setter = &inst;
        }
    }
    if (!setter || setter->GetArg(0).IsImmediate())
        return false;
    auto* flags = setter->GetArg(0).GetInst();
    if (flags->GetOpcode() != IR::Opcode::GetNZCVFromOp)
        return false;
    auto* producer = flags->GetArg(0).GetInst();
    switch (producer->GetOpcode()) {
    case IR::Opcode::Add32:
    case IR::Opcode::Add64:
    case IR::Opcode::Sub32:
    case IR::Opcode::Sub64:
        break;
    default:
        return false;
    }
    if (producer->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp) ||
        producer->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp))
        return false;

    bool after_producer = false;
    for (auto& inst : block) {
        if (&inst == producer) {
            after_producer = true;
            continue;
        }
        if (!after_producer || &inst == flags || &inst == setter)
            continue;
        // These stores and their register-allocation moves preserve host
        // flags. Any arithmetic, observer, or other operation falls back to
        // materialized NZCV and the ordinary terminal condition emitter.
        if (inst.GetOpcode() != IR::Opcode::A64SetW && inst.GetOpcode() != IR::Opcode::A64SetX)
            return false;
    }
    return after_producer;
}

} // namespace

void A64EmitX64::PrepareSharedLoop(IR::Block& block, std::vector<HostLoc>& gpr_order) {
    shared_loop_registers.fill(std::nullopt);
    shared_loop_flags = false;
    if (!shared_optimized || conf.very_verbose_debugging_output ||
        A64::LocationDescriptor{block.Location()}.SingleStepping())
        return;
    bool self_edge = false;
    if (!SharedLoopTerminal(block.GetTerminal(), block.Location(), self_edge) || !self_edge)
        return;
    std::array<unsigned, 31> reads{}, writes{};
    for (const auto& inst : block) {
        if (!SharedLoopInstruction(inst))
            return;
        switch (inst.GetOpcode()) {
        case IR::Opcode::A64GetW:
        case IR::Opcode::A64GetX:
            ++reads[A64::RegNumber(inst.GetArg(0).GetA64RegRef())];
            break;
        case IR::Opcode::A64SetW:
        case IR::Opcode::A64SetX:
            ++writes[A64::RegNumber(inst.GetArg(0).GetA64RegRef())];
            break;
        default:
            break;
        }
    }
    // Reserve only two registers, leaving the general instruction selector
    // enough scratch registers even when an operation requires RAX/RCX/RDX.
    // These are not fixed scratch registers in scalar IR emission.
    for (const HostLoc host : {HostLoc::R12, HostLoc::RBP}) {
        unsigned best = 31, best_score = 0;
        for (unsigned guest = 0; guest < reads.size(); ++guest) {
            if (shared_loop_registers[guest] || !reads[guest] || !writes[guest])
                continue;
            const unsigned score = reads[guest] + writes[guest];
            if (score > best_score) {
                best = guest;
                best_score = score;
            }
        }
        if (best == 31)
            break;
        const auto available = std::find(gpr_order.begin(), gpr_order.end(), host);
        if (available == gpr_order.end())
            continue;
        gpr_order.erase(available);
        shared_loop_registers[best] = host;
    }
    // The cycle accounting emitted after the body changes host flags. Keep
    // that configuration on the ordinary condition/materialization path.
    if (!conf.enable_cycle_counting && SharedLoopNativeFlags(block)) {
        const auto available = std::find(gpr_order.begin(), gpr_order.end(), HostLoc::R10);
        if (available != gpr_order.end()) {
            gpr_order.erase(available);
            shared_loop_flags = true;
        }
    }
}

void A64EmitX64::EmitSharedLoopEntry() {
    for (std::size_t guest = 0; guest < shared_loop_registers.size(); ++guest) {
        if (const auto host = shared_loop_registers[guest])
            code.mov(HostLocToReg64(*host), qword[r15 + offsetof(A64JitState, reg) + guest * sizeof(u64)]);
    }
    if (shared_loop_flags)
        code.mov(r10d, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);
    // Published entrypoints remain before the loads. Only the immutable
    // self-edge enters here, carrying the live values from the prior iteration.
    shared_block_entrypoint = code.getCurr();
}

void A64EmitX64::EmitSharedLoopFlush() {
    for (std::size_t guest = 0; guest < shared_loop_registers.size(); ++guest) {
        if (const auto host = shared_loop_registers[guest])
            code.mov(qword[r15 + offsetof(A64JitState, reg) + guest * sizeof(u64)], HostLocToReg64(*host));
    }
    if (shared_loop_flags)
        code.mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], r10d);
}
