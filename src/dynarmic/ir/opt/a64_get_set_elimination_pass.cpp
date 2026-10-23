/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <array>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/A64/a64_types.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/ir_emitter.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"
#include "dynarmic/ir/value.h"

namespace Dynarmic::Optimization {

void A64GetSetElimination(IR::Block& block, bool preserve_memory_fault_state, bool memory_callbacks_preserve_state) {
    using Iterator = IR::Block::iterator;

    enum class TrackingType {
        W,
        X,
        S,
        D,
        Q,
        SP,
        NZCV,
        NZCVRaw,
    };
    struct RegisterInfo {
        IR::Value register_value;
        TrackingType tracking_type;
        bool set_instruction_present = false;
        bool known_zero_upper = false;
        Iterator last_set_instruction;
    };
    std::array<RegisterInfo, 31> reg_info;
    std::array<RegisterInfo, 32> vec_info;
    RegisterInfo sp_info;
    RegisterInfo nzcv_info;
    IR::IREmitter ir{block};

    const auto same_value = [](IR::Value a, IR::Value b) {
        if (a.IsEmpty() || b.IsEmpty() || a.GetType() != b.GetType())
            return false;
        if (!a.IsImmediate() && !b.IsImmediate())
            return a.GetInstRecursive() == b.GetInstRecursive();
        if (a.IsImmediate() && b.IsImmediate() && (a.GetType() == IR::Type::U32 || a.GetType() == IR::Type::U64))
            return a.GetImmediateAsU64() == b.GetImmediateAsU64();
        return false;
    };
    const auto do_set = [&block, &same_value](RegisterInfo& info, IR::Value value, Iterator set_inst, TrackingType tracking_type) {
        const bool preserves_upper = tracking_type != TrackingType::S && tracking_type != TrackingType::D
            && (tracking_type != TrackingType::W || info.known_zero_upper);
        if (preserves_upper && !info.register_value.IsEmpty()
            && info.tracking_type == tracking_type && same_value(info.register_value, value)) {
            // The architectural value is already visible. Keep any preceding
            // store, including one frozen by a memory observer, and discard
            // only this redundant assignment. DCE removes the invalidated node.
            set_inst->Invalidate();
            return;
        }
        if (info.set_instruction_present) {
            info.last_set_instruction->Invalidate();
            block.Instructions().erase(info.last_set_instruction);
        }

        info.register_value = value;
        info.tracking_type = tracking_type;
        info.set_instruction_present = true;
        info.known_zero_upper = tracking_type == TrackingType::W;
        info.last_set_instruction = set_inst;
    };

    const auto do_get = [&ir](RegisterInfo& info, Iterator get_inst, TrackingType tracking_type) {
        const auto do_nothing = [&] {
            info = {};
            info.register_value = IR::Value(&*get_inst);
            info.tracking_type = tracking_type;
        };

        if (info.register_value.IsEmpty()) {
            do_nothing();
            return;
        }

        if (info.tracking_type == tracking_type) {
            get_inst->ReplaceUsesWith(info.register_value);
            return;
        }

        // Wn and Xn name the same architectural register. Keep its SSA value
        // across mixed-width reads, including W writes which clear the upper
        // half, so integer/address calculations need no state round trip.
        if (info.tracking_type == TrackingType::W && tracking_type == TrackingType::X && info.known_zero_upper) {
            ir.SetInsertionPointBefore(get_inst);
            get_inst->ReplaceUsesWith(ir.ZeroExtendWordToLong(IR::U32{info.register_value}));
            return;
        }
        if (info.tracking_type == TrackingType::X && tracking_type == TrackingType::W) {
            ir.SetInsertionPointBefore(get_inst);
            get_inst->ReplaceUsesWith(ir.LeastSignificantWord(IR::U64{info.register_value}));
            return;
        }

        do_nothing();
    };

    for (auto inst = block.begin(); inst != block.end(); ++inst) {
        // A memory callback may halt at this instruction. Keep every preceding
        // architectural write visible in that case. Register-preserving memory
        // callbacks need no state reload on the normal continuation, so freeze
        // prior stores without forgetting their SSA values or upper-bit facts.
        const auto opcode = inst->GetOpcode();
        const bool trace_guard = opcode == IR::Opcode::A64TraceCheckZero32 ||
            opcode == IR::Opcode::A64TraceCheckZero64 || opcode == IR::Opcode::A64TraceCheckBit;
        const bool observer = inst->IsMemoryReadOrWrite() || inst->CausesCPUException() ||
            opcode == IR::Opcode::CallHostFunction || opcode == IR::Opcode::A64GetCNTPCT ||
            opcode == IR::Opcode::A64DataCacheOperationRaised ||
            opcode == IR::Opcode::A64InstructionCacheOperationRaised ||
            opcode == IR::Opcode::A64InstructionSynchronizationBarrier;
        if (trace_guard || (preserve_memory_fault_state && observer)) {
            // A guard can leave before its continuation executes. Keep its
            // eager state stores while forwarding values on the fallthrough.
            if (trace_guard || (inst->IsMemoryReadOrWrite() && memory_callbacks_preserve_state)) {
                for (auto& info : reg_info)
                    info.set_instruction_present = false;
                for (auto& info : vec_info)
                    info.set_instruction_present = false;
                sp_info.set_instruction_present = false;
                nzcv_info.set_instruction_present = false;
            } else {
                reg_info = {};
                vec_info = {};
                sp_info = {};
                nzcv_info = {};
            }
        }
        switch (inst->GetOpcode()) {
        case IR::Opcode::A64GetW: {
            const size_t index = A64::RegNumber(inst->GetArg(0).GetA64RegRef());
            do_get(reg_info.at(index), inst, TrackingType::W);
            break;
        }
        case IR::Opcode::A64GetX: {
            const size_t index = A64::RegNumber(inst->GetArg(0).GetA64RegRef());
            do_get(reg_info.at(index), inst, TrackingType::X);
            break;
        }
        case IR::Opcode::A64GetS: {
            const size_t index = A64::VecNumber(inst->GetArg(0).GetA64VecRef());
            do_get(vec_info.at(index), inst, TrackingType::S);
            break;
        }
        case IR::Opcode::A64GetD: {
            const size_t index = A64::VecNumber(inst->GetArg(0).GetA64VecRef());
            do_get(vec_info.at(index), inst, TrackingType::D);
            break;
        }
        case IR::Opcode::A64GetQ: {
            const size_t index = A64::VecNumber(inst->GetArg(0).GetA64VecRef());
            do_get(vec_info.at(index), inst, TrackingType::Q);
            break;
        }
        case IR::Opcode::A64GetSP: {
            do_get(sp_info, inst, TrackingType::SP);
            break;
        }
        case IR::Opcode::A64GetNZCVRaw: {
            do_get(nzcv_info, inst, TrackingType::NZCVRaw);
            break;
        }
        case IR::Opcode::A64SetW: {
            const size_t index = A64::RegNumber(inst->GetArg(0).GetA64RegRef());
            do_set(reg_info.at(index), inst->GetArg(1), inst, TrackingType::W);
            break;
        }
        case IR::Opcode::A64SetX: {
            const size_t index = A64::RegNumber(inst->GetArg(0).GetA64RegRef());
            do_set(reg_info.at(index), inst->GetArg(1), inst, TrackingType::X);
            break;
        }
        case IR::Opcode::A64SetS: {
            const size_t index = A64::VecNumber(inst->GetArg(0).GetA64VecRef());
            do_set(vec_info.at(index), inst->GetArg(1), inst, TrackingType::S);
            break;
        }
        case IR::Opcode::A64SetD: {
            const size_t index = A64::VecNumber(inst->GetArg(0).GetA64VecRef());
            do_set(vec_info.at(index), inst->GetArg(1), inst, TrackingType::D);
            break;
        }
        case IR::Opcode::A64SetQ: {
            const size_t index = A64::VecNumber(inst->GetArg(0).GetA64VecRef());
            do_set(vec_info.at(index), inst->GetArg(1), inst, TrackingType::Q);
            break;
        }
        case IR::Opcode::A64SetSP: {
            do_set(sp_info, inst->GetArg(0), inst, TrackingType::SP);
            break;
        }
        case IR::Opcode::A64SetNZCV: {
            do_set(nzcv_info, inst->GetArg(0), inst, TrackingType::NZCV);
            break;
        }
        case IR::Opcode::A64SetNZCVRaw: {
            do_set(nzcv_info, inst->GetArg(0), inst, TrackingType::NZCVRaw);
            break;
        }
        default: {
            if (inst->ReadsFromCPSR() || inst->WritesToCPSR()) {
                nzcv_info = {};
            }
            if (inst->ReadsFromCoreRegister() || inst->WritesToCoreRegister()) {
                reg_info = {};
                vec_info = {};
                sp_info = {};
            }
            break;
        }
        }
    }
}

}  // namespace Dynarmic::Optimization
