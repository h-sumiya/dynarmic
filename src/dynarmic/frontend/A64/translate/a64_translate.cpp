/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/a64_translate.h"

#include <algorithm>
#include <array>

#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/decoder/a64.h"
#include "dynarmic/frontend/A64/translate/impl/impl.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::A64 {

IR::Block Translate(LocationDescriptor descriptor, MemoryReadCodeFuncType memory_read_code, TranslationOptions options) {
    const bool single_step = descriptor.SingleStepping();

    IR::Block block{descriptor};
    TranslatorVisitor visitor{block, descriptor, std::move(options)};

    bool should_continue = true;
    do {
        const u64 pc = visitor.ir.current_location->PC();

        if (const auto instruction = memory_read_code(pc)) {
            if (auto decoder = Decode<TranslatorVisitor>(*instruction)) {
                should_continue = decoder->get().call(visitor, *instruction);
            } else {
                should_continue = visitor.InterpretThisInstruction();
            }
        } else {
            should_continue = visitor.RaiseException(Exception::NoExecuteFault);
        }

        visitor.ir.current_location = visitor.ir.current_location->AdvancePC(4);
        block.CycleCount()++;
    } while (should_continue && !single_step && block.CycleCount() < 64);

    if (should_continue && (single_step || block.CycleCount() >= 64)) {
        visitor.ir.SetTerm(IR::Term::LinkBlock{*visitor.ir.current_location});
    }

    ASSERT_MSG(block.HasTerminal(), "Terminal has not been set");

    block.SetEndLocation(*visitor.ir.current_location);

    return block;
}

IR::Block TranslateTrace(LocationDescriptor descriptor, MemoryReadCodeFuncType memory_read_code,
                         TranslationOptions options, bool constant_branches, bool conditional_fallthrough) {
    if (descriptor.SingleStepping())
        return Translate(descriptor, std::move(memory_read_code), std::move(options));

    IR::Block block{descriptor};
    TranslatorVisitor visitor{block, descriptor, options};
    std::array<u64, 64> visited{};
    std::size_t visited_count = 0;
    unsigned guard_count = 0;

    while (visited_count < visited.size()) {
        const auto current = *visitor.ir.current_location;
        const u64 pc = current.PC();
        visited[visited_count++] = pc;
        const auto instruction = memory_read_code(pc);
        bool should_continue = false;
        if (instruction) {
            if (auto decoder = Decode<TranslatorVisitor>(*instruction))
                should_continue = decoder->get().call(visitor, *instruction);
            else
                should_continue = visitor.InterpretThisInstruction();
        } else {
            should_continue = visitor.RaiseException(Exception::NoExecuteFault);
        }
        visitor.ir.current_location = current.AdvancePC(4);
        ++block.CycleCount();
        block.SetEndLocation(*visitor.ir.current_location);

        if (should_continue) {
            // A non-contiguous successor may exhaust its capture slice
            // before the trace budget. Missing snapshot bytes are not proof
            // of an architectural execute fault: resume translation at that
            // PC through the normal dispatcher instead of fabricating one.
            if (visited_count == visited.size() || !memory_read_code(visitor.ir.current_location->PC())) {
                visitor.ir.SetTerm(IR::Term::LinkBlock{*visitor.ir.current_location});
                break;
            }
            continue;
        }
        if (!instruction || visited_count == visited.size())
            break;

        // Only architectural branches are merge candidates. Other LinkBlock
        // terminals can represent a system-state change and retain their
        // original dispatch boundary.
        const bool direct = (*instruction & 0x7c000000u) == 0x14000000u;
        const u32 branch_opcode = *instruction & 0xfffffc1fu;
        const bool indirect = branch_opcode == 0xd61f0000u ||
                              branch_opcode == 0xd63f0000u ||
                              branch_opcode == 0xd65f0000u;
        std::optional<LocationDescriptor> next;
        IR::Inst* branch_set_pc = nullptr;
        const auto terminal = block.GetTerminal();
        const auto* branch = boost::get<IR::Term::CheckBit>(&terminal);
        // A guarded continuation can join one more CB/TB fallthrough. Other
        // branches and returns end the trace without following another tail.
        if (guard_count && (!branch || guard_count == 2))
            break;
        bool added_guard = false;
        if (conditional_fallthrough && branch) {
            const auto* yes = boost::get<IR::Term::LinkBlock>(&branch->then_);
            const auto* no = boost::get<IR::Term::LinkBlock>(&branch->else_);
            const auto continuation = current.AdvancePC(4);
            const auto seen = [&](u64 address) {
                return std::find(visited.begin(), visited.begin() + visited_count, address) != visited.begin() + visited_count;
            };
            if (yes && no && (yes->next == continuation || no->next == continuation)) {
                const bool exit_on_true = no->next == continuation;
                const auto exit = LocationDescriptor{exit_on_true ? yes->next : no->next};
                // Both edges must leave the visited prefix. In particular,
                // a forward branch can still return to the trace's entry.
                if (exit.PC() > pc && exit != continuation && !seen(exit.PC()) &&
                    continuation.SetPC(descriptor.PC()) == descriptor &&
                    !seen(continuation.PC()) && memory_read_code(continuation.PC()) &&
                    !block.empty() && block.back().GetOpcode() == IR::Opcode::A64SetCheckBit &&
                    !block.back().GetArg(0).IsImmediate()) {
                    auto* setter = &block.back();
                    auto* predicate = setter->GetArg(0).GetInst();
                    const auto opcode = predicate->GetOpcode();
                    if (opcode == IR::Opcode::IsZero32 || opcode == IR::Opcode::IsZero64) {
                        block.AppendNewInst(opcode == IR::Opcode::IsZero32 ? IR::Opcode::A64TraceCheckZero32 : IR::Opcode::A64TraceCheckZero64,
                            {IR::Value(exit.UniqueHash()), predicate->GetArg(0), IR::Value(exit_on_true)});
                        added_guard = true;
                    } else if (opcode == IR::Opcode::TestBit && predicate->GetArg(1).IsImmediate() &&
                               predicate->GetArg(1).GetU8() < 64) {
                        block.AppendNewInst(IR::Opcode::A64TraceCheckBit,
                            {IR::Value(exit.UniqueHash()), predicate->GetArg(0), predicate->GetArg(1), IR::Value(exit_on_true)});
                        added_guard = true;
                    }
                    if (added_guard) {
                        ++guard_count;
                        setter->Invalidate();
                        block.Instructions().erase(setter);
                        if (!predicate->HasUses()) {
                            predicate->Invalidate();
                            block.Instructions().erase(predicate);
                        }
                        next = continuation;
                    }
                }
            }
        } else if (direct) {
            if (const auto* link = boost::get<IR::Term::LinkBlock>(&terminal))
                next = LocationDescriptor{link->next};
        } else if (indirect && constant_branches) {
            // This is worker-side work. Reuse the validated IR passes instead
            // of a second, incomplete A64 constant evaluator. Memory accesses
            // remain observation barriers even when the final backend does
            // not request precise memory halt state.
            Optimization::A64GetSetElimination(block, true);
            Optimization::ConstantPropagation(block);
            Optimization::DeadCodeElimination(block);
            for (auto it = block.rbegin(); it != block.rend(); ++it) {
                if (it->GetOpcode() != IR::Opcode::A64SetPC)
                    continue;
                if (it->GetArg(0).IsImmediate()) {
                    next = current.SetPC(it->GetArg(0).GetU64());
                    branch_set_pc = &*it;
                }
                break;
            }
        }
        if (!next || next->SetPC(descriptor.PC()) != descriptor || (next->PC() & 3) != 0 ||
            std::find(visited.begin(), visited.begin() + visited_count, next->PC()) != visited.begin() + visited_count ||
            !memory_read_code(next->PC()))
            break;

        if (branch_set_pc) {
            branch_set_pc->Invalidate();
            block.Instructions().erase(branch_set_pc);
        }
        if (branch_opcode == 0xd65f0000u) {
            // A fully inlined return consumes no native RSB entry. Remove its
            // matching inlined call hint, retaining all architectural X30
            // writes and leaving unmatched calls available to the backend.
            for (auto it = block.rbegin(); it != block.rend(); ++it) {
                if (it->GetOpcode() == IR::Opcode::PushRSB &&
                    it->GetArg(0).IsUnsignedImmediate(next->UniqueHash())) {
                    auto* hint = &*it;
                    hint->Invalidate();
                    block.Instructions().erase(hint);
                    break;
                }
            }
        }
        block.ReplaceTerminal(IR::Term::Invalid{});
        visitor.ir.current_location = *next;
    }
    ASSERT_MSG(block.HasTerminal(), "Trace terminal has not been set");
    if (guard_count) {
        const auto revisits = [&](const auto& self, const IR::Terminal& terminal) -> bool {
            const auto seen = [&](IR::LocationDescriptor target) {
                const auto pc = LocationDescriptor{target}.PC();
                return std::find(visited.begin(), visited.begin() + visited_count, pc) != visited.begin() + visited_count;
            };
            if (const auto* link = boost::get<IR::Term::LinkBlock>(&terminal))
                return seen(link->next);
            if (const auto* link = boost::get<IR::Term::LinkBlockFast>(&terminal))
                return seen(link->next);
            if (const auto* branch = boost::get<IR::Term::If>(&terminal))
                return self(self, branch->then_) || self(self, branch->else_);
            if (const auto* branch = boost::get<IR::Term::CheckBit>(&terminal))
                return self(self, branch->then_) || self(self, branch->else_);
            if (const auto* halt = boost::get<IR::Term::CheckHalt>(&terminal))
                return self(self, halt->else_);
            return false;
        };
        // A guard prevents loop-register retention. Preserve cyclic traces
        // instead of turning their backedge into a state-materializing exit.
        if (revisits(revisits, block.GetTerminal()))
            return TranslateTrace(descriptor, std::move(memory_read_code), options, constant_branches, false);
    }
    return block;
}

bool TranslateSingleInstruction(IR::Block& block, LocationDescriptor descriptor, u32 instruction) {
    TranslatorVisitor visitor{block, descriptor, {}};

    bool should_continue = true;
    if (auto decoder = Decode<TranslatorVisitor>(instruction)) {
        should_continue = decoder->get().call(visitor, instruction);
    } else {
        should_continue = visitor.InterpretThisInstruction();
    }

    visitor.ir.current_location = visitor.ir.current_location->AdvancePC(4);
    block.CycleCount()++;

    block.SetEndLocation(*visitor.ir.current_location);

    return should_continue;
}

}  // namespace Dynarmic::A64
