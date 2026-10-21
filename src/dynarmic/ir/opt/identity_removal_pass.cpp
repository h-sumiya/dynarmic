/* This file is part of the dynarmic project.
 * Copyright (c) 2020 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <array>
#include <vector>

#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"

#ifndef A2E_DYNARMIC_LINEAR_IDENTITIES
#define A2E_DYNARMIC_LINEAR_IDENTITIES 1
#endif

namespace Dynarmic::Optimization {

void IdentityRemovalPass(IR::Block& block) {
#if A2E_DYNARMIC_LINEAR_IDENTITIES
    // Most chains fit entirely on the stack. Retain overflow capacity between
    // arguments for unusual generated blocks, without imposing a chain limit.
    std::array<IR::Inst*, 64> short_path;
    std::vector<IR::Inst*> long_path;
    std::array<IR::Inst*, 256> short_removed;
    size_t short_removed_size = 0;
    std::vector<IR::Inst*> long_removed;

    auto iter = block.begin();
    while (iter != block.end()) {
        auto& inst = *iter;
        const size_t num_args = inst.NumArgs();
        for (size_t i = 0; i < num_args; ++i) {
            auto replacement = inst.GetArg(i);
            if (!replacement.IsIdentity())
                continue;

            // Earlier definitions normally already point directly to their
            // final value. Avoid building any path for this common case.
            const auto next = replacement.GetInst()->GetArg(0);
            if (!next.IsIdentity()) {
                inst.SetArg(i, next);
                continue;
            }

            size_t short_size = 0;
            long_path.clear();
            do {
                auto* identity = replacement.GetInst();
                if (short_size < short_path.size())
                    short_path[short_size++] = identity;
                else
                    long_path.push_back(identity);
                replacement = identity->GetArg(0);
            } while (replacement.IsIdentity());

            // Compress from the tail. SetArg/GetType/IsImmediate themselves
            // follow identity arguments, so head-first compression would still
            // perform quadratic work for a long forward-reference chain.
            const auto compress = [&](IR::Inst* identity) {
                if (identity->GetArg(0).IsIdentity())
                    identity->SetArg(0, replacement);
            };
            for (auto it = long_path.rbegin(); it != long_path.rend(); ++it)
                compress(*it);
            while (short_size)
                compress(short_path[--short_size]);

            inst.SetArg(i, replacement);
        }
        if (inst.GetOpcode() == IR::Opcode::Identity || inst.GetOpcode() == IR::Opcode::Void) {
            iter = block.Instructions().erase(inst);
            if (short_removed_size < short_removed.size())
                short_removed[short_removed_size++] = &inst;
            else
                long_removed.push_back(&inst);
        } else {
            ++iter;
        }
    }

    // Preserve removed identities until later users have been redirected. The
    // normal block needs no heap worklist; large blocks have no artificial cap.
    for (size_t i = 0; i < short_removed_size; ++i)
        short_removed[i]->Invalidate();
    for (auto* inst : long_removed)
        inst->Invalidate();
#else
    std::vector<IR::Inst*> to_invalidate;

    auto iter = block.begin();
    while (iter != block.end()) {
        IR::Inst& inst = *iter;

        const size_t num_args = inst.NumArgs();
        for (size_t i = 0; i < num_args; i++) {
            while (true) {
                IR::Value arg = inst.GetArg(i);
                if (!arg.IsIdentity())
                    break;
                inst.SetArg(i, arg.GetInst()->GetArg(0));
            }
        }

        if (inst.GetOpcode() == IR::Opcode::Identity || inst.GetOpcode() == IR::Opcode::Void) {
            iter = block.Instructions().erase(inst);
            to_invalidate.push_back(&inst);
        } else {
            ++iter;
        }
    }

    for (IR::Inst* inst : to_invalidate) {
        inst->Invalidate();
    }
#endif
}

}  // namespace Dynarmic::Optimization
