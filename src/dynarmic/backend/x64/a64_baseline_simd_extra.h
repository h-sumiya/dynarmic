// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <mcl/stdint.hpp>

namespace Dynarmic::FP {
class FPCR;
class FPSR;
}

namespace Dynarmic::Backend::X64 {
struct A64JitState;

// Optional AdvSIMD extension families share the enclosing SIMD executor's
// precise exception wrapper. The support predicate validates full encodings.
bool BaselineSimdExtraSupported(u32 instruction);

// Precondition: BaselineSimdExtraSupported(instruction). Source registers are
// snapshotted before any destination write, including all alias combinations.
void ExecuteBaselineSimdExtra(A64JitState* state, u32 instruction,
                             FP::FPCR fpcr, FP::FPSR& fpsr);
}  // namespace Dynarmic::Backend::X64
