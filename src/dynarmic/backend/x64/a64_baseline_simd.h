// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <mcl/stdint.hpp>
#include "dynarmic/backend/x64/a64_baseline_simd_extra.h"

namespace Dynarmic::Backend::X64 {
struct A64JitState;

// Both the direct emitter and the semantic veneer use the same allocation
// checks. This is a decoder for instruction families, not an IR frontend.
bool BaselineSimdSupported(u32 instruction);

// Runs one allocated FP/AdvSIMD instruction with architectural state already
// materialized. Ordinary instructions leave PC unchanged. A false result means
// that no instruction completed; the caller must retain the faulting PC.
bool ExecuteBaselineSimd(A64JitState* state, u32 instruction);
}  // namespace Dynarmic::Backend::X64
