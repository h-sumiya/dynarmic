// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <mcl/stdint.hpp>

namespace Dynarmic::Backend::X64 {
struct A64JitState;
struct StackLayout;

bool BaselineSystemSupported(u32 instruction);
bool BaselineSystemTerminal(u32 instruction);
bool ExecuteBaselineSystem(A64JitState* state, u32 instruction, StackLayout* stack);
bool ExecuteBaselineUndefined(A64JitState* state, u32 instruction);
bool ExecuteBaselineNoExecute(A64JitState* state, u32 unused);
}  // namespace Dynarmic::Backend::X64
