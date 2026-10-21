// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <mcl/stdint.hpp>
namespace Dynarmic::Backend::X64 {
struct A64JitState;
bool BaselineCryptoSupported(u32 instruction);
bool ExecuteBaselineCrypto(A64JitState* state, u32 instruction);
}  // namespace Dynarmic::Backend::X64
