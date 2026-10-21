// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <mcl/stdint.hpp>

namespace Dynarmic::Backend::X64 {

struct A64JitState;

// A bounded, allocation-free decode shared by the native templates and the
// uncommon instruction helpers. There is no Dynarmic frontend/IR dependency.
struct BaselineMemoryOp {
    enum class Kind : u8 {
        Invalid, Simd, SimdPair, SimdLiteral, StructureMultiple,
        StructureSingle, Exclusive, Ordered, AtomicCAS, AtomicCASP, AtomicRmw,
    } kind = Kind::Invalid;
    enum class Address : u8 {
        Base, Offset, PostIndex, PreIndex, RegisterOffset, Literal,
    } address = Address::Base;
    u8 bytes = 0;
    u8 rt = 0;
    u8 rt2 = 0;
    u8 rn = 0;
    u8 rm = 0;
    u8 rs = 0;
    u8 option = 0;
    u8 shift = 0;
    u8 lane = 0;
    u8 structures = 1;
    u8 repetitions = 1;
    u8 operation = 0;
    u8 register_bytes = 0;
    bool load = false;
    bool quad = false;
    bool pair = false;
    bool replicate = false;
    bool acquire = false;
    bool release = false;
    bool sign_extend = false;
    s64 displacement = 0;

    bool Writeback() const {
        return address == Address::PreIndex || address == Address::PostIndex;
    }
};

bool DecodeBaselineMemory(u32 instruction, BaselineMemoryOp& result) noexcept;
bool BaselineMemorySupported(u32 instruction) noexcept;

// Returns true only after the entire architectural instruction completed.
// Faults and interrupted atomic retries leave PC and destinations uncommitted.
// No frontend, optimizing compiler, JIT lookup, or interpreter is invoked.
bool ExecuteBaselineMemory(A64JitState* state, u32 instruction);

}  // namespace Dynarmic::Backend::X64
