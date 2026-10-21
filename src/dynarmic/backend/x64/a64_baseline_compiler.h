// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <mcl/stdint.hpp>

namespace Dynarmic::A64 {
struct UserConfig;
}

namespace Dynarmic::Backend::X64 {

struct A64JitState;
using BaselineReadCode = std::function<std::optional<u32>(u64)>;

/// A fault at code[fault_offset] resumes at code[recovery_offset]. The handler
/// changes only RIP; it must preserve the native register/stack state. Both
/// offsets belong to this immutable image and survive address relocation.
struct A64BaselineFault {
    u32 fault_offset = 0;
    u32 recovery_offset = 0;
};

/// Immutable, position-independent Tier 0 machine code. There are deliberately
/// no guest addresses, callback objects, code-cache addresses, or relocations
/// in this payload. It can be copied to another address and run by another CPU
/// context after validating the originating guest instruction bytes.
struct A64BaselineBlock {
    std::vector<u8> code;
    std::vector<A64BaselineFault> faults;
    u32 guest_size = 0;
    u32 instruction_count = 0;
    u32 helper_instruction_count = 0;
    u32 unsupported_instruction_count = 0;
    bool stopped_at_unsupported = false;
};

/// Compile a bounded linear prefix directly to x86-64, without constructing
/// Dynarmic IR, discovering a CFG, running SSA passes, or allocating an IR
/// register allocator.
/// Complex instructions call independent, precompiled semantic helpers. There
/// is no optimizing-translator fallback. Unallocated/unimplemented encodings
/// emit an explicit exception at their exact PC. max_instructions is clamped
/// to [1, 256]. A missing first code word emits NoExecuteFault; a readable
/// prefix ends before a missing later word, which faults on its next entry.
///
/// Entry: r15 points to A64JitState; rsp obeys Dynarmic's StackLayout ABI and
/// guest MXCSR is active and state.pc names the block entry. Exit: pc names the
/// next instruction or exact faulting instruction, and all architectural state
/// is materialized. Edges use the thread's A64JitState::baseline_return;
/// self-edges check halt and instruction budget before repeating. r13/r14
/// retain the thread's fastmem/page-table bases.
/// For 64-bit unmirrored fastmem, r13 supplies this execution's fastmem base;
/// the caller must install the immutable fault records before executing code.
std::optional<A64BaselineBlock> CompileA64Baseline(
    u64 pc, const BaselineReadCode& get_code,
    const A64::UserConfig& config, u32 max_instructions = 32,
    bool single_step = false);

/// Install process-local callback veneers into a CPU context. The pointers
/// live in the context, not in persistent machine code. Set runtime_callbacks
/// and baseline_return separately before entering the generated block.
void InitializeA64BaselineState(A64JitState& state);

}  // namespace Dynarmic::Backend::X64
