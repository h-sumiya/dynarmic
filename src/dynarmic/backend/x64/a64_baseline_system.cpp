// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
// Independent scalar extension and EL0 system semantics. No IR is constructed.

#include "dynarmic/backend/x64/a64_baseline_system.h"

#include <array>
#include <atomic>
#include <bit>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic::Backend::X64 {
namespace {
u64 Read(const A64JitState& state, unsigned reg) {
    return reg == 31 ? 0 : state.reg[reg];
}
void Write(A64JitState& state, unsigned reg, u64 value) {
    if (reg != 31) state.reg[reg] = value;
}

constexpr auto MakeCrcTable(u32 polynomial) {
    std::array<u32, 256> result{};
    for (u32 i = 0; i < result.size(); ++i) {
        u32 value = i;
        for (unsigned j = 0; j < 8; ++j)
            value = (value >> 1) ^ ((0u - (value & 1)) & polynomial);
        result[i] = value;
    }
    return result;
}
constexpr auto Crc32 = MakeCrcTable(0xedb88320u);
constexpr auto Crc32c = MakeCrcTable(0x82f63b78u);

bool CrcEncoding(u32 instruction) {
    if ((instruction & 0x7fe0e000u) != 0x1ac04000u) return false;
    const unsigned size = (instruction >> 10) & 3;
    return bool(instruction >> 31) == (size == 3);
}

bool CacheOperation(u32 instruction, A64::DataCacheOperation& operation) {
    using D = A64::DataCacheOperation;
    switch (instruction & ~31u) {
    case 0xd5087620: operation = D::InvalidateByVAToPoC; return true;
    case 0xd5087640: operation = D::InvalidateBySetWay; return true;
    case 0xd5087a40: operation = D::CleanBySetWay; return true;
    case 0xd5087e40: operation = D::CleanAndInvalidateBySetWay; return true;
    case 0xd50b7420: operation = D::ZeroByVA; return true;
    case 0xd50b7a20: operation = D::CleanByVAToPoC; return true;
    case 0xd50b7b20: operation = D::CleanByVAToPoU; return true;
    case 0xd50b7c20: operation = D::CleanByVAToPoP; return true;
    case 0xd50b7e20: operation = D::CleanAndInvalidateByVAToPoC; return true;
    default: return false;
    }
}

bool Fail(A64JitState& state, A64::Exception exception, u32 instruction, bool unsupported) {
    if (unsupported) {
        state.baseline_last_unsupported = instruction;
        ++state.baseline_unsupported_executions;
    }
    state.runtime_callbacks->ExceptionRaised(state.pc, exception);
    // A callback is permitted merely to record the exception. Still stop here:
    // returning to an unchanged PC must never create an invisible infinite loop.
    if (!Atomic::Load(&state.halt_reason))
        Atomic::Or(&state.halt_reason, static_cast<u32>(HaltReason::UserDefined3));
    return false;
}
}  // namespace

bool ExecuteBaselineUndefined(A64JitState* state, u32 instruction) {
    return Fail(*state, A64::Exception::UnallocatedEncoding, instruction, true);
}
bool ExecuteBaselineNoExecute(A64JitState* state, u32) {
    return Fail(*state, A64::Exception::NoExecuteFault, 0, false);
}

bool BaselineSystemSupported(u32 instruction) {
    if (CrcEncoding(instruction)) return true;
    if ((instruction & 0xffe07c10u) == 0xba000400u) return true;  // RMIF
    if ((instruction & 0xffffbc1fu) == 0x3a00080du) return true;  // SETF8/16
    if ((instruction & 0xfffff01fu) == 0xd503201fu) return true;  // HINT space
    if ((instruction & 0xfffff0ffu) == 0xd503305fu ||
        (instruction & 0xfffff0ffu) == 0xd503309fu ||
        (instruction & 0xfffff0ffu) == 0xd50330bfu ||
        (instruction & 0xfffff0ffu) == 0xd50330dfu) return true;
    if ((instruction & 0xffe0001fu) == 0xd4000001u ||
        (instruction & 0xffe0001fu) == 0xd4200000u) return true;
    if (instruction == 0xd500401fu || instruction == 0xd500403fu ||
        instruction == 0xd500405fu || instruction == 0xd50330ffu ||
        instruction == 0xd508751fu || instruction == 0xd508711fu) return true;
    A64::DataCacheOperation operation{};
    if (CacheOperation(instruction, operation)) return true;
    switch (instruction & ~31u) {
    case 0xd53b4400: case 0xd53b4420: case 0xd53b4200:
    case 0xd51b4400: case 0xd51b4420: case 0xd51b4200:
    case 0xd53be000: case 0xd53be020: case 0xd53be040:
    case 0xd53b0020: case 0xd53b00e0:
    case 0xd53bd040: case 0xd53bd060: case 0xd51bd040:
    case 0xd50b7520: return true;
    default: return false;
    }
}

bool BaselineSystemTerminal(u32 instruction) {
    return (instruction & 0xffe0001fu) == 0xd4000001u ||
           (instruction & 0xffe0001fu) == 0xd4200000u ||
           (instruction & 0xfffff0ffu) == 0xd50330dfu ||
           (instruction & ~31u) == 0xd51b4400u ||
           (instruction & ~31u) == 0xd50b7520u ||
           instruction == 0xd508751fu || instruction == 0xd508711fu;
}

bool ExecuteBaselineSystem(A64JitState* pointer, u32 instruction) {
    auto& state = *pointer;
    const auto& config = *state.runtime_config;
    auto& callbacks = *state.runtime_callbacks;
    const unsigned rd = instruction & 31;
    const unsigned rn = (instruction >> 5) & 31;
    const unsigned rm = (instruction >> 16) & 31;

    if (CrcEncoding(instruction)) {
        const auto& table = instruction & (1u << 12) ? Crc32c : Crc32;
        u32 crc = static_cast<u32>(Read(state, rn));
        u64 value = Read(state, rm);
        for (unsigned i = 0, bytes = 1u << ((instruction >> 10) & 3); i < bytes; ++i) {
            crc = table[(crc ^ value) & 255] ^ (crc >> 8);
            value >>= 8;
        }
        Write(state, rd, crc);
        return true;
    }
    if ((instruction & 0xffe07c10u) == 0xba000400u) {
        const u32 mask = (instruction & 15) << 28;
        const u32 value = static_cast<u32>(std::rotr(Read(state, rn), int((instruction >> 15) & 63))) << 28;
        state.SetPstate((state.GetPstate() & ~mask) | (value & mask));
        return true;
    }
    if ((instruction & 0xffffbc1fu) == 0x3a00080du) {
        const unsigned size = instruction & (1u << 14) ? 16 : 8;
        const u32 value = static_cast<u32>(Read(state, rn));
        const bool n = (value >> (size - 1)) & 1;
        const bool z = !(value & ((1u << size) - 1));
        const bool v = n ^ bool((value >> size) & 1);
        state.SetPstate((state.GetPstate() & 0x20000000u) | (u32(n) << 31) | (u32(z) << 30) | (u32(v) << 28));
        return true;
    }
    if (instruction == 0xd500401fu) {
        state.SetPstate(state.GetPstate() ^ 0x20000000u);
        return true;
    }
    if (instruction == 0xd500403fu || instruction == 0xd500405fu) {
        const u32 old = state.GetPstate();
        const bool z = (old >> 30) & 1, c = (old >> 29) & 1, v = (old >> 28) & 1;
        if (instruction == 0xd500405fu)
            state.SetPstate((u32(z || v) << 30) | (u32(c && !v) << 29));
        else
            state.SetPstate((u32(!c && !z) << 31) | (u32(z && c) << 30) |
                            (u32(c || z) << 29) | (u32(!c && z) << 28));
        return true;
    }

    if ((instruction & 0xfffff01fu) == 0xd503201fu) {
        const unsigned hint = (instruction >> 5) & 127;
        if (config.hook_hint_instructions && hint >= 1 && hint <= 5) {
            constexpr A64::Exception exceptions[] = {A64::Exception::Yield,
                A64::Exception::WaitForEvent, A64::Exception::WaitForInterrupt,
                A64::Exception::SendEvent, A64::Exception::SendEventLocal};
            callbacks.ExceptionRaised(state.pc, exceptions[hint - 1]);
        }
        // Architecturally allocated HINT space is NOP when its optional feature
        // is absent. This includes PAC/BTI hints in this non-PAuth EL0 profile.
        return true;
    }
    if ((instruction & 0xfffff0ffu) == 0xd503305fu) {
        state.exclusive_state = 0;
        // As in the optimizing backend, the local bit gates every STXR. The
        // next LDXR replaces this processor's global reservation. Clearing all
        // monitor shards here is unnecessary and would serialize CLREX.
        return true;
    }
    if ((instruction & 0xfffff0ffu) == 0xd503309fu ||
        (instruction & 0xfffff0ffu) == 0xd50330bfu || instruction == 0xd50330ffu) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return true;
    }
    if ((instruction & 0xfffff0ffu) == 0xd50330dfu) {
        if (config.hook_isb) callbacks.InstructionSynchronizationBarrierRaised();
        state.pc += 4;
        return true;
    }
    if ((instruction & 0xffe0001fu) == 0xd4000001u) {
        state.pc += 4;
        callbacks.CallSVC((instruction >> 5) & 65535);
        return true;
    }
    // BRK uses the configured architectural breakpoint callback. HLT needs a
    // separate external-debug/semihosting model and remains explicitly unsupported.
    if ((instruction & 0xffe0001fu) == 0xd4200000u)
        return Fail(state, A64::Exception::Breakpoint, instruction, false);

    if ((instruction & ~31u) == 0xd50b7520u || instruction == 0xd508751fu || instruction == 0xd508711fu) {
        using I = A64::InstructionCacheOperation;
        if ((instruction & ~31u) == 0xd50b7520u)
            callbacks.InstructionCacheOperationRaised(I::InvalidateByVAToPoU, Read(state, rd));
        else
            callbacks.InstructionCacheOperationRaised(instruction == 0xd508751fu ? I::InvalidateAllToPoU : I::InvalidateAllToPoUInnerSharable, 0);
        state.pc += 4;
        return true;
    }
    A64::DataCacheOperation operation{};
    if (CacheOperation(instruction, operation)) {
        const u64 address = Read(state, rd);
        if (config.hook_data_cache_operations) {
            callbacks.DataCacheOperationRaised(operation, address);
        } else if (operation == A64::DataCacheOperation::ZeroByVA) {
            if (config.dczid_el0 & 16)
                return Fail(state, A64::Exception::UnallocatedEncoding, instruction, false);
            const u64 bytes = u64{4} << (config.dczid_el0 & 15);
            const u64 base = address & ~(bytes - 1);
            for (u64 i = 0; i < bytes; i += 4) {
                callbacks.MemoryWrite32(base + i, 0);
                if (Atomic::Load(&state.halt_reason) & static_cast<u32>(HaltReason::MemoryAbort)) return false;
            }
        }
        return true;
    }

    switch (instruction & ~31u) {
    case 0xd53b4400: Write(state, rd, state.GetFpcr()); return true;
    case 0xd53b4420: Write(state, rd, state.GetFpsr()); return true;
    case 0xd53b4200: Write(state, rd, state.GetPstate()); return true;
    case 0xd51b4400: state.SetFpcr(static_cast<u32>(Read(state, rd))); state.pc += 4; return true;
    case 0xd51b4420: state.SetFpsr(static_cast<u32>(Read(state, rd))); return true;
    case 0xd51b4200: state.SetPstate(static_cast<u32>(Read(state, rd))); return true;
    case 0xd53be000: Write(state, rd, config.cntfrq_el0); return true;
    case 0xd53be020: case 0xd53be040: Write(state, rd, callbacks.GetCNTPCT()); return true;
    case 0xd53b0020: Write(state, rd, config.ctr_el0); return true;
    case 0xd53b00e0: Write(state, rd, config.dczid_el0); return true;
    case 0xd53bd040: Write(state, rd, state.runtime_tpidr_el0 ? *state.runtime_tpidr_el0 : 0); return true;
    case 0xd53bd060: Write(state, rd, state.runtime_tpidrro_el0 ? *state.runtime_tpidrro_el0 : 0); return true;
    case 0xd51bd040: if (state.runtime_tpidr_el0) *state.runtime_tpidr_el0 = Read(state, rd); return true;
    default: return ExecuteBaselineUndefined(&state, instruction);
    }
}
}  // namespace Dynarmic::Backend::X64
