// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
//
// This emitter intentionally trades peak throughput for bounded translation
// latency. Four GPR and four vector slots remove local get/set traffic,
// but there are no IR nodes, CFG walks, liveness passes, or global optimization
// decisions. All support tests finish before emitting an instruction.

#include "dynarmic/backend/x64/a64_baseline_compiler.h"
#include "dynarmic/backend/x64/a64_baseline_memory.h"
#include "dynarmic/backend/x64/a64_baseline_simd.h"
#include "dynarmic/backend/x64/a64_baseline_system.h"
#include "dynarmic/backend/x64/a64_baseline_crypto.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include <xbyak/xbyak.h>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic::Backend::X64 {
namespace {

// The spelling of these encodings follows the pinned A64 decoder. A consteval
// constructor turns every pattern into a mask/test at compile time; decoding
// a guest instruction never scans these strings.
struct Pattern {
    u32 mask = 0;
    u32 value = 0;
    template<std::size_t N>
    consteval Pattern(const char (&bits)[N]) {
        static_assert(N == 33);
        for (std::size_t i = 0; i < 32; ++i) {
            mask = (mask << 1) | (bits[i] == '0' || bits[i] == '1');
            value = (value << 1) | (bits[i] == '1');
        }
    }
    constexpr bool operator()(u32 instruction) const {
        return (instruction & mask) == value;
    }
};

constexpr u64 Ones(unsigned count) {
    return count >= 64 ? ~u64{0} : (u64{1} << count) - 1;
}

constexpr s64 SignExtend(u64 value, unsigned bits) {
    const u64 sign = u64{1} << (bits - 1);
    return static_cast<s64>((value ^ sign) - sign);
}

std::optional<u64> LogicalImmediate(bool wide, bool n, unsigned s, unsigned r) {
    if (!wide && n)
        return std::nullopt;
    const unsigned combined = (unsigned(n) << 6) | (s ^ 63);
    if (combined < 2)
        return std::nullopt;
    const unsigned length = 31 - std::countl_zero(combined);
    const unsigned size = 1u << length;
    const unsigned levels = size - 1;
    if ((s & levels) == levels)
        return std::nullopt;
    u64 element = Ones((s & levels) + 1);
    const unsigned rotate = r & levels;
    if (rotate)
        element = ((element >> rotate) | (element << (size - rotate))) & Ones(size);
    u64 result = 0;
    for (unsigned offset = 0; offset < (wide ? 64u : 32u); offset += size)
        result |= element << offset;
    return result;
}

u64 ReadMemory(A64JitState* state, u64 address, u32 bytes) {
    switch (bytes) {
    case 1: return state->runtime_callbacks->MemoryRead8(address);
    case 2: return state->runtime_callbacks->MemoryRead16(address);
    case 4: return state->runtime_callbacks->MemoryRead32(address);
    case 8: return state->runtime_callbacks->MemoryRead64(address);
    default: return 0;  // Only validated emitter-owned sizes reach this veneer.
    }
}

void WriteMemory(A64JitState* state, u64 address, u64 value, u32 bytes) {
    switch (bytes) {
    case 1: state->runtime_callbacks->MemoryWrite8(address, static_cast<u8>(value)); break;
    case 2: state->runtime_callbacks->MemoryWrite16(address, static_cast<u16>(value)); break;
    case 4: state->runtime_callbacks->MemoryWrite32(address, static_cast<u32>(value)); break;
    case 8: state->runtime_callbacks->MemoryWrite64(address, value); break;
    }
}

class Emitter final : public Xbyak::CodeGenerator {
public:
    Emitter(void* buffer, std::size_t capacity, const A64::UserConfig& config, bool single_step)
            : CodeGenerator(capacity, buffer), config(config), single_step(single_step) {
        // The thread's dispatcher publishes the entry PC before every block.
        // r12 retains it across callbacks and PC-relative instructions.
        mov(r12, qword[r15 + offsetof(A64JitState, pc)]);
        L(loop_entry);
        if (config.enable_baseline_hot_optimization && !single_step) {
            // One thread-local decrement per entry/self-edge. The C++ lookup
            // and optimizing compiler run only after repeated sparse samples.
            // This fixed sequence uses only state offsets. Copying it avoids
            // encoding instructions and allocating a label on every cold block.
            static constexpr auto probe = [] {
                std::array<u8, 27> bytes{0x41, 0xff, 0x8f, 0, 0, 0, 0,  // dec [r15+countdown]
                    0x75, 18,                                          // jnz body
                    0x41, 0xc7, 0x87, 0, 0, 0, 0, 1, 0, 0, 0,         // mov [r15+sample],1
                    0x41, 0xff, 0xa7, 0, 0, 0, 0};                     // jmp [r15+return]
                const auto offset = [&](unsigned at, u32 value) {
                    for (unsigned n = 0; n < 4; ++n) bytes[at + n] = static_cast<u8>(value >> (n * 8));
                };
                offset(3, offsetof(A64JitState, baseline_hot_countdown));
                offset(12, offsetof(A64JitState, baseline_hot_sample));
                offset(23, offsetof(A64JitState, baseline_hot_return));
                return bytes;
            }();
            db(probe.data(), probe.size());
        }
    }

    enum class Result { Unsupported, Continue, Terminal };

    Result Instruction(u32 instruction, u32 byte_offset);

    void Undefined(u32 instruction, u32 byte_offset, bool no_execute = false) {
        offset = byte_offset;
        EmitHelper(no_execute ? offsetof(A64JitState, baseline_no_execute)
                              : offsetof(A64JitState, baseline_undefined_instruction), instruction, true);
        unsupported_instructions += !no_execute;
    }

    void Finish(u32 byte_count, u32 instruction_count) {
        Flush();
        lea(rax, ptr[r12 + byte_count]);
        mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        BranchReturn(instruction_count);
    }

    void CopyFaults(A64BaselineBlock& block) const {
        block.faults.assign(faults.begin(), faults.begin() + fault_count);
        block.helper_instruction_count = helper_instructions;
        block.unsupported_instruction_count = unsupported_instructions;
    }

private:
    const A64::UserConfig& config;
    const bool single_step;
    Xbyak::Label loop_entry;
    u32 helper_instructions = 0;
    u32 unsupported_instructions = 0;
    std::array<bool, 256> helper_sites{};
    struct RegisterSlot {
        int guest = -1;
        bool dirty = false;
    };
    std::array<RegisterSlot, 4> slots{};
    std::array<RegisterSlot, 4> vector_slots{};
    unsigned replacement = 0;
    unsigned vector_replacement = 0;
    u32 offset = 0;
    // At most 256 guest instructions and four fault sites per instruction.
    std::array<A64BaselineFault, 1024> faults{};
    unsigned fault_count = 0;

    bool FastMemory() const {
        // More general page-table/mirrored layouts stay with precise callbacks.
        // A process-specific base never appears in the immutable machine code.
        return config.fastmem_pointer.has_value() &&
               config.fastmem_address_space_bits == 64 && !config.silently_mirror_fastmem;
    }

    Xbyak::RegExp FastmemAddress(s32 displacement = 0) const {
        if (config.optimize_zero_base_fastmem && config.fastmem_pointer == uintptr_t{0})
            return rbx + displacement;
        return r13 + rbx + displacement;
    }

    static std::size_t RegisterOffset(unsigned guest) {
        return guest == 31 ? offsetof(A64JitState, sp)
                           : offsetof(A64JitState, reg) + guest * sizeof(u64);
    }

    Xbyak::Reg64 SlotRegister(unsigned slot) const {
        return Xbyak::Reg64(8 + static_cast<int>(slot));
    }

    void Read(unsigned guest, const Xbyak::Reg64& destination, bool wide = true, bool sp = false) {
        if (guest == 31 && !sp) {
            xor_(destination.cvt32(), destination.cvt32());
            return;
        }
        for (unsigned i = 0; i < slots.size(); ++i) {
            if (slots[i].guest == static_cast<int>(guest)) {
                if (wide)
                    mov(destination, SlotRegister(i));
                else
                    mov(destination.cvt32(), SlotRegister(i).cvt32());
                return;
            }
        }
        if (wide)
            mov(destination, qword[r15 + RegisterOffset(guest)]);
        else
            mov(destination.cvt32(), dword[r15 + RegisterOffset(guest)]);
    }

    void Write(unsigned guest, const Xbyak::Reg64& source, bool wide = true, bool sp = false) {
        if (guest == 31 && !sp)
            return;
        unsigned slot = replacement;
        for (unsigned i = 0; i < slots.size(); ++i) {
            if (slots[i].guest == static_cast<int>(guest)) {
                slot = i;
                break;
            }
            if (slots[i].guest == -1)
                slot = i;
        }
        if (slots[slot].guest != static_cast<int>(guest)) {
            Spill(slot);
            replacement = (slot + 1) % slots.size();
        }
        if (wide)
            mov(SlotRegister(slot), source);
        else
            mov(SlotRegister(slot).cvt32(), source.cvt32());
        slots[slot] = {static_cast<int>(guest), true};
    }

    void Spill(unsigned slot) {
        if (slots[slot].dirty) {
            mov(qword[r15 + RegisterOffset(slots[slot].guest)], SlotRegister(slot));
            slots[slot].dirty = false;
        }
    }

    void Flush(bool invalidate = false) {
        for (unsigned i = 0; i < slots.size(); ++i) {
            Spill(i);
            if (invalidate)
                slots[i].guest = -1;
        }
        for (unsigned i = 0; i < vector_slots.size(); ++i) {
            auto& slot = vector_slots[i];
            if (slot.dirty) {
                movdqu(xword[r15 + offsetof(A64JitState, vec) + slot.guest * 16], VectorSlotRegister(i));
                slot.dirty = false;
            }
            if (invalidate) slot.guest = -1;
        }
    }

    // Cold fault veneers materialize the compile-time snapshot without changing
    // the hot-path cache mapping or dirty bits. After a callback, reload that
    // mapping because the ABI and the callback may have changed its registers.
    void SpillForMemoryCallback() {
        for (unsigned i = 0; i < slots.size(); ++i)
            if (slots[i].dirty)
                mov(qword[r15 + RegisterOffset(slots[i].guest)], SlotRegister(i));
        for (unsigned i = 0; i < vector_slots.size(); ++i)
            if (vector_slots[i].dirty)
                movdqu(xword[r15 + offsetof(A64JitState, vec) + vector_slots[i].guest * 16], VectorSlotRegister(i));
        lea(rcx, ptr[r12 + offset]);
        mov(qword[r15 + offsetof(A64JitState, pc)], rcx);
    }

    void ReloadAfterMemoryCallback() {
        for (unsigned i = 0; i < slots.size(); ++i)
            if (slots[i].guest != -1)
                mov(SlotRegister(i), qword[r15 + RegisterOffset(slots[i].guest)]);
        for (unsigned i = 0; i < vector_slots.size(); ++i)
            if (vector_slots[i].guest != -1)
                movdqu(VectorSlotRegister(i), xword[r15 + offsetof(A64JitState, vec) + vector_slots[i].guest * 16]);
    }

    void Mask(const Xbyak::Reg64& target, u64 value, bool wide) {
        if (!wide) {
            and_(target.cvt32(), static_cast<u32>(value));
        } else if (static_cast<u64>(static_cast<s64>(static_cast<s32>(value))) == value) {
            and_(target, static_cast<u32>(value));
        } else {
            mov(rbx, value);
            and_(target, rbx);
        }
    }

    void AddOffset(const Xbyak::Reg64& target, s64 delta) {
        if (!delta)
            return;
        if (delta >= std::numeric_limits<s32>::min() && delta <= std::numeric_limits<s32>::max()) {
            add(target, static_cast<u32>(delta));
        } else {
            mov(rcx, static_cast<u64>(delta));
            add(target, rcx);
        }
    }

    void Shift(const Xbyak::Reg64& target, unsigned kind, unsigned amount, bool wide) {
        if (!amount)
            return;
        const Xbyak::Reg operand = wide ? Xbyak::Reg(target) : Xbyak::Reg(target.cvt32());
        switch (kind) {
        case 0: shl(operand, amount); break;
        case 1: shr(operand, amount); break;
        case 2: sar(operand, amount); break;
        case 3: ror(operand, amount); break;
        }
    }

    void Extend(const Xbyak::Reg64& target, unsigned option, bool wide) {
        switch (option) {
        case 0: movzx(target.cvt32(), target.cvt8()); break;
        case 1: movzx(target.cvt32(), target.cvt16()); break;
        case 2: mov(target.cvt32(), target.cvt32()); break;
        case 3: if (!wide) mov(target.cvt32(), target.cvt32()); break;
        case 4:
            if (wide) movsx(target, target.cvt8());
            else movsx(target.cvt32(), target.cvt8());
            break;
        case 5:
            if (wide) movsx(target, target.cvt16());
            else movsx(target.cvt32(), target.cvt16());
            break;
        case 6:
            if (wide) movsxd(target, target.cvt32());
            else mov(target.cvt32(), target.cvt32());
            break;
        case 7: if (!wide) mov(target.cvt32(), target.cvt32()); break;
        }
    }

    // LAHF packs N/Z/C into the exact positions used by A64JitState; SETO
    // supplies V. MOV preserves the arithmetic flags and the result in rax.
    void SaveFlags(bool subtract) {
        mov(rbx, rax);
        seto(al);
        lahf();
        and_(eax, NZCV::x64_mask);
        if (subtract)
            xor_(eax, 1u << NZCV::x64_c_flag_bit);
        mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], eax);
        mov(rax, rbx);
    }

    void LoadFlags(unsigned condition) {
        if (condition >= 14)
            return;
        mov(eax, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);
        if (condition >= 6)
            cmp(al, 0x81);  // Restore OF from packed V without touching AH.
        sahf();
        if (condition == 8 || condition == 9)
            cmc();  // ARM HI/LS use C as no-borrow.
    }

    void Select(unsigned condition, const Xbyak::Reg64& destination, const Xbyak::Reg64& if_true) {
        switch (condition) {
        case 0: cmovz(destination, if_true); break;
        case 1: cmovnz(destination, if_true); break;
        case 2: cmovc(destination, if_true); break;
        case 3: cmovnc(destination, if_true); break;
        case 4: cmovs(destination, if_true); break;
        case 5: cmovns(destination, if_true); break;
        case 6: cmovo(destination, if_true); break;
        case 7: cmovno(destination, if_true); break;
        case 8: cmova(destination, if_true); break;
        case 9: cmovbe(destination, if_true); break;
        case 10: cmovge(destination, if_true); break;
        case 11: cmovl(destination, if_true); break;
        case 12: cmovg(destination, if_true); break;
        case 13: cmovle(destination, if_true); break;
        case 14: case 15: mov(destination, if_true); break;
        }
    }

    void Return(u32 instruction_count) {
        if (config.enable_cycle_counting)
            sub(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], instruction_count);
        jmp(ptr[r15 + offsetof(A64JitState, baseline_return)]);
    }

    void BranchReturn(u32 instruction_count, bool has_self_edge = false) {
        if (!has_self_edge || single_step) {
            Return(instruction_count);
            return;
        }
        if (config.enable_cycle_counting)
            sub(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], instruction_count);
        Xbyak::Label dispatch;
        cmp(qword[r15 + offsetof(A64JitState, pc)], r12);
        jne(dispatch, T_NEAR);
        cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
        jne(dispatch, T_NEAR);
        if (config.enable_cycle_counting) {
            cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
            jle(dispatch, T_NEAR);
        }
        // Code invalidation sets halt_reason; the owner reclaims code only
        // after RunCode returns. No shared image, epoch or batch is involved.
        jmp(loop_entry, T_NEAR);
        L(dispatch);
        jmp(ptr[r15 + offsetof(A64JitState, baseline_return)]);
    }

    void ExactPC() {
        lea(rax, ptr[r12 + offset]);
        mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    }

    void EnterHost() {
        stmxcsr(dword[r15 + offsetof(A64JitState, guest_MXCSR)]);
        ldmxcsr(dword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, save_host_MXCSR)]);
    }

    void EnterGuest() {
        // Match BlockOfCode::SwitchMxcsrOnEntry: callbacks may change the host
        // floating-point environment, which must be restored on the final exit.
        stmxcsr(dword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, save_host_MXCSR)]);
        ldmxcsr(dword[r15 + offsetof(A64JitState, guest_MXCSR)]);
    }

    void EmitHelper(std::size_t helper_offset, u32 instruction, bool terminal = false, bool count_site = true) {
        if (count_site && !helper_sites[offset / 4]) {
            helper_sites[offset / 4] = true;
            ++helper_instructions;
        }
        Flush(true);
        ExactPC();
        EnterHost();
#ifdef _WIN32
        mov(rcx, r15);
        mov(edx, instruction);
        if (helper_offset == offsetof(A64JitState, baseline_system_instruction))
            lea(r8, ptr[rsp + ABI_SHADOW_SPACE]);
#else
        mov(rdi, r15);
        mov(esi, instruction);
        if (helper_offset == offsetof(A64JitState, baseline_system_instruction))
            lea(rdx, ptr[rsp + ABI_SHADOW_SPACE]);
#endif
        call(ptr[r15 + helper_offset]);
        EnterGuest();
        if (terminal) {
            Return(offset / 4 + 1);
            return;
        }
        Xbyak::Label complete, resume;
        test(al, al);
        jnz(complete, T_NEAR);
        Return(offset / 4 + 1);
        L(complete);
        cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
        je(resume, T_NEAR);
        lea(rax, ptr[r12 + offset + 4]);
        mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        Return(offset / 4 + 1);
        L(resume);
    }

    void CheckMemoryAbort() {
        Xbyak::Label valid;
        test(dword[r15 + offsetof(A64JitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
        jz(valid, T_NEAR);
        Return(offset / 4 + 1);
        L(valid);
    }

    // Address is kept in callee-saved rbx. Only the callback/fault path spills
    // cached registers; a successful native access leaves both caches intact.
    void MemoryRead(unsigned bytes) {
        Xbyak::Label done;
        if (FastMemory()) {
            const auto fault = static_cast<u32>(getSize());
            switch (bytes) {
            case 1: movzx(eax, byte[FastmemAddress()]); break;
            case 2: movzx(eax, word[FastmemAddress()]); break;
            case 4: mov(eax, dword[FastmemAddress()]); break;
            case 8: mov(rax, qword[FastmemAddress()]); break;
            }
            jmp(done, T_NEAR);
            faults[fault_count++] = {fault, static_cast<u32>(getSize())};
        }
        SpillForMemoryCallback();
        EnterHost();
#ifdef _WIN32
        mov(rcx, r15);
        mov(rdx, rbx);
        mov(r8d, bytes);
#else
        mov(rdi, r15);
        mov(rsi, rbx);
        mov(edx, bytes);
#endif
        call(ptr[r15 + offsetof(A64JitState, baseline_memory_read)]);
        EnterGuest();
        CheckMemoryAbort();
        ReloadAfterMemoryCallback();
        L(done);
    }

    // Value is in rax on entry. The callback/fault veneer materializes precise
    // state without changing the compile-time register-cache assignments.
    void MemoryWrite(unsigned bytes) {
        Xbyak::Label done;
        if (FastMemory()) {
            const auto fault = static_cast<u32>(getSize());
            switch (bytes) {
            case 1: mov(byte[FastmemAddress()], al); break;
            case 2: mov(word[FastmemAddress()], ax); break;
            case 4: mov(dword[FastmemAddress()], eax); break;
            case 8: mov(qword[FastmemAddress()], rax); break;
            }
            jmp(done, T_NEAR);
            faults[fault_count++] = {fault, static_cast<u32>(getSize())};
        }
        SpillForMemoryCallback();
        EnterHost();
#ifdef _WIN32
        mov(rcx, r15);
        mov(rdx, rbx);
        mov(r8, rax);
        mov(r9d, bytes);
#else
        mov(rdi, r15);
        mov(rsi, rbx);
        mov(rdx, rax);
        mov(ecx, bytes);
#endif
        call(ptr[r15 + offsetof(A64JitState, baseline_memory_write)]);
        EnterGuest();
        CheckMemoryAbort();
        ReloadAfterMemoryCallback();
        L(done);
    }

    void ExtendLoad(unsigned bytes, bool signed_load, bool wide) {
        if (signed_load) {
            if (bytes == 1) {
                if (wide) movsx(rax, al); else movsx(eax, al);
            } else if (bytes == 2) {
                if (wide) movsx(rax, ax); else movsx(eax, ax);
            } else if (bytes == 4) {
                movsxd(rax, eax);
            }
        } else if (bytes < 8) {
            mov(eax, eax);
        }
    }

    Xbyak::Xmm VectorSlotRegister(unsigned slot) const {
        return Xbyak::Xmm(8 + static_cast<int>(slot));
    }

    void ReadVector(unsigned guest, const Xbyak::Xmm& destination, bool quad = true) {
        for (unsigned i = 0; i < vector_slots.size(); ++i) {
            if (vector_slots[i].guest != static_cast<int>(guest)) continue;
            if (quad) movdqa(destination, VectorSlotRegister(i));
            else movq(destination, VectorSlotRegister(i));
            return;
        }
        const auto displacement = offsetof(A64JitState, vec) + guest * 16;
        if (quad) movdqu(destination, xword[r15 + displacement]);
        else movq(destination, qword[r15 + displacement]);
    }

    void WriteVector(unsigned guest, const Xbyak::Xmm& source, bool quad = true) {
        if (!quad)
            movq(source, source);  // Every D-register write clears V[127:64].
        unsigned selected = vector_replacement;
        for (unsigned i = 0; i < vector_slots.size(); ++i) {
            if (vector_slots[i].guest == static_cast<int>(guest)) { selected = i; break; }
            if (vector_slots[i].guest == -1) selected = i;
        }
        auto& slot = vector_slots[selected];
        if (slot.guest != static_cast<int>(guest)) {
            if (slot.dirty)
                movdqu(xword[r15 + offsetof(A64JitState, vec) + slot.guest * 16], VectorSlotRegister(selected));
            vector_replacement = (selected + 1) % vector_slots.size();
        }
        movdqa(VectorSlotRegister(selected), source);
        slot = {static_cast<int>(guest), true};
    }

    Result LoadStore(u32 instruction, bool register_offset, bool unsigned_offset);
    Result LoadStorePair(u32 instruction);
    Result MemoryInstruction(u32 instruction);
    Result SimdInstruction(u32 instruction);
};

#include "dynarmic/backend/x64/a64_baseline_memory_emit.inc"
#include "dynarmic/backend/x64/a64_baseline_simd_emit.inc"

Emitter::Result Emitter::LoadStore(u32 instruction, bool register_offset, bool unsigned_offset) {
    const unsigned size = instruction >> 30;
    const unsigned opcode = (instruction >> 22) & 3;
    const unsigned rn = (instruction >> 5) & 31;
    const unsigned rt = instruction & 31;
    const unsigned mode = (instruction >> 10) & 3;
    const bool writeback = !unsigned_offset && !register_offset && (mode & 1);
    const bool postindex = writeback && mode == 1;
    const bool load = opcode != 0;
    const bool signed_load = opcode >= 2;
    const bool wide = signed_load ? opcode == 2 : size == 3;
    const unsigned bytes = 1u << size;
    const unsigned option = (instruction >> 13) & 7;
    if (register_offset && !(option & 2))
        return Result::Unsupported;
    if ((size == 3 && opcode >= 2) || (size == 2 && opcode == 3)) {
        // Only architecturally valid PRFM encodings are consumed as NOPs.
        if (size == 3 && opcode == 2 &&
            (unsigned_offset || register_offset || mode == 0))
            return Result::Continue;
        return Result::Unsupported;
    }
    if (writeback && rn == rt && rn != 31)
        return Result::Unsupported;
    const s64 delta = unsigned_offset ? static_cast<s64>(((instruction >> 10) & 4095) << size)
                                     : SignExtend((instruction >> 12) & 511, 9);

    Read(rn, rbx, true, true);
    if (register_offset) {
        Read((instruction >> 16) & 31, rcx, true);
        Extend(rcx, option, true);
        if ((instruction >> 12) & 1)
            shl(rcx, size);
        add(rbx, rcx);
    } else if (!postindex) {
        AddOffset(rbx, delta);
    }
    if (load) {
        MemoryRead(bytes);
        ExtendLoad(bytes, signed_load, wide);
        Write(rt, rax, wide);
    } else {
        Read(rt, rax, bytes == 8);
        MemoryWrite(bytes);
    }
    if (writeback) {
        if (postindex)
            AddOffset(rbx, delta);
        Write(rn, rbx, true, true);
    }
    return Result::Continue;
}

Emitter::Result Emitter::LoadStorePair(u32 instruction) {
    const unsigned opc = instruction >> 30;
    const bool load = (instruction >> 22) & 1;
    const bool writeback = (instruction >> 23) & 1;
    const bool not_postindex = (instruction >> 24) & 1;
    const bool postindex = writeback && !not_postindex;
    const unsigned rt = instruction & 31;
    const unsigned rn = (instruction >> 5) & 31;
    const unsigned rt2 = (instruction >> 10) & 31;
    if (opc == 3 || (!load && (opc & 1)) ||
        (!not_postindex && !writeback && (opc & 1)) ||
        (writeback && rn != 31 && (rn == rt || rn == rt2)) ||
        (load && rt == rt2))
        return Result::Unsupported;
    const unsigned bytes = opc == 2 ? 8 : 4;
    const bool signed_load = opc == 1;
    const s64 delta = SignExtend((instruction >> 15) & 127, 7) * bytes;

    Read(rn, rbx, true, true);
    if (!postindex)
        AddOffset(rbx, delta);
    if (load) {
        MemoryRead(bytes);
        ExtendLoad(bytes, signed_load, bytes == 8 || signed_load);
        mov(rbp, rax);
        add(rbx, bytes);
        MemoryRead(bytes);
        ExtendLoad(bytes, signed_load, bytes == 8 || signed_load);
        // Commit both register destinations only after both loads succeed.
        Write(rt, rbp, bytes == 8 || signed_load);
        Write(rt2, rax, bytes == 8 || signed_load);
    } else {
        Read(rt, rax, bytes == 8);
        MemoryWrite(bytes);
        add(rbx, bytes);
        Read(rt2, rax, bytes == 8);
        MemoryWrite(bytes);
    }
    if (writeback) {
        AddOffset(rbx, postindex ? delta - bytes : -static_cast<s64>(bytes));
        Write(rn, rbx, true, true);
    }
    return Result::Continue;
}

Emitter::Result Emitter::Instruction(u32 instruction, u32 byte_offset) {
    offset = byte_offset;
    const bool wide = instruction >> 31;
    const unsigned bits = wide ? 64 : 32;
    const unsigned rd = instruction & 31;
    const unsigned rn = (instruction >> 5) & 31;
    const unsigned rm = (instruction >> 16) & 31;
    const Xbyak::Reg32e accumulator = wide ? Xbyak::Reg32e(rax) : Xbyak::Reg32e(eax);
    const Xbyak::Reg32e operand = wide ? Xbyak::Reg32e(rcx) : Xbyak::Reg32e(ecx);

    // The architectural major opcode is a cheap first-level dispatch. A branch
    // never scans SIMD patterns and an ordinary load never scans arithmetic.
    // This stays a single-pass template emitter; no decode tree is allocated.
    const unsigned major = (instruction >> 25) & 15;
    if (major == 8 || major == 9) {
        // Data processing: immediates.
        if (Pattern{"z0010001ssiiiiiiiiiiiinnnnnddddd"}(instruction) ||
            Pattern{"z0110001ssiiiiiiiiiiiinnnnnddddd"}(instruction) ||
            Pattern{"z1010001ssiiiiiiiiiiiinnnnnddddd"}(instruction) ||
            Pattern{"z1110001ssiiiiiiiiiiiinnnnnddddd"}(instruction)) {
            const unsigned shift = (instruction >> 22) & 3;
            if (shift > 1)
                return Result::Unsupported;
            const unsigned immediate = ((instruction >> 10) & 4095) << (shift * 12);
            const bool subtract = (instruction >> 30) & 1;
            const bool flags = (instruction >> 29) & 1;
            Read(rn, rax, wide, true);
            if (subtract) sub(accumulator, immediate); else add(accumulator, immediate);
            if (flags)
                SaveFlags(subtract);
            Write(rd, rax, wide, !flags);
            return Result::Continue;
        }
        if (Pattern{"zii10000iiiiiiiiiiiiiiiiiiiddddd"}(instruction)) {
            const s64 immediate = SignExtend(((instruction >> 5) & 0x7ffffu) * 4u + ((instruction >> 29) & 3), 21);
            mov(rax, r12);
            AddOffset(rax, offset);
            if (wide) {
                and_(rax, static_cast<u32>(-4096));
                AddOffset(rax, immediate * 4096);
            } else {
                AddOffset(rax, immediate);
            }
            Write(rd, rax);
            return Result::Continue;
        }
        if (Pattern{"zoo100101ssiiiiiiiiiiiiiiiiddddd"}(instruction)) {
            const unsigned opcode = (instruction >> 29) & 3;
            const unsigned shift = ((instruction >> 21) & 3) * 16;
            if (opcode == 1 || (!wide && shift >= 32))
                return Result::Unsupported;
            const u64 immediate = u64((instruction >> 5) & 65535) << shift;
            if (opcode == 3) {
                Read(rd, rax, wide);
                Mask(rax, ~(u64{65535} << shift), wide);
                mov(rcx, immediate);
                or_(accumulator, operand);
            } else {
                if (wide) mov(rax, opcode == 0 ? ~immediate : immediate);
                else mov(eax, static_cast<u32>(opcode == 0 ? ~immediate : immediate));
            }
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if (Pattern{"zoo100100Nrrrrrrssssssnnnnnddddd"}(instruction)) {
            const auto immediate = LogicalImmediate(wide, (instruction >> 22) & 1,
                                                    (instruction >> 10) & 63, (instruction >> 16) & 63);
            if (!immediate)
                return Result::Unsupported;
            const unsigned opcode = (instruction >> 29) & 3;
            Read(rn, rax, wide);
            if (wide) mov(rcx, *immediate); else mov(ecx, static_cast<u32>(*immediate));
            if (opcode == 0 || opcode == 3) and_(accumulator, operand);
            else if (opcode == 1) or_(accumulator, operand);
            else xor_(accumulator, operand);
            if (opcode == 3)
                SaveFlags(false);
            // AND/ORR/EOR immediate permit SP as destination; ANDS does not.
            Write(rd, rax, wide, opcode != 3);
            return Result::Continue;
        }
        if (Pattern{"zoo100110Nrrrrrrssssssnnnnnddddd"}(instruction)) {
            const unsigned opcode = (instruction >> 29) & 3;
            const unsigned r = (instruction >> 16) & 63;
            const unsigned s = (instruction >> 10) & 63;
            if (opcode == 3 || bool((instruction >> 22) & 1) != wide || r >= bits || s >= bits)
                return Result::Unsupported;
            Read(rn, rax, wide);
            if (opcode == 0) {  // SBFM, including all sign-extending aliases.
                Shift(rax, 0, bits - 1 - s, wide);
                Shift(rax, 2, s >= r ? bits - 1 - s + r : r - s - 1, wide);
            } else {
                const u64 mask = s >= r ? Ones(s - r + 1) : Ones(s + 1) << (bits - r);
                Shift(rax, 3, r, wide);
                Mask(rax, mask, wide);
                if (opcode == 1) {
                    Read(rd, rcx, wide);
                    Mask(rcx, ~mask, wide);
                    or_(accumulator, operand);
                }
            }
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if (Pattern{"z00100111N0mmmmmssssssnnnnnddddd"}(instruction)) {
            const unsigned shift = (instruction >> 10) & 63;
            if (bool((instruction >> 22) & 1) != wide || shift >= bits)
                return Result::Unsupported;
            Read(rm, rax, wide);
            Read(rn, rcx, wide);
            if (shift)
                shrd(accumulator, operand, shift);
            Write(rd, rax, wide);
            return Result::Continue;
        }
        return Result::Unsupported;
    }
    if (major == 5 || major == 13) {
        // Data processing: registers.
        if (Pattern{"zoo01010ssNmmmmmiiiiiinnnnnddddd"}(instruction)) {
            const unsigned shift = (instruction >> 22) & 3;
            const unsigned amount = (instruction >> 10) & 63;
            const unsigned opcode = (instruction >> 29) & 3;
            if (amount >= bits)
                return Result::Unsupported;
            Read(rn, rax, wide);
            Read(rm, rcx, wide);
            Shift(rcx, shift, amount, wide);
            if ((instruction >> 21) & 1)
                not_(operand);
            if (opcode == 0 || opcode == 3) and_(accumulator, operand);
            else if (opcode == 1) or_(accumulator, operand);
            else xor_(accumulator, operand);
            if (opcode == 3)
                SaveFlags(false);
            Write(rd, rax, wide);
            return Result::Continue;
        }
        const bool addsub_shift = Pattern{"zoo01011ss0mmmmmiiiiiinnnnnddddd"}(instruction);
        const bool addsub_extend = Pattern{"zoo01011001mmmmmxxxiiinnnnnddddd"}(instruction);
        if (addsub_shift || addsub_extend) {
            const unsigned shift = (instruction >> 22) & 3;
            const unsigned amount = (instruction >> 10) & (addsub_extend ? 7 : 63);
            const bool subtract = (instruction >> 30) & 1;
            const bool flags = (instruction >> 29) & 1;
            if ((addsub_shift && (shift == 3 || amount >= bits)) || (addsub_extend && amount > 4))
                return Result::Unsupported;
            Read(rn, rax, wide, addsub_extend);
            Read(rm, rcx, wide);
            if (addsub_extend)
                Extend(rcx, (instruction >> 13) & 7, wide);
            Shift(rcx, addsub_extend ? 0 : shift, amount, wide);
            if (subtract) sub(accumulator, operand); else add(accumulator, operand);
            if (flags)
                SaveFlags(subtract);
            Write(rd, rax, wide, addsub_extend && !flags);
            return Result::Continue;
        }
        if (Pattern{"zoo11010000mmmmm000000nnnnnddddd"}(instruction)) {
            const bool subtract = (instruction >> 30) & 1;
            const bool flags = (instruction >> 29) & 1;
            Read(rn, rax, wide);
            Read(rm, rcx, wide);
            bt(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], NZCV::x64_c_flag_bit);
            if (subtract) { cmc(); sbb(accumulator, operand); }
            else adc(accumulator, operand);
            if (flags)
                SaveFlags(subtract);
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if (Pattern{"zo011010100mmmmmcccc0onnnnnddddd"}(instruction)) {
            const unsigned condition = (instruction >> 12) & 15;
            const unsigned opcode = (((instruction >> 30) & 1) << 1) | ((instruction >> 10) & 1);
            Read(rn, rbx, wide);
            Read(rm, rdx, wide);
            const Xbyak::Reg selected = wide ? Xbyak::Reg(rdx) : Xbyak::Reg(edx);
            if (opcode == 1) inc(selected);
            else if (opcode == 2) not_(selected);
            else if (opcode == 3) neg(selected);
            LoadFlags(condition);
            Select(condition, rdx, rbx);
            Write(rd, rdx, wide);
            return Result::Continue;
        }
        if (Pattern{"zo111010010xxxxxccccI0nnnnn0ffff"}(instruction)) {
            const unsigned condition = (instruction >> 12) & 15;
            const bool subtract = (instruction >> 30) & 1;
            mov(ebp, 0);
            mov(edx, 1);
            LoadFlags(condition);
            Select(condition, rbp, rdx);
            Read(rn, rax, wide);
            if ((instruction >> 11) & 1) mov(ecx, rm);
            else Read(rm, rcx, wide);
            if (subtract) sub(accumulator, operand); else add(accumulator, operand);
            seto(al);
            lahf();
            and_(eax, NZCV::x64_mask);
            if (subtract)
                xor_(eax, 1u << NZCV::x64_c_flag_bit);
            mov(edx, NZCV::ToX64((instruction & 15) << 28));
            test(rbp, rbp);
            cmovz(eax, edx);
            mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], eax);
            return Result::Continue;
        }
        if (Pattern{"z0011010110mmmmm0010oonnnnnddddd"}(instruction)) {
            const unsigned kind = (instruction >> 10) & 3;
            Read(rn, rax, wide);
            Read(rm, rcx, wide);
            switch (kind) {
            case 0: shl(accumulator, cl); break;
            case 1: shr(accumulator, cl); break;
            case 2: sar(accumulator, cl); break;
            case 3: ror(accumulator, cl); break;
            }
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if (Pattern{"z0011011000mmmmmoaaaaannnnnddddd"}(instruction)) {
            Read(rn, rax, wide);
            Read(rm, rcx, wide);
            imul(accumulator, operand);
            Read((instruction >> 10) & 31, rcx, wide);
            if ((instruction >> 15) & 1) {
                sub(operand, accumulator);
                Write(rd, rcx, wide);
            } else {
                add(accumulator, operand);
                Write(rd, rax, wide);
            }
            return Result::Continue;
        }
        if (Pattern{"10011011u01mmmmmoaaaaannnnnddddd"}(instruction)) {
            const bool unsigned_multiply = (instruction >> 23) & 1;
            Read(rn, rax, false);
            Read(rm, rcx, false);
            if (!unsigned_multiply) {
                movsxd(rax, eax);
                movsxd(rcx, ecx);
            }
            imul(rax, rcx);
            Read((instruction >> 10) & 31, rcx);
            if ((instruction >> 15) & 1) {
                sub(rcx, rax);
                Write(rd, rcx);
            } else {
                add(rax, rcx);
                Write(rd, rax);
            }
            return Result::Continue;
        }
        if (Pattern{"10011011u10mmmmm011111nnnnnddddd"}(instruction)) {
            Read(rn, rax);
            Read(rm, rcx);
            if ((instruction >> 23) & 1) mul(rcx); else imul(rcx);
            Write(rd, rdx);
            return Result::Continue;
        }
        if (Pattern{"z0011010110mmmmm00001onnnnnddddd"}(instruction)) {
            const bool signed_divide = (instruction >> 10) & 1;
            Xbyak::Label zero, done, ordinary;
            Read(rn, rax, wide);
            Read(rm, rcx, wide);
            test(operand, operand);
            jz(zero, T_NEAR);
            if (signed_divide) {
                // Unlike x86 IDIV, ARM defines MIN / -1 as MIN, without a fault.
                cmp(operand, static_cast<u32>(-1));
                jne(ordinary, T_NEAR);
                mov(rdx, u64{1} << (bits - 1));
                cmp(accumulator, wide ? Xbyak::Reg(rdx) : Xbyak::Reg(edx));
                je(done, T_NEAR);
                L(ordinary);
                if (wide) cqo(); else cdq();
                idiv(operand);
            } else {
                xor_(edx, edx);
                div(operand);
            }
            jmp(done, T_NEAR);
            L(zero);
            xor_(eax, eax);
            L(done);
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if (Pattern{"z10110101100000000010onnnnnddddd"}(instruction)) {
            // CLZ / CLS, using baseline x86 BSR rather than optional LZCNT.
            const bool sign_count = (instruction >> 10) & 1;
            Read(rn, rax, wide);
            if (sign_count) {
                mov(rcx, rax);
                Shift(rcx, 2, bits - 1, wide);
                xor_(accumulator, operand);
            }
            mov(ecx, bits * 2 - 1);
            bsr(accumulator, accumulator);
            cmovz(rax, rcx);
            xor_(eax, bits - 1);
            if (sign_count)
                dec(eax);
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if (Pattern{"z1011010110000000000oonnnnnddddd"}(instruction)) {
            const unsigned opcode = (instruction >> 10) & 3;
            if (!wide && opcode == 3)
                return Result::Unsupported;
            Read(rn, rax, wide);
            if (opcode == 0) {  // RBIT: fixed masks; no host feature dependency.
                constexpr u64 masks[] = {0x5555555555555555ull, 0x3333333333333333ull, 0x0f0f0f0f0f0f0f0full};
                for (unsigned index = 0; index < 3; ++index) {
                    const unsigned shift = 1u << index;
                    mov(rcx, rax);
                    Shift(rcx, 1, shift, wide);
                    Mask(rcx, masks[index], wide);
                    Mask(rax, masks[index], wide);
                    Shift(rax, 0, shift, wide);
                    or_(accumulator, operand);
                }
                bswap(accumulator);
            } else if (opcode == 1) {
                mov(rcx, rax);
                Shift(rcx, 1, 8, wide);
                Mask(rcx, 0x00ff00ff00ff00ffull, wide);
                Mask(rax, 0x00ff00ff00ff00ffull, wide);
                Shift(rax, 0, 8, wide);
                or_(accumulator, operand);
            } else {
                bswap(accumulator);
                if (wide && opcode == 2)
                    ror(rax, 32);
            }
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if (BaselineSystemSupported(instruction)) {
            const bool terminal = BaselineSystemTerminal(instruction);
            EmitHelper(offsetof(A64JitState, baseline_system_instruction), instruction, terminal);
            return terminal ? Result::Terminal : Result::Continue;
        }
        return Result::Unsupported;
    }
    if (major == 7 || major == 15) {
        // SIMD operations with exact SSE2 equivalents need no FP environment or
        // feature-dependent lowering. All 64-bit vector writes clear the upper
        // half, including ORN whose bitwise inversion would otherwise set it.
        if (Pattern{"0Qo01110ss1mmmmm000111nnnnnddddd"}(instruction)) {
            const bool quad = (instruction >> 30) & 1;
            const bool upper_group = (instruction >> 29) & 1;
            const unsigned operation = (instruction >> 22) & 3;
            ReadVector(rn, xmm0, quad);
            ReadVector(rm, xmm1, quad);
            if (!upper_group) {
                switch (operation) {
                case 0: pand(xmm0, xmm1); break;
                case 1: pandn(xmm1, xmm0); movdqa(xmm0, xmm1); break;
                case 2: por(xmm0, xmm1); break;
                case 3:
                    pcmpeqd(xmm2, xmm2);
                    pxor(xmm1, xmm2);
                    por(xmm0, xmm1);
                    break;
                }
            } else if (!operation) {
                pxor(xmm0, xmm1);
            } else {
                ReadVector(rd, xmm2, quad);
                if (operation == 1) {  // BSL: old Vd selects bits from Vn / Vm.
                    pxor(xmm0, xmm1);
                    pand(xmm0, xmm2);
                    pxor(xmm0, xmm1);
                } else if (operation == 2) {  // BIT.
                    pxor(xmm0, xmm2);
                    pand(xmm0, xmm1);
                    pxor(xmm0, xmm2);
                } else {  // BIF.
                    pxor(xmm0, xmm2);
                    pandn(xmm1, xmm0);
                    pxor(xmm1, xmm2);
                    movdqa(xmm0, xmm1);
                }
            }
            WriteVector(rd, xmm0, quad);
            return Result::Continue;
        }
        if (Pattern{"0Qo01110zz1mmmmm100001nnnnnddddd"}(instruction)) {
            const bool quad = (instruction >> 30) & 1;
            const bool subtract = (instruction >> 29) & 1;
            const unsigned element = (instruction >> 22) & 3;
            if (!quad && element == 3)
                return Result::Unsupported;
            ReadVector(rn, xmm0, quad);
            ReadVector(rm, xmm1, quad);
            if (subtract) {
                switch (element) {
                case 0: psubb(xmm0, xmm1); break;
                case 1: psubw(xmm0, xmm1); break;
                case 2: psubd(xmm0, xmm1); break;
                case 3: psubq(xmm0, xmm1); break;
                }
            } else {
                switch (element) {
                case 0: paddb(xmm0, xmm1); break;
                case 1: paddw(xmm0, xmm1); break;
                case 2: paddd(xmm0, xmm1); break;
                case 3: paddq(xmm0, xmm1); break;
                }
            }
            WriteVector(rd, xmm0, quad);
            return Result::Continue;
        }
        if ((instruction & 0xfffffc00u) == 0x1e260000u ||
            (instruction & 0xfffffc00u) == 0x9e660000u ||
            (instruction & 0xfffffc00u) == 0x9eae0000u) {
            const bool upper = (instruction & 0xfffffc00u) == 0x9eae0000u;
            ReadVector(rn, xmm0);
            if (upper) psrldq(xmm0, 8);
            if (wide) movq(rax, xmm0); else movd(eax, xmm0);
            Write(rd, rax, wide);
            return Result::Continue;
        }
        if ((instruction & 0xfffffc00u) == 0x1e270000u ||
            (instruction & 0xfffffc00u) == 0x9e670000u ||
            (instruction & 0xfffffc00u) == 0x9eaf0000u) {
            Read(rn, rax, wide);
            if ((instruction & 0xfffffc00u) == 0x9eaf0000u) {
                ReadVector(rd, xmm0);
                movq(xmm1, rax);
                punpcklqdq(xmm0, xmm1);
                WriteVector(rd, xmm0);
            } else {
                if (wide) movq(xmm0, rax); else movd(xmm0, eax);
                WriteVector(rd, xmm0);
            }
            return Result::Continue;
        }
        if ((instruction & 0xfffffc00u) == 0x1e204000u ||
            (instruction & 0xfffffc00u) == 0x1e604000u) {
            ReadVector(rn, xmm0, false);
            if (!((instruction >> 22) & 1)) { movd(eax, xmm0); movd(xmm0, eax); }
            WriteVector(rd, xmm0);
            return Result::Continue;
        }

        // Decode remaining FP/SIMD and memory families with independent bounded
        // emitters. Rare operations call precompiled semantics, never an IR JIT.
        if (const auto result = SimdInstruction(instruction); result != Result::Unsupported)
            return result;
        if (BaselineCryptoSupported(instruction)) {
            EmitHelper(offsetof(A64JitState, baseline_crypto_instruction), instruction);
            return Result::Continue;
        }
        return Result::Unsupported;
    }
    if (major == 4 || major == 6 || major == 12 || major == 14) {
        if (const auto result = MemoryInstruction(instruction); result != Result::Unsupported)
            return result;

        // Scalar GPR memory templates preserve both register caches on the fast
        // path; fault-only veneers materialize exact state before invoking callbacks.
        if (Pattern{"zz111001ooiiiiiiiiiiiinnnnnttttt"}(instruction))
            return LoadStore(instruction, false, true);
        if (Pattern{"zz111000oo0iiiiiiiiip1nnnnnttttt"}(instruction) ||
            Pattern{"zz111000oo0iiiiiiiii00nnnnnttttt"}(instruction) ||
            Pattern{"zz111000oo0iiiiiiiii10nnnnnttttt"}(instruction))
            return LoadStore(instruction, false, false);
        if (Pattern{"zz111000oo1mmmmmxxxS10nnnnnttttt"}(instruction))
            return LoadStore(instruction, true, false);
        if (Pattern{"oo10100pwLiiiiiiiuuuuunnnnnttttt"}(instruction))
            return LoadStorePair(instruction);
        if (Pattern{"oo011000iiiiiiiiiiiiiiiiiiittttt"}(instruction)) {
            const unsigned opcode = instruction >> 30;
            if (opcode == 3)
                return Result::Continue;  // PRFM literal.
            const s64 delta = SignExtend((instruction >> 5) & 0x7ffff, 19) * 4;
            lea(rbx, ptr[r12 + offset]);
            AddOffset(rbx, delta);
            MemoryRead(opcode == 1 ? 8 : 4);
            ExtendLoad(opcode == 1 ? 8 : 4, opcode == 2, opcode != 0);
            Write(rd, rax, opcode != 0);
            return Result::Continue;
        }
        return Result::Unsupported;
    }
    if (major == 10 || major == 11) {
        // All branch destinations remain relative to the runtime entry PC.
        if (Pattern{"o00101iiiiiiiiiiiiiiiiiiiiiiiiii"}(instruction)) {
            const s64 delta = SignExtend(instruction & 0x03ffffff, 26) * 4;
            if (wide) {
                lea(rax, ptr[r12 + offset + 4]);
                Write(30, rax);
            }
            Flush();
            lea(rax, ptr[r12 + offset]);
            AddOffset(rax, delta);
            mov(qword[r15 + offsetof(A64JitState, pc)], rax);
            BranchReturn(offset / 4 + 1, s64(offset) + delta == 0);
            return Result::Terminal;
        }
        if (Pattern{"01010100iiiiiiiiiiiiiiiiiii0cccc"}(instruction)) {
            const s64 delta = SignExtend((instruction >> 5) & 0x7ffff, 19) * 4;
            Flush();
            lea(rcx, ptr[r12 + offset + delta]);
            lea(rdx, ptr[r12 + offset + 4]);
            LoadFlags(rd & 15);
            Select(rd & 15, rdx, rcx);
            mov(qword[r15 + offsetof(A64JitState, pc)], rdx);
            BranchReturn(offset / 4 + 1, s64(offset) + delta == 0);
            return Result::Terminal;
        }
        if (Pattern{"z011010oiiiiiiiiiiiiiiiiiiittttt"}(instruction)) {
            const bool nonzero = (instruction >> 24) & 1;
            const s64 delta = SignExtend((instruction >> 5) & 0x7ffff, 19) * 4;
            Read(rd, rax, wide);
            Flush();
            lea(rcx, ptr[r12 + offset + delta]);
            lea(rdx, ptr[r12 + offset + 4]);
            test(accumulator, accumulator);
            if (nonzero) cmovnz(rdx, rcx); else cmovz(rdx, rcx);
            mov(qword[r15 + offsetof(A64JitState, pc)], rdx);
            BranchReturn(offset / 4 + 1, s64(offset) + delta == 0);
            return Result::Terminal;
        }
        if (Pattern{"b011011obbbbbiiiiiiiiiiiiiittttt"}(instruction)) {
            const unsigned bit = ((instruction >> 26) & 32) | ((instruction >> 19) & 31);
            const bool nonzero = (instruction >> 24) & 1;
            const s64 delta = SignExtend((instruction >> 5) & 0x3fff, 14) * 4;
            Read(rd, rax, wide);
            Flush();
            lea(rcx, ptr[r12 + offset + delta]);
            lea(rdx, ptr[r12 + offset + 4]);
            bt(accumulator, bit);
            if (nonzero) cmovc(rdx, rcx); else cmovnc(rdx, rcx);
            mov(qword[r15 + offsetof(A64JitState, pc)], rdx);
            BranchReturn(offset / 4 + 1, s64(offset) + delta == 0);
            return Result::Terminal;
        }
        if (Pattern{"1101011000011111000000nnnnn00000"}(instruction) ||
            Pattern{"1101011000111111000000nnnnn00000"}(instruction) ||
            Pattern{"1101011001011111000000nnnnn00000"}(instruction)) {
            Read(rn, rdx);
            if (((instruction >> 21) & 3) == 1) {
                lea(rax, ptr[r12 + offset + 4]);
                Write(30, rax);
            }
            Flush();
            mov(qword[r15 + offsetof(A64JitState, pc)], rdx);
            BranchReturn(offset / 4 + 1);
            return Result::Terminal;
        }

        // Common system-register transfers use the current CPU context. No TLS
        // pointer belonging to the compiling thread is embedded into machine code.
        if ((instruction & ~31u) == 0xd53bd040u && config.tpidr_el0) {
            Xbyak::Label absent;
            mov(rax, qword[r15 + offsetof(A64JitState, runtime_tpidr_el0)]);
            test(rax, rax);
            jz(absent, T_NEAR);
            mov(rax, qword[rax]);
            L(absent);
            Write(rd, rax);
            return Result::Continue;
        }
        if ((instruction & ~31u) == 0xd53bd060u && config.tpidrro_el0) {
            Xbyak::Label absent;
            mov(rax, qword[r15 + offsetof(A64JitState, runtime_tpidrro_el0)]);
            test(rax, rax);
            jz(absent, T_NEAR);
            mov(rax, qword[rax]);
            L(absent);
            Write(rd, rax);
            return Result::Continue;
        }
        if ((instruction & ~31u) == 0xd51bd040u && config.tpidr_el0) {
            Xbyak::Label absent;
            Read(rd, rax);
            mov(rcx, qword[r15 + offsetof(A64JitState, runtime_tpidr_el0)]);
            test(rcx, rcx);
            jz(absent, T_NEAR);
            mov(qword[rcx], rax);
            L(absent);
            return Result::Continue;
        }
        if ((instruction & ~31u) == 0xd53b4200u) {
            mov(eax, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);
            and_(eax, NZCV::x64_mask);
            imul(eax, eax, NZCV::from_x64_multiplier);
            and_(eax, NZCV::arm_mask);
            Write(rd, rax);
            return Result::Continue;
        }
        if ((instruction & ~31u) == 0xd51b4200u) {
            Read(rd, rax, false);
            shr(eax, 28);
            imul(eax, eax, NZCV::to_x64_multiplier);
            and_(eax, NZCV::x64_mask);
            mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], eax);
            return Result::Continue;
        }
        if ((instruction & 0xfffff01fu) == 0xd503201fu) {
            const unsigned hint = (instruction >> 5) & 127;
            if (!config.hook_hint_instructions || hint == 0 || hint > 5)
                return Result::Continue;
        }
        if ((instruction & 0xfffff0ffu) == 0xd503309fu ||
            (instruction & 0xfffff0ffu) == 0xd50330bfu) {
            mfence();
            return Result::Continue;
        }
        if (BaselineSystemSupported(instruction)) {
            const bool terminal = BaselineSystemTerminal(instruction);
            EmitHelper(offsetof(A64JitState, baseline_system_instruction), instruction, terminal);
            return terminal ? Result::Terminal : Result::Continue;
        }
    }
    return Result::Unsupported;
}

}  // namespace

std::optional<A64BaselineBlock> CompileA64Baseline(
    u64 pc, const BaselineReadCode& get_code,
    const A64::UserConfig& config, u32 max_instructions, bool single_step) {
    if ((pc & 3) || !get_code)
        return std::nullopt;
    const u32 limit = single_step ? 1 : std::clamp(max_instructions, 1u, 256u);
    // Reuse non-executable scratch and avoid probing 64+ KiB of the native stack
    // on every translation. A recursive code-read callback gets separate scratch.
    constexpr std::size_t capacity = 256 * 2048;
    struct Workspace { std::array<u8, capacity> bytes; bool busy = false; };
    thread_local Workspace workspace;
    const bool recursive = std::exchange(workspace.busy, true);
    struct Release { bool& busy; bool previous; ~Release() { busy = previous; } } release{workspace.busy, recursive};
    auto nested = recursive ? std::unique_ptr<std::array<u8, capacity>>(new std::array<u8, capacity>) : nullptr;
    auto& buffer = nested ? *nested : workspace.bytes;
    Emitter emitter{buffer.data(), buffer.size(), config, single_step};
    A64BaselineBlock result;
    bool terminal = false;
    for (u32 index = 0; index < limit; ++index) {
        const u32 offset = index * 4;
        if (pc > std::numeric_limits<u64>::max() - offset)
            break;
        const auto instruction = get_code(pc + offset);
        if (!instruction) {
            if (!index) {
                emitter.Undefined(0, offset, true);
                result.instruction_count = 1;
                result.guest_size = 4;
                terminal = true;
            }
            break;
        }
        // Cycle-based counter callbacks require an exact block boundary before
        // the read. Wall-clock configurations need no such split.
        if (index && !config.wall_clock_cntpct &&
            ((*instruction & ~31u) == 0xd53be020u || (*instruction & ~31u) == 0xd53be040u))
            break;
        auto emitted = emitter.Instruction(*instruction, offset);
        if (emitted == Emitter::Result::Unsupported) {
            result.stopped_at_unsupported = true;
            emitter.Undefined(*instruction, offset);
            emitted = Emitter::Result::Terminal;
        }
        ++result.instruction_count;
        result.guest_size += 4;
        if (emitted == Emitter::Result::Terminal) {
            terminal = true;
            break;
        }
    }
    if (!result.instruction_count)
        return std::nullopt;
    if (!terminal)
        emitter.Finish(result.guest_size, result.instruction_count);
    emitter.ready();
    result.code.assign(emitter.getCode(), emitter.getCurr());
    emitter.CopyFaults(result);
    return result;
}

void InitializeA64BaselineState(A64JitState& state) {
    state.baseline_memory_read = &ReadMemory;
    state.baseline_memory_write = &WriteMemory;
    state.baseline_memory_instruction = &ExecuteBaselineMemory;
    state.baseline_simd_instruction = &ExecuteBaselineSimd;
    state.baseline_system_instruction = &ExecuteBaselineSystem;
    state.baseline_crypto_instruction = &ExecuteBaselineCrypto;
    state.baseline_undefined_instruction = &ExecuteBaselineUndefined;
    state.baseline_no_execute = &ExecuteBaselineNoExecute;
}

}  // namespace Dynarmic::Backend::X64
