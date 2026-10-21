// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
//
// Independent A64 memory semantics for the low-latency compiler. Common SIMD
// loads/stores use native templates; these routines are also their precise
// fault veneers. Uncommon structures and atomics are decoded in bounded time
// and executed without ever entering the optimizing frontend or JIT.

#include "dynarmic/backend/x64/a64_baseline_memory.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <type_traits>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic::Backend::X64 {
namespace {

constexpr s64 SignedImmediate(u64 value, unsigned bits) {
    const u64 sign = u64{1} << (bits - 1);
    return static_cast<s64>((value ^ sign) - sign);
}

bool Matches(u32 instruction, u32 mask, u32 value) {
    return (instruction & mask) == value;
}

u64 X(const A64JitState& state, unsigned reg) {
    return reg == 31 ? 0 : state.reg[reg];
}

u64 Base(const A64JitState& state, unsigned reg) {
    return reg == 31 ? state.sp : state.reg[reg];
}

void SetX(A64JitState& state, unsigned reg, u64 value, unsigned bytes) {
    if (reg != 31)
        state.reg[reg] = bytes < 8 ? static_cast<u32>(value) : value;
}

void SetBase(A64JitState& state, unsigned reg, u64 value) {
    if (reg == 31)
        state.sp = value;
    else
        state.reg[reg] = value;
}

bool Aborted(A64JitState* state) {
    return (Atomic::Load(&state->halt_reason) & static_cast<u32>(HaltReason::MemoryAbort)) != 0;
}

void ClearReservation(A64JitState* state) {
    state->exclusive_state = 0;
    state->baseline_exclusive_size = 0;
    const auto& config = *state->runtime_config;
    if (config.global_monitor)
        config.global_monitor->ClearProcessor(config.processor_id);
}

bool Alignment(A64JitState* state, u64 address, unsigned bytes) {
    if ((address & (bytes - 1)) == 0)
        return true;
    ClearReservation(state);
    state->runtime_callbacks->ExceptionRaised(state->pc, A64::Exception::AlignmentFault);
    Atomic::Or(&state->halt_reason, static_cast<u32>(HaltReason::MemoryAbort));
    return false;
}

bool BadInstruction(A64JitState* state) {
    state->runtime_callbacks->ExceptionRaised(state->pc, A64::Exception::UnallocatedEncoding);
    Atomic::Or(&state->halt_reason, static_cast<u32>(HaltReason::MemoryAbort));
    return false;
}

// Full fences conservatively implement RCsc, RCpc, and limited ordering on
// both host ABIs, including callbacks which use ordinary C++ memory accesses.
// The native templates only elide fences where x86 TSO already supplies them.
void Fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

template<class T>
T ReadValue(A64JitState* state, u64 address) {
    auto& callbacks = *state->runtime_callbacks;
    if constexpr (std::is_same_v<T, u8>) return callbacks.MemoryRead8(address);
    else if constexpr (std::is_same_v<T, u16>) return callbacks.MemoryRead16(address);
    else if constexpr (std::is_same_v<T, u32>) return callbacks.MemoryRead32(address);
    else if constexpr (std::is_same_v<T, u64>) return callbacks.MemoryRead64(address);
    else return callbacks.MemoryRead128(address);
}

template<class T>
void WriteValue(A64JitState* state, u64 address, T value) {
    auto& callbacks = *state->runtime_callbacks;
    if constexpr (std::is_same_v<T, u8>) callbacks.MemoryWrite8(address, value);
    else if constexpr (std::is_same_v<T, u16>) callbacks.MemoryWrite16(address, value);
    else if constexpr (std::is_same_v<T, u32>) callbacks.MemoryWrite32(address, value);
    else if constexpr (std::is_same_v<T, u64>) callbacks.MemoryWrite64(address, value);
    else callbacks.MemoryWrite128(address, value);
}

template<class T>
bool CompareExchange(A64JitState* state, u64 address, T desired, T expected) {
    auto& callbacks = *state->runtime_callbacks;
    if constexpr (std::is_same_v<T, u8>) return callbacks.MemoryWriteExclusive8(address, desired, expected);
    else if constexpr (std::is_same_v<T, u16>) return callbacks.MemoryWriteExclusive16(address, desired, expected);
    else if constexpr (std::is_same_v<T, u32>) return callbacks.MemoryWriteExclusive32(address, desired, expected);
    else if constexpr (std::is_same_v<T, u64>) return callbacks.MemoryWriteExclusive64(address, desired, expected);
    else return callbacks.MemoryWriteExclusive128(address, desired, expected);
}

u64 ReadScalar(A64JitState* state, u64 address, unsigned bytes) {
    switch (bytes) {
    case 1: return ReadValue<u8>(state, address);
    case 2: return ReadValue<u16>(state, address);
    case 4: return ReadValue<u32>(state, address);
    case 8: return ReadValue<u64>(state, address);
    default: return 0;  // All callers have a validated scalar width.
    }
}

void WriteScalar(A64JitState* state, u64 address, u64 value, unsigned bytes) {
    switch (bytes) {
    case 1: WriteValue(state, address, static_cast<u8>(value)); break;
    case 2: WriteValue(state, address, static_cast<u16>(value)); break;
    case 4: WriteValue(state, address, static_cast<u32>(value)); break;
    case 8: WriteValue(state, address, value); break;
    }
}

A64::Vector ReadVectorMemory(A64JitState* state, u64 address, unsigned bytes) {
    if (bytes == 16)
        return ReadValue<A64::Vector>(state, address);
    return {ReadScalar(state, address, bytes), 0};
}

void WriteVectorMemory(A64JitState* state, u64 address, const A64::Vector& value, unsigned bytes) {
    if (bytes == 16)
        WriteValue(state, address, value);
    else
        WriteScalar(state, address, value[0], bytes);
}

A64::Vector VectorRegister(const A64JitState& state, unsigned reg) {
    return {state.vec[reg * 2], state.vec[reg * 2 + 1]};
}

void SetVector(A64JitState& state, unsigned reg, const A64::Vector& value) {
    state.vec[reg * 2] = value[0];
    state.vec[reg * 2 + 1] = value[1];
}

u64 GetLane(const A64::Vector& value, unsigned lane, unsigned bytes) {
    const unsigned position = lane * bytes;
    const u64 shifted = value[position / 8] >> ((position % 8) * 8);
    return bytes == 8 ? shifted : shifted & ((u64{1} << (bytes * 8)) - 1);
}

void SetLane(A64::Vector& value, unsigned lane, unsigned bytes, u64 element) {
    const unsigned position = lane * bytes;
    const unsigned shift = (position % 8) * 8;
    const u64 mask = bytes == 8 ? ~u64{0} : (u64{1} << (bytes * 8)) - 1;
    value[position / 8] = (value[position / 8] & ~(mask << shift)) | ((element & mask) << shift);
}

u64 EffectiveAddress(const A64JitState& state, const BaselineMemoryOp& op) {
    using Address = BaselineMemoryOp::Address;
    if (op.address == Address::Literal)
        return state.pc + static_cast<u64>(op.displacement);
    u64 address = Base(state, op.rn);
    if (op.address == Address::RegisterOffset) {
        u64 offset = X(state, op.rm);
        switch (op.option) {
        case 2: offset = static_cast<u32>(offset); break;  // UXTW
        case 3: break;                                  // LSL/UXTX
        case 6: offset = static_cast<s32>(offset); break; // SXTW
        case 7: break;                                  // SXTX
        }
        return address + (offset << op.shift);
    }
    if (op.address != Address::PostIndex)
        address += static_cast<u64>(op.displacement);
    return address;
}

bool SimdMemory(A64JitState* state, const BaselineMemoryOp& op, unsigned phase) {
    const unsigned count = op.pair ? 2 : 1;
    if (phase >= count)
        return BadInstruction(state);
    const u64 address = EffectiveAddress(*state, op);
    const u64 writeback = Base(*state, op.rn) + static_cast<u64>(op.displacement);
    std::array<A64::Vector, 2> values{};
    if (op.load) {
        for (unsigned n = 0; n < phase; ++n)
            values[n] = {state->baseline_memory_saved[n * 2], state->baseline_memory_saved[n * 2 + 1]};
        for (unsigned n = phase; n < count; ++n) {
            values[n] = ReadVectorMemory(state, address + n * op.bytes, op.bytes);
            if (Aborted(state))
                return false;
        }
        SetVector(*state, op.rt, values[0]);
        if (op.pair)
            SetVector(*state, op.rt2, values[1]);
    } else {
        values[0] = VectorRegister(*state, op.rt);
        if (op.pair)
            values[1] = VectorRegister(*state, op.rt2);
        for (unsigned n = phase; n < count; ++n) {
            WriteVectorMemory(state, address + n * op.bytes, values[n], op.bytes);
            if (Aborted(state))
                return false;
        }
    }
    if (op.Writeback())
        SetBase(*state, op.rn, writeback);
    return true;
}

bool StructureMultiple(A64JitState* state, const BaselineMemoryOp& op, unsigned phase) {
    const unsigned vector_bytes = op.quad ? 16 : 8;
    const unsigned count = op.structures * op.repetitions;
    if (phase >= count || (phase && op.structures != 1))
        return BadInstruction(state);
    const u64 address = Base(*state, op.rn);
    const u64 increment = op.rm == 31 ? vector_bytes * count : X(*state, op.rm);
    std::array<A64::Vector, 4> values{};
    if (!op.load) {
        for (unsigned n = 0; n < count; ++n)
            values[n] = VectorRegister(*state, (op.rt + n) % 32);
    }

    if (op.structures == 1) {
        // LD1/ST1 transfer whole registers. This also matches the callback
        // widths of the native templates, including a resumed partial store.
        for (unsigned n = 0; n < phase; ++n)
            values[n] = {state->baseline_memory_saved[n * 2], state->baseline_memory_saved[n * 2 + 1]};
        for (unsigned n = phase; n < count; ++n) {
            if (op.load)
                values[n] = ReadVectorMemory(state, address + n * vector_bytes, vector_bytes);
            else
                WriteVectorMemory(state, address + n * vector_bytes, values[n], vector_bytes);
            if (Aborted(state))
                return false;
        }
    } else {
        // Interleaved LD2/LD3/LD4 use the architectural element accesses. Do
        // not coalesce MMIO callbacks or access beyond the specified elements.
        u64 current = address;
        for (unsigned lane = 0; lane < vector_bytes / op.bytes; ++lane) {
            for (unsigned structure = 0; structure < op.structures; ++structure) {
                if (op.load) {
                    const u64 element = ReadScalar(state, current, op.bytes);
                    if (Aborted(state))
                        return false;
                    SetLane(values[structure], lane, op.bytes, element);
                } else {
                    WriteScalar(state, current, GetLane(values[structure], lane, op.bytes), op.bytes);
                    if (Aborted(state))
                        return false;
                }
                current += op.bytes;
            }
        }
    }
    if (op.load) {
        for (unsigned n = 0; n < count; ++n) {
            if (!op.quad)
                values[n][1] = 0;
            SetVector(*state, (op.rt + n) % 32, values[n]);
        }
    }
    if (op.Writeback())
        SetBase(*state, op.rn, address + increment);
    return true;
}

bool StructureSingle(A64JitState* state, const BaselineMemoryOp& op) {
    const u64 address = Base(*state, op.rn);
    const u64 increment = op.rm == 31 ? op.bytes * op.structures : X(*state, op.rm);
    std::array<A64::Vector, 4> values{};
    for (unsigned n = 0; n < op.structures; ++n)
        values[n] = VectorRegister(*state, (op.rt + n) % 32);
    for (unsigned n = 0; n < op.structures; ++n) {
        const u64 current = address + n * op.bytes;
        if (op.load) {
            const u64 element = ReadScalar(state, current, op.bytes);
            if (Aborted(state))
                return false;
            if (op.replicate) {
                values[n] = {};
                for (unsigned lane = 0; lane < (op.quad ? 16u : 8u) / op.bytes; ++lane)
                    SetLane(values[n], lane, op.bytes, element);
            } else {
                // A lane load preserves every other lane, including the high
                // half when Q is zero. Q is an index bit for these encodings.
                SetLane(values[n], op.lane, op.bytes, element);
            }
        } else {
            WriteScalar(state, current, GetLane(values[n], op.lane, op.bytes), op.bytes);
            if (Aborted(state))
                return false;
        }
    }
    if (op.load)
        for (unsigned n = 0; n < op.structures; ++n)
            SetVector(*state, (op.rt + n) % 32, values[n]);
    if (op.Writeback())
        SetBase(*state, op.rn, address + increment);
    return true;
}

template<class T>
bool Exclusive(A64JitState* state, const BaselineMemoryOp& op, u64 address) {
    const auto& config = *state->runtime_config;
    if (op.load) {
        state->exclusive_state = 0;
        const T value = config.global_monitor
            ? config.global_monitor->ReadAndMark<T>(config.processor_id, address, [&] { return ReadValue<T>(state, address); })
            : ReadValue<T>(state, address);
        if (Aborted(state)) {
            ClearReservation(state);
            return false;
        }
        state->baseline_exclusive_address = address;
        state->baseline_exclusive_value = {};
        std::memcpy(state->baseline_exclusive_value.data(), &value, sizeof(value));
        state->baseline_exclusive_size = sizeof(T);
        state->exclusive_state = 1;
        if (op.acquire)
            Fence();
        if constexpr (std::is_same_v<T, A64::Vector>) {
            SetX(*state, op.rt, value[0], 8);
            SetX(*state, op.rt2, value[1], 8);
        } else if (op.pair) {
            SetX(*state, op.rt, static_cast<u32>(value), 4);
            SetX(*state, op.rt2, static_cast<u64>(value) >> 32, 4);
        } else {
            SetX(*state, op.rt, value, sizeof(T));
        }
        return true;
    }

    const bool reserved = state->exclusive_state != 0 &&
        state->baseline_exclusive_address == address && state->baseline_exclusive_size == sizeof(T);
    state->exclusive_state = 0;
    state->baseline_exclusive_size = 0;
    if (op.release)
        Fence();
    if (!reserved) {
        // The cleared local bit is sufficient to prevent another STXR from
        // using this reservation. Its next LDXR replaces the monitor slot;
        // avoid ClearProcessor's all-bucket sweep on a routine failed store.
        SetX(*state, op.rs, 1, 4);
        return true;
    }
    T value;
    if constexpr (std::is_same_v<T, A64::Vector>) {
        value = {X(*state, op.rt), X(*state, op.rt2)};
    } else if (op.pair) {
        value = static_cast<T>(static_cast<u32>(X(*state, op.rt)) |
            (static_cast<u64>(static_cast<u32>(X(*state, op.rt2))) << 32));
    } else {
        value = static_cast<T>(X(*state, op.rt));
    }
    bool success;
    if (config.global_monitor) {
        success = config.global_monitor->DoExclusiveOperation<T>(config.processor_id, address,
            [&](T expected) { return CompareExchange(state, address, value, expected); });
    } else {
        T expected;
        std::memcpy(&expected, state->baseline_exclusive_value.data(), sizeof(expected));
        success = CompareExchange(state, address, value, expected);
    }
    if (Aborted(state))
        return false;
    SetX(*state, op.rs, success ? 0 : 1, 4);
    return true;
}

template<class T, class Modify>
bool AtomicModify(A64JitState* state, u64 address, bool acquire, bool release,
                  Modify&& modify, T& observed) {
    const auto& config = *state->runtime_config;
    // ReadAndMark below replaces this processor's previous monitor slot.
    // Clearing the local reservation is sufficient here; ClearProcessor on
    // the sharded monitor otherwise locks thousands of unrelated buckets.
    state->exclusive_state = 0;
    state->baseline_exclusive_size = 0;
    if (release)
        Fence();
    // The callback is a real compare-exchange, not a non-atomic read/write
    // pair. The global monitor additionally invalidates other reservations,
    // including same-value updates and ABA through other atomic instructions.
    for (;;) {
        if (Atomic::Load(&state->halt_reason) != 0) {
            ClearReservation(state);
            return false;
        }
        const T expected = config.global_monitor
            ? config.global_monitor->ReadAndMark<T>(config.processor_id, address, [&] { return ReadValue<T>(state, address); })
            : ReadValue<T>(state, address);
        if (Aborted(state)) {
            ClearReservation(state);
            return false;
        }
        const T desired = modify(expected);
        const bool success = config.global_monitor
            ? config.global_monitor->DoExclusiveOperation<T>(config.processor_id, address,
                  [&](T saved) { return CompareExchange(state, address, desired, saved); })
            : CompareExchange(state, address, desired, expected);
        if (Aborted(state)) {
            ClearReservation(state);
            return false;
        }
        if (success) {
            observed = expected;
            if (acquire)
                Fence();
            return true;
        }
    }
}

template<class T>
bool AtomicScalar(A64JitState* state, const BaselineMemoryOp& op, u64 address) {
    const T operand = static_cast<T>(X(*state, op.rs));
    const T replacement = static_cast<T>(X(*state, op.rt));
    const bool compare = op.kind == BaselineMemoryOp::Kind::AtomicCAS;
    T old{};
    const auto modify = [&](T value) -> T {
        if (compare)
            return value == operand ? replacement : value;
        using Signed = std::make_signed_t<T>;
        switch (op.operation) {
        case 0: return static_cast<T>(value + operand);                 // LDADD/STADD
        case 1: return static_cast<T>(value & ~operand);                // LDCLR/STCLR
        case 2: return static_cast<T>(value ^ operand);                 // LDEOR/STEOR
        case 3: return static_cast<T>(value | operand);                 // LDSET/STSET
        case 4: return std::bit_cast<Signed>(value) > std::bit_cast<Signed>(operand) ? value : operand;
        case 5: return std::bit_cast<Signed>(value) < std::bit_cast<Signed>(operand) ? value : operand;
        case 6: return value > operand ? value : operand;               // LDUMAX/STUMAX
        case 7: return value < operand ? value : operand;               // LDUMIN/STUMIN
        case 8: return operand;                                        // SWP
        default: return value;  // The decoder rejects every other operation.
        }
    };
    if (!AtomicModify(state, address, op.acquire, op.release, modify, old))
        return false;
    SetX(*state, compare ? op.rs : op.rt, old, sizeof(T));
    return true;
}

template<class T>
bool AtomicPair(A64JitState* state, const BaselineMemoryOp& op, u64 address) {
    T compare, replacement;
    if constexpr (std::is_same_v<T, A64::Vector>) {
        compare = {X(*state, op.rs), X(*state, op.rs + 1)};
        replacement = {X(*state, op.rt), X(*state, op.rt + 1)};
    } else {
        compare = static_cast<u32>(X(*state, op.rs)) |
            (static_cast<u64>(static_cast<u32>(X(*state, op.rs + 1))) << 32);
        replacement = static_cast<u32>(X(*state, op.rt)) |
            (static_cast<u64>(static_cast<u32>(X(*state, op.rt + 1))) << 32);
    }
    T old{};
    if (!AtomicModify(state, address, op.acquire, op.release,
                      [&](T value) { return value == compare ? replacement : value; }, old))
        return false;
    if constexpr (std::is_same_v<T, A64::Vector>) {
        SetX(*state, op.rs, old[0], 8);
        SetX(*state, op.rs + 1, old[1], 8);
    } else {
        SetX(*state, op.rs, static_cast<u32>(old), 4);
        SetX(*state, op.rs + 1, old >> 32, 4);
    }
    return true;
}

}  // namespace

bool DecodeBaselineMemory(u32 instruction, BaselineMemoryOp& op) noexcept {
    using Kind = BaselineMemoryOp::Kind;
    using Address = BaselineMemoryOp::Address;
    op = {};
    op.rt = instruction & 31;
    op.rn = (instruction >> 5) & 31;
    op.rt2 = (instruction >> 10) & 31;
    op.rm = op.rs = (instruction >> 16) & 31;
    op.load = ((instruction >> 22) & 1) != 0;

    if (Matches(instruction, 0x3f000000, 0x08000000)) {
        const unsigned size = instruction >> 30;
        op.bytes = 1u << size;
        if ((instruction >> 21) & 1) {
            if ((instruction >> 23) & 1) {
                if (op.rt2 != 31)
                    return false;
                op.kind = Kind::AtomicCAS;
                op.acquire = op.load;
                op.release = (instruction >> 15) & 1;
                return true;
            }
            if (size < 2) {
                if (op.rt2 != 31 || (op.rs & 1) || (op.rt & 1))
                    return false;
                op.kind = Kind::AtomicCASP;
                op.bytes = 8u << size;
                op.acquire = op.load;
                op.release = (instruction >> 15) & 1;
                op.pair = true;
                return true;
            }
            op.pair = true;
            op.bytes *= 2;
        } else if (op.rt2 != 31) {
            return false;
        }
        if ((instruction >> 23) & 1) {
            if (op.rs != 31)
                return false;
            op.kind = Kind::Ordered;
            op.acquire = op.load;
            op.release = !op.load;
            return true;  // LDAR/LDLAR/STLR/STLLR, including byte/halfword.
        }
        if ((op.load && (op.rs != 31 || (op.pair && op.rt == op.rt2))) ||
            (!op.load && (op.rs == op.rt || (op.pair && op.rs == op.rt2) ||
                          (op.rs == op.rn && op.rn != 31))))
            return false;
        op.kind = Kind::Exclusive;
        op.acquire = op.load && ((instruction >> 15) & 1);
        op.release = !op.load && ((instruction >> 15) & 1);
        return true;
    }

    // The encodings are disjoint from register-offset and immediate accesses.
    if (Matches(instruction, 0x3f200c00, 0x38200000)) {
        op.bytes = 1u << (instruction >> 30);
        op.operation = (instruction >> 12) & 15;
        if (op.operation == 12 && Matches(instruction, 0x3ffffc00, 0x38bfc000)) {
            op.kind = Kind::Ordered;  // LDAPR/LDAPRB/LDAPRH (RCpc acquire).
            op.load = op.acquire = true;
            return true;
        }
        if (op.operation > 8)
            return false;
        op.kind = Kind::AtomicRmw;
        op.acquire = ((instruction >> 23) & 1) && op.rt != 31;
        op.release = (instruction >> 22) & 1;
        return true;
    }

    // FEAT_LRCPC2: LDAPUR/STLUR, including zero-extending byte/halfword
    // loads and signed byte/halfword/word loads. The signed imm9 is a byte
    // offset, never a scaled displacement or a request for base writeback.
    if (Matches(instruction, 0x3f200c00, 0x19000000)) {
        const unsigned size = instruction >> 30;
        const unsigned opcode = (instruction >> 22) & 3;
        if ((size == 3 && opcode >= 2) || (size == 2 && opcode == 3))
            return false;
        op.kind = Kind::Ordered;
        op.address = Address::Offset;
        op.bytes = 1u << size;
        op.register_bytes = size == 3 || opcode == 2 ? 8 : 4;
        op.sign_extend = opcode >= 2;
        op.load = opcode != 0;
        op.acquire = op.load;
        op.release = !op.load;
        op.displacement = SignedImmediate((instruction >> 12) & 511, 9);
        return true;
    }

    if (Matches(instruction, 0x3f000000, 0x3d000000) ||
        Matches(instruction, 0x3f000000, 0x3c000000)) {
        const unsigned scale = (instruction >> 30) | (((instruction >> 23) & 1) << 2);
        if (scale > 4)
            return false;
        op.kind = Kind::Simd;
        op.bytes = 1u << scale;
        if ((instruction >> 24) & 1) {
            op.address = Address::Offset;
            op.displacement = ((instruction >> 10) & 4095) << scale;
            return true;
        }
        const unsigned mode = (instruction >> 10) & 3;
        if ((instruction >> 21) & 1) {
            op.option = (instruction >> 13) & 7;
            if (mode != 2 || !(op.option & 2))
                return false;
            op.address = Address::RegisterOffset;
            op.shift = ((instruction >> 12) & 1) ? scale : 0;
            return true;
        }
        if (mode == 2)
            return false;  // No unprivileged SIMD encoding exists.
        op.address = mode == 1 ? Address::PostIndex : mode == 3 ? Address::PreIndex : Address::Offset;
        op.displacement = SignedImmediate((instruction >> 12) & 511, 9);
        return true;
    }

    if (Matches(instruction, 0x3e000000, 0x2c000000)) {
        const unsigned scale = instruction >> 30;
        if (scale == 3 || (op.load && op.rt == op.rt2))
            return false;
        op.kind = Kind::SimdPair;
        op.pair = true;
        op.bytes = 4u << scale;
        const unsigned mode = (instruction >> 23) & 3;
        op.address = mode == 1 ? Address::PostIndex : mode == 3 ? Address::PreIndex : Address::Offset;
        op.displacement = SignedImmediate((instruction >> 15) & 127, 7) * op.bytes;
        return true;  // Includes LDNP/STNP.
    }

    if (Matches(instruction, 0x3f000000, 0x1c000000)) {
        const unsigned scale = instruction >> 30;
        if (scale == 3)
            return false;
        op.kind = Kind::SimdLiteral;
        op.load = true;
        op.bytes = 4u << scale;
        op.address = Address::Literal;
        op.displacement = SignedImmediate((instruction >> 5) & 0x7ffff, 19) * 4;
        return true;
    }

    if (Matches(instruction, 0xbf200000, 0x0c000000)) {
        op.kind = Kind::StructureMultiple;
        op.quad = (instruction >> 30) & 1;
        const bool writeback = (instruction >> 23) & 1;
        if (!writeback && op.rm != 0)
            return false;
        op.address = writeback ? Address::PostIndex : Address::Base;
        op.bytes = 1u << ((instruction >> 10) & 3);
        switch ((instruction >> 12) & 15) {
        case 0: op.structures = 4; break;
        case 2: op.repetitions = 4; break;
        case 4: op.structures = 3; break;
        case 6: op.repetitions = 3; break;
        case 7: break;
        case 8: op.structures = 2; break;
        case 10: op.repetitions = 2; break;
        default: return false;
        }
        return op.bytes != 8 || op.quad || op.structures == 1;
    }

    if (Matches(instruction, 0xbf000000, 0x0d000000)) {
        op.kind = Kind::StructureSingle;
        op.quad = (instruction >> 30) & 1;
        const bool writeback = (instruction >> 23) & 1;
        if (!writeback && op.rm != 0)
            return false;
        op.address = writeback ? Address::PostIndex : Address::Base;
        op.structures = (((instruction >> 13) & 1) << 1 | ((instruction >> 21) & 1)) + 1;
        const unsigned size = (instruction >> 10) & 3;
        const unsigned s = (instruction >> 12) & 1;
        unsigned scale = (instruction >> 14) & 3;
        switch (scale) {
        case 0: op.lane = (unsigned(op.quad) << 3) | (s << 2) | size; break;
        case 1:
            if (size & 1)
                return false;
            op.lane = (unsigned(op.quad) << 2) | (s << 1) | (size >> 1);
            break;
        case 2:
            if ((size & 2) || ((size & 1) && s))
                return false;
            if (size & 1) {
                scale = 3;
                op.lane = op.quad;
            } else {
                op.lane = (unsigned(op.quad) << 1) | s;
            }
            break;
        case 3:
            if (!op.load || s)
                return false;
            op.replicate = true;
            scale = size;
            break;
        }
        op.bytes = 1u << scale;
        return true;
    }
    op.kind = Kind::Invalid;
    return false;
}

bool BaselineMemorySupported(u32 instruction) noexcept {
    BaselineMemoryOp op;
    return DecodeBaselineMemory(instruction, op);
}

bool ExecuteBaselineMemory(A64JitState* state, u32 instruction) {
    const unsigned phase = std::exchange(state->baseline_memory_phase, u8{0});
    BaselineMemoryOp op;
    if (!DecodeBaselineMemory(instruction, op))
        return BadInstruction(state);
    using Kind = BaselineMemoryOp::Kind;
    switch (op.kind) {
    case Kind::Simd: case Kind::SimdPair: case Kind::SimdLiteral:
        return SimdMemory(state, op, phase);
    case Kind::StructureMultiple:
        return StructureMultiple(state, op, phase);
    case Kind::StructureSingle:
        return phase ? BadInstruction(state) : StructureSingle(state, op);
    default: break;
    }
    if (phase)
        return BadInstruction(state);
    const u64 address = EffectiveAddress(*state, op);
    if (!Alignment(state, address, op.bytes))
        return false;
    if (op.kind == Kind::Ordered) {
        if (op.release)
            Fence();
        if (op.load) {
            u64 value = ReadScalar(state, address, op.bytes);
            if (Aborted(state))
                return false;
            if (op.acquire)
                Fence();
            if (op.sign_extend)
                value = static_cast<u64>(SignedImmediate(value, op.bytes * 8));
            SetX(*state, op.rt, value, op.register_bytes ? op.register_bytes : op.bytes);
        } else {
            WriteScalar(state, address, X(*state, op.rt), op.bytes);
            if (Aborted(state))
                return false;
            if (op.release)
                Fence();  // RCsc STLR must also order a following LDAR.
        }
        return true;
    }
    if (op.kind == Kind::Exclusive) {
        switch (op.bytes) {
        case 1: return Exclusive<u8>(state, op, address);
        case 2: return Exclusive<u16>(state, op, address);
        case 4: return Exclusive<u32>(state, op, address);
        case 8: return Exclusive<u64>(state, op, address);
        case 16: return Exclusive<A64::Vector>(state, op, address);
        }
    }
    if (op.kind == Kind::AtomicCASP)
        return op.bytes == 16 ? AtomicPair<A64::Vector>(state, op, address) : AtomicPair<u64>(state, op, address);
    if (op.kind == Kind::AtomicRmw || op.kind == Kind::AtomicCAS) {
        switch (op.bytes) {
        case 1: return AtomicScalar<u8>(state, op, address);
        case 2: return AtomicScalar<u16>(state, op, address);
        case 4: return AtomicScalar<u32>(state, op, address);
        case 8: return AtomicScalar<u64>(state, op, address);
        }
    }
    return BadInstruction(state);
}

}  // namespace Dynarmic::Backend::X64
