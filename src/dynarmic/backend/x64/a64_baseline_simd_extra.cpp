// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
//
// Independent AdvSIMD I8MM/BF16 semantics. These are bounded precompiled
// arithmetic helpers; they do not invoke a decoder frontend, IR or another JIT.
// Architectural references and independent oracle tests are documented in
// validation/simd-extra/notes.txt in the accompanying validation bundle.

#include "dynarmic/backend/x64/a64_baseline_simd_extra.h"

#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/fused.h"
#include "dynarmic/common/fp/op/FPMulAdd.h"
#include "dynarmic/common/fp/process_exception.h"
#include "dynarmic/common/fp/rounding_mode.h"
#include "dynarmic/common/fp/unpacked.h"

namespace Dynarmic::Backend::X64 {
namespace {
enum class Extra {
    None, SignedMatrix, UnsignedMatrix, MixedMatrix,
    SignedUnsignedDotElement, UnsignedSignedDotElement, UnsignedSignedDot,
    BfConvert, BfConvertNarrow, BfDotElement, BfDot,
    BfLongElement, BfLong, BfMatrix,
};

Extra Decode(u32 i) {
    switch (i & 0xffe0fc00u) {
    case 0x4e80a400: return Extra::SignedMatrix;
    case 0x6e80a400: return Extra::UnsignedMatrix;
    case 0x4e80ac00: return Extra::MixedMatrix;
    case 0x6e40ec00: return Extra::BfMatrix;
    }
    switch (i & 0xbfc0f400u) {
    case 0x0f00f000: return Extra::SignedUnsignedDotElement;
    case 0x0f80f000: return Extra::UnsignedSignedDotElement;
    case 0x0f40f000: return Extra::BfDotElement;
    case 0x0fc0f000: return Extra::BfLongElement;
    }
    switch (i & 0xbfe0fc00u) {
    case 0x0e809c00: return Extra::UnsignedSignedDot;
    case 0x2e40fc00: return Extra::BfDot;
    case 0x2ec0fc00: return Extra::BfLong;
    }
    if ((i & 0xfffffc00u) == 0x1e634000)
        return Extra::BfConvert;
    if ((i & 0xbffffc00u) == 0x0ea16800)
        return Extra::BfConvertNarrow;
    return Extra::None;
}

template<class T>
using Lanes = std::array<T, 16 / sizeof(T)>;

template<class T>
Lanes<T> Read(const A64JitState& state, unsigned reg) {
    Lanes<T> result;
    std::memcpy(result.data(), &state.vec[2 * reg], 16);
    return result;
}

template<class T>
void Write(A64JitState& state, unsigned reg, const Lanes<T>& value) {
    std::memcpy(&state.vec[2 * reg], value.data(), 16);
}

s32 Byte(u8 value, bool is_unsigned) {
    return is_unsigned ? s32(value) : s32(std::bit_cast<s8>(value));
}

void IntegerMatrix(A64JitState& state, u32 instruction, Extra kind) {
    const unsigned d = instruction & 31;
    const auto lhs = Read<u8>(state, (instruction >> 5) & 31);
    const auto rhs = Read<u8>(state, (instruction >> 16) & 31);
    auto result = Read<u32>(state, d);
    const bool lhs_unsigned = kind != Extra::SignedMatrix;
    const bool rhs_unsigned = kind == Extra::UnsignedMatrix;
    // Both input registers hold contiguous eight-byte rows. The second
    // register is the column-major representation of the right-hand matrix.
    for (unsigned output = 0; output < 4; ++output) {
        s32 dot = 0;
        for (unsigned k = 0; k < 8; ++k)
            dot += Byte(lhs[8 * (output >> 1) + k], lhs_unsigned) *
                   Byte(rhs[8 * (output & 1) + k], rhs_unsigned);
        // Addition wraps modulo 2^32 and never sets FPSR.QC.
        result[output] += static_cast<u32>(dot);
    }
    Write(state, d, result);
}

void IntegerDot(A64JitState& state, u32 instruction, Extra kind) {
    const unsigned d = instruction & 31;
    const auto lhs = Read<u8>(state, (instruction >> 5) & 31);
    const auto rhs = Read<u8>(state, (instruction >> 16) & 31);
    const auto accumulator = Read<u32>(state, d);
    Lanes<u32> result{};
    const unsigned count = (instruction & (1u << 30)) ? 4 : 2;
    const unsigned index = ((instruction >> 10) & 2) | ((instruction >> 21) & 1);
    const bool element = kind != Extra::UnsignedSignedDot;
    const bool lhs_unsigned = kind != Extra::SignedUnsignedDotElement;
    for (unsigned output = 0; output < count; ++output) {
        s32 dot = 0;
        const unsigned rhs_base = 4 * (element ? index : output);
        for (unsigned k = 0; k < 4; ++k)
            dot += Byte(lhs[4 * output + k], lhs_unsigned) *
                   Byte(rhs[rhs_base + k], !lhs_unsigned);
        result[output] = accumulator[output] + static_cast<u32>(dot);
    }
    // A 64-bit vector result clears Vd[127:64], even when Vd aliases Vm.
    Write(state, d, result);
}

// Base FEAT_BF16 is independent of FEAT_EBF16 and FEAT_AFP. The enclosing
// runtime exposes the original FPCR layout, whose AH/EBF fields are RES0.
// BFCVT follows the architectural FPCR rounding/exception controls; it is
// not the fixed-round-to-odd arithmetic used by BFDOT and BFMMLA.
u16 ConvertBf(u32 value, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned exponent = (value >> 23) & 255;
    const u32 fraction = value & 0x7fffff;
    const bool negative = (value >> 31) != 0;
    u16 result = static_cast<u16>(value >> 16);
    if (exponent == 255) {
        if (!fraction)
            return result;
        if (!(fraction & 0x400000))
            FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return fpcr.DN() ? u16{0x7fc0} : u16(result | 0x40);
    }
    if (!exponent && fraction && fpcr.FZ()) {
        FP::FPProcessException(FP::FPExc::InputDenorm, fpcr, fpsr);
        return static_cast<u16>(negative ? 0x8000 : 0);
    }
    const u32 remainder = value & 0xffff;
    // Tininess is tested before rounding. An enabled underflow trap also
    // observes an exactly representable nonzero subnormal conversion.
    if (!exponent && fraction && (remainder || fpcr.UFE()))
        FP::FPProcessException(FP::FPExc::Underflow, fpcr, fpsr);
    if (!remainder)
        return result;

    bool increment = false;
    switch (fpcr.RMode()) {
    case FP::RoundingMode::ToNearest_TieEven:
        increment = remainder > 0x8000 || (remainder == 0x8000 && (result & 1));
        break;
    case FP::RoundingMode::TowardsPlusInfinity:
        increment = !negative;
        break;
    case FP::RoundingMode::TowardsMinusInfinity:
        increment = negative;
        break;
    default:
        break;  // The remaining FPCR encoding is round toward zero.
    }
    result = static_cast<u16>(result + increment);
    if ((result & 0x7f80) == 0x7f80)
        FP::FPProcessException(FP::FPExc::Overflow, fpcr, fpsr);
    FP::FPProcessException(FP::FPExc::Inexact, fpcr, fpsr);
    return result;
}

constexpr u32 default_nan = 0x7fc00000;
constexpr u32 infinity = 0x7f800000;
constexpr u32 sign_bit = 0x80000000;

bool IsNan(u32 value) { return (value & ~sign_bit) > infinity; }
bool IsInfinity(u32 value) { return (value & ~sign_bit) == infinity; }
bool IsFlushedZero(u32 value) { return (value & infinity) == 0; }

FP::FPUnpacked FiniteValue(u32 value) {
    return FP::ToNormalized((value & sign_bit) != 0,
                            int((value >> 23) & 255) - 127 - 23,
                            (value & 0x7fffff) | 0x800000);
}

// Only pure integer floating-point primitives are reused. BF16 dot arithmetic
// suppresses all exception flags and traps, flushes tiny results, and differs
// from generic IEEE round-to-odd by overflowing to infinity, not max-finite.
u32 RoundOdd(FP::FPUnpacked value) {
    if (!value.mantissa)
        return 0;  // An exact cancellation has positive zero in this mode.
    FP::FPCR control;
    control.FZ(true);
    FP::FPSR discarded;
    const u32 result = FP::FPRound<u32>(value, control, FP::RoundingMode::ToOdd, discarded);
    return discarded.OFC() ? (value.sign ? sign_bit : 0) | infinity : result;
}

u32 BfMultiply(u16 left, u16 right) {
    const u32 a = u32(left) << 16;
    const u32 b = u32(right) << 16;
    const u32 sign = (a ^ b) & sign_bit;
    if (IsNan(a) || IsNan(b) ||
        (IsInfinity(a) && IsFlushedZero(b)) || (IsInfinity(b) && IsFlushedZero(a)))
        return default_nan;
    if (IsInfinity(a) || IsInfinity(b))
        return sign | infinity;
    if (IsFlushedZero(a) || IsFlushedZero(b))
        return sign;
    return RoundOdd(FP::FusedMulAdd({false, 0, 0}, FiniteValue(a), FiniteValue(b)));
}

u32 BfAdd(u32 a, u32 b) {
    if (IsNan(a) || IsNan(b))
        return default_nan;
    if (IsInfinity(a) && IsInfinity(b) && ((a ^ b) & sign_bit))
        return default_nan;
    if (IsInfinity(a))
        return a;
    if (IsInfinity(b))
        return b;
    if (IsFlushedZero(a) && IsFlushedZero(b))
        return (a & b) & sign_bit;
    if (IsFlushedZero(a))
        return b;
    if (IsFlushedZero(b))
        return a;
    const FP::FPUnpacked one{false, 0, u64{1} << FP::normalized_point_position};
    return RoundOdd(FP::FusedMulAdd(FiniteValue(a), FiniteValue(b), one));
}

u32 BfDot(u32 accumulator, u16 a0, u16 a1, u16 b0, u16 b1) {
    // Keep both architectural rounding boundaries: pair sum, then accumulator.
    return BfAdd(accumulator, BfAdd(BfMultiply(a0, b0), BfMultiply(a1, b1)));
}

void BfloatConvert(A64JitState& state, u32 instruction, Extra kind,
                   FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = instruction & 31;
    const auto source = Read<u32>(state, (instruction >> 5) & 31);
    const bool upper = kind == Extra::BfConvertNarrow && (instruction & (1u << 30));
    auto result = upper ? Read<u16>(state, d) : Lanes<u16>{};
    const unsigned count = kind == Extra::BfConvert ? 1 : 4;
    for (unsigned n = 0; n < count; ++n)
        result[(upper ? 4 : 0) + n] = ConvertBf(source[n], fpcr, fpsr);
    Write(state, d, result);
}

void BfloatDot(A64JitState& state, u32 instruction, Extra kind) {
    const unsigned d = instruction & 31;
    const auto lhs = Read<u16>(state, (instruction >> 5) & 31);
    const auto rhs = Read<u16>(state, (instruction >> 16) & 31);
    const auto accumulator = Read<u32>(state, d);
    Lanes<u32> result{};
    const unsigned count = (instruction & (1u << 30)) ? 4 : 2;
    const unsigned index = ((instruction >> 10) & 2) | ((instruction >> 21) & 1);
    for (unsigned output = 0; output < count; ++output) {
        const unsigned r = 2 * (kind == Extra::BfDotElement ? index : output);
        result[output] = BfDot(accumulator[output], lhs[2 * output], lhs[2 * output + 1],
                               rhs[r], rhs[r + 1]);
    }
    Write(state, d, result);
}

void BfloatMatrix(A64JitState& state, u32 instruction) {
    const unsigned d = instruction & 31;
    const auto lhs = Read<u16>(state, (instruction >> 5) & 31);
    const auto rhs = Read<u16>(state, (instruction >> 16) & 31);
    auto result = Read<u32>(state, d);
    for (unsigned output = 0; output < 4; ++output) {
        const unsigned l = 4 * (output >> 1);
        const unsigned r = 4 * (output & 1);
        result[output] = BfDot(result[output], lhs[l], lhs[l + 1], rhs[r], rhs[r + 1]);
        result[output] = BfDot(result[output], lhs[l + 2], lhs[l + 3], rhs[r + 2], rhs[r + 3]);
    }
    Write(state, d, result);
}

void BfloatLong(A64JitState& state, u32 instruction, Extra kind,
                FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = instruction & 31;
    const bool element = kind == Extra::BfLongElement;
    const unsigned m = (instruction >> 16) & (element ? 15 : 31);
    const auto lhs = Read<u16>(state, (instruction >> 5) & 31);
    const auto rhs = Read<u16>(state, m);
    auto result = Read<u32>(state, d);
    const unsigned selected = (instruction >> 30) & 1;
    const unsigned index = ((instruction >> 9) & 4) | ((instruction >> 20) & 3);
    for (unsigned output = 0; output < 4; ++output) {
        const u32 a = u32(lhs[2 * output + selected]) << 16;
        const u32 b = u32(rhs[element ? index : 2 * output + selected]) << 16;
        // BFMLAL has ordinary FP32 fused semantics and cumulative exceptions;
        // it must not use the fixed dot-product control word or two roundings.
        result[output] = FP::FPMulAdd<u32>(result[output], a, b, fpcr, fpsr);
    }
    Write(state, d, result);
}
}  // namespace

bool BaselineSimdExtraSupported(u32 instruction) {
    return Decode(instruction) != Extra::None;
}

void ExecuteBaselineSimdExtra(A64JitState* state, u32 instruction,
                             FP::FPCR fpcr, FP::FPSR& fpsr) {
    const auto kind = Decode(instruction);
    switch (kind) {
    case Extra::SignedMatrix: case Extra::UnsignedMatrix: case Extra::MixedMatrix:
        IntegerMatrix(*state, instruction, kind);
        return;
    case Extra::SignedUnsignedDotElement: case Extra::UnsignedSignedDotElement:
    case Extra::UnsignedSignedDot:
        IntegerDot(*state, instruction, kind);
        return;
    case Extra::BfConvert: case Extra::BfConvertNarrow:
        BfloatConvert(*state, instruction, kind, fpcr, fpsr);
        return;
    case Extra::BfDotElement: case Extra::BfDot:
        BfloatDot(*state, instruction, kind);
        return;
    case Extra::BfMatrix:
        BfloatMatrix(*state, instruction);
        return;
    case Extra::BfLongElement: case Extra::BfLong:
        BfloatLong(*state, instruction, kind, fpcr, fpsr);
        return;
    case Extra::None:
        throw std::invalid_argument("Unallocated baseline SIMD extension instruction");
    }
}
}  // namespace Dynarmic::Backend::X64
