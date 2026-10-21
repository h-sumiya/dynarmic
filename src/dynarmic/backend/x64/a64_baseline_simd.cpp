// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
//
// Independent low-latency ARMv8 FP/AdvSIMD semantics. There is no invocation of
// the Dynarmic decoder, translator, IR, register allocator, or optimizing JIT.
// The shared Common::FP routines below are pure arithmetic primitives; using
// them preserves ARM rounding, NaN priority, fused operations and exceptions.

#include "dynarmic/backend/x64/a64_baseline_simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/info.h"
#include "dynarmic/common/fp/op.h"
#include "dynarmic/common/fp/process_exception.h"
#include "dynarmic/common/fp/rounding_mode.h"
#include "dynarmic/common/fp/unpacked.h"
#include "dynarmic/common/fp/util.h"
#include "dynarmic/common/u128.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic::Backend::X64 {
namespace {

enum class Kind {
    None, FpUnary, FpBinary, FpFma, FpCompare, FpCondCompare, FpSelect,
    FpImmediate, FpInteger, FpFixed, Copy, Table, Extract, Permute,
    Same, Different, Misc, Reduce, Immediate, Shift, Indexed, Extra,
    HalfSame, HalfMisc, HalfReduce, Complex, ComplexIndexed, Extension,
    FpRoundRange, FpJavascript, FpLong, FpLongIndexed,
};

constexpr u64 Mask(unsigned bits) {
    return bits == 64 ? ~u64{0} : (u64{1} << bits) - 1;
}
constexpr s64 Signed(u64 value, unsigned bits) {
    const u64 sign = u64{1} << (bits - 1);
    return std::bit_cast<s64>((value ^ sign) - sign);
}
constexpr unsigned FpBits(unsigned type) {
    return type == 3 ? 16 : 32u << type;
}

// Tests complete encoding groups, then rejects reserved sizes/opcode forms.
// No input-data-dependent probing or heavyweight decode structures are built.
Kind Decode(u32 i) {
    if (BaselineSimdExtraSupported(i)) return Kind::Extension;
    const bool q = (i >> 30) & 1;
    const bool u = (i >> 29) & 1;
    const bool scalar = (i >> 28) & 1;
    const unsigned size = (i >> 22) & 3;
    const unsigned op = (i >> 11) & 31;
    if ((i & 0xfffffc00) == 0x1e7e0000)
        return Kind::FpJavascript;
    if ((i & 0xff3e7c00) == 0x1e284000)
        return size <= 1 ? Kind::FpRoundRange : Kind::None;
    if ((i & 0x9fbfec00) == 0x0e21e800)
        return q || size == 0 ? Kind::FpRoundRange : Kind::None;
    if ((i & 0xbf60fc00) == 0x0e20ec00 || (i & 0xbf60fc00) == 0x2e20cc00)
        return Kind::FpLong;
    if ((i & 0xbfc0b400) == 0x0f800000 || (i & 0xbfc0b400) == 0x2f808000)
        return Kind::FpLongIndexed;
    if ((i & 0xff000000) == 0x1f000000)
        return size != 2 ? Kind::FpFma : Kind::None;
    if ((i & 0xff207c00) == 0x1e204000) {
        const unsigned unary = (i >> 15) & 63;
        if (size == 2)
            return Kind::None;
        if (unary <= 3 || (unary >= 8 && unary <= 15 && unary != 13))
            return Kind::FpUnary;
        if ((unary == 4 || unary == 5 || unary == 7) && size != (unary & 3))
            return Kind::FpUnary;
        return Kind::None;
    }
    if ((i & 0xff200c00) == 0x1e200800)
        return size != 2 && ((i >> 12) & 15) <= 8 ? Kind::FpBinary : Kind::None;
    if ((i & 0xff20fc07) == 0x1e202000)
        return size != 2 ? Kind::FpCompare : Kind::None;
    if ((i & 0xff200c00) == 0x1e200400)
        return size != 2 ? Kind::FpCondCompare : Kind::None;
    if ((i & 0xff200c00) == 0x1e200c00)
        return size != 2 ? Kind::FpSelect : Kind::None;
    if ((i & 0xff201fe0) == 0x1e201000)
        return size != 2 ? Kind::FpImmediate : Kind::None;
    if ((i & 0x7f20fc00) == 0x1e200000) {
        const unsigned cvt = (i >> 16) & 31;
        const bool sf = i >> 31;
        if (cvt == 6 || cvt == 7)
            return (size == 0 && !sf) || (size == 1 && sf) || size == 3 ? Kind::FpInteger : Kind::None;
        if (cvt == 14 || cvt == 15)
            return size == 2 && sf ? Kind::FpInteger : Kind::None;
        if (size == 2)
            return Kind::None;
        switch (cvt) {
        case 0: case 1: case 2: case 3: case 4: case 5:
        case 8: case 9: case 16: case 17: case 24: case 25:
            return Kind::FpInteger;
        default: return Kind::None;
        }
    }
    if ((i & 0x7f200000) == 0x1e000000) {
        const unsigned cvt = (i >> 16) & 31;
        return size != 2 && (cvt == 2 || cvt == 3 || cvt == 24 || cvt == 25) &&
                       ((i >> 31) || ((i >> 10) & 63) >= 32)
                   ? Kind::FpFixed : Kind::None;
    }

    // Scalar AdvSIMD encodings always have Q=1. The scalar FP group above is
    // distinct and must never be accidentally treated as a vector operation.
    if (scalar && !q)
        return Kind::None;
    if ((i & 0x8fe08400) == 0x0e000400) {
        const unsigned imm5 = (i >> 16) & 31;
        if (!imm5 || ((imm5 & 15) == 0 && imm5 != 8))
            return Kind::None;
        const unsigned esize = std::countr_zero(imm5);
        if (esize > 3)
            return Kind::None;
        if (scalar)
            return !u && op == 0 ? Kind::Copy : Kind::None;
        if (u)
            return q ? Kind::Copy : Kind::None;
        switch (op) {
        case 0: case 1: return q || esize != 3 ? Kind::Copy : Kind::None;
        case 3: return q && imm5 ? Kind::Copy : Kind::None;
        case 5: return esize < (q ? 3u : 2u) ? Kind::Copy : Kind::None;
        case 7: return q ? (esize == 3 ? Kind::Copy : Kind::None) : (esize < 3 ? Kind::Copy : Kind::None);
        default: return Kind::None;
        }
    }
    if ((i & 0xbfe08c00) == 0x0e000000)
        return Kind::Table;
    if ((i & 0xbfe08400) == 0x2e000000)
        return q || ((i >> 11) & 15) < 8 ? Kind::Extract : Kind::None;
    if ((i & 0xbf208c00) == 0x0e000800) {
        const unsigned permute = (i >> 12) & 7;
        return (q || size != 3) && permute != 0 && permute != 4 ? Kind::Permute : Kind::None;
    }
    if ((i & 0x9ff80c00) == 0x0f000400) {
        const unsigned cmode = (i >> 12) & 15;
        return cmode != 15 || !u || q ? Kind::Immediate : Kind::None;
    }
    if ((i & 0x9ff80c00) == 0x0f000c00)
        return !u && ((i >> 12) & 15) == 15 ? Kind::Immediate : Kind::None;
    if ((i & 0x8f800400) == 0x0f000400) {
        const unsigned immh = (i >> 19) & 15;
        if (!immh)
            return Kind::None;
        const unsigned s = 31 - std::countl_zero(immh);
        switch (op) {
        case 0: case 2: case 4: case 6: case 10:
            return scalar ? (s == 3 ? Kind::Shift : Kind::None) : (q || s != 3 ? Kind::Shift : Kind::None);
        case 8:
            return u && (scalar ? s == 3 : q || s != 3) ? Kind::Shift : Kind::None;
        case 12:
            return u && (scalar || q || s != 3) ? Kind::Shift : Kind::None;
        case 14:
            return scalar || q || s != 3 ? Kind::Shift : Kind::None;
        case 16: case 17:
            return s < 3 && (!scalar || u) ? Kind::Shift : Kind::None;
        case 18: case 19:
            return s < 3 ? Kind::Shift : Kind::None;
        case 20:
            return !scalar && s < 3 ? Kind::Shift : Kind::None;
        case 28: case 31:
            return s >= 1 && (scalar || q || s != 3) ? Kind::Shift : Kind::None;
        default: return Kind::None;
        }
    }
    if ((i & 0x8f000400) == 0x0f000000) {
        const unsigned x = (i >> 12) & 15;
        const bool l = (i >> 21) & 1;
        if (u && !scalar && (x == 1 || x == 3 || x == 5 || x == 7))
            return (size == 1 && (q || !(i & 0x800))) || (size == 2 && q && !l) ? Kind::ComplexIndexed : Kind::None;
        if (x == 1 || x == 5 || x == 9) {
            // Half-precision and ordinary FP element multiply. M belongs to
            // Vm for S/D, but belongs to the lane index for H.
            return (size == 0 || size >= 2) && (!u || x == 9) &&
                           (size != 3 || (!l && (scalar || q)))
                       ? Kind::Indexed : Kind::None;
        }
        if (x == 14)
            return !scalar && size == 2 ? Kind::Indexed : Kind::None;
        if (size != 1 && size != 2)
            return Kind::None;
        switch (x) {
        case 0: case 4: return !scalar && u ? Kind::Indexed : Kind::None;
        case 2: case 6: case 8: case 10:
            return !scalar && (x != 8 || !u) ? Kind::Indexed : Kind::None;
        case 3: case 7: case 11: case 12:
            return !u ? Kind::Indexed : Kind::None;
        case 13: return Kind::Indexed;
        case 15: return u ? Kind::Indexed : Kind::None;
        default: return Kind::None;
        }
    }
    if ((i & 0x8f200400) == 0x0e200400) {
        if (op >= 24) {
            if (!scalar && (size & 1) && !q)
                return Kind::None;
            const bool upper = size & 2;
            if (scalar) {
                if (op == 27) return !u && !upper ? Kind::Same : Kind::None;
                if (op == 28) return !upper || u ? Kind::Same : Kind::None;
                if (op == 29) return u ? Kind::Same : Kind::None;
                if (op == 26) return u && upper ? Kind::Same : Kind::None;
                if (op == 31) return !u ? Kind::Same : Kind::None;
                return Kind::None;
            }
            if (op == 25) return !u ? Kind::Same : Kind::None;
            if (op == 27) return !upper ? Kind::Same : Kind::None;
            if (op == 28) return !upper || u ? Kind::Same : Kind::None;
            if (op == 29) return u ? Kind::Same : Kind::None;
            if (op == 31) return !u || !upper ? Kind::Same : Kind::None;
            return Kind::Same;
        }
        if (op == 3)
            return !scalar ? Kind::Same : Kind::None;
        if (scalar) {
            switch (op) {
            case 1: case 5: case 9: case 11: return Kind::Same;
            case 22: return size == 1 || size == 2 ? Kind::Same : Kind::None;
            case 6: case 7: case 8: case 10: case 16: case 17:
                return size == 3 ? Kind::Same : Kind::None;
            default: return Kind::None;
            }
        }
        if (size == 3 && (!q || op == 0 || op == 2 || op == 4 ||
                         (op >= 12 && op <= 15) || (op >= 18 && op <= 22)))
            return Kind::None;
        if ((op == 22 && size == 0) || (op == 19 && u && size != 0) || (op == 23 && u))
            return Kind::None;
        return Kind::Same;
    }
    if ((i & 0x8f200c00) == 0x0e200000) {
        const unsigned x = (i >> 12) & 15;
        if (x == 14)
            return !scalar && !u && (size == 0 || size == 3) ? Kind::Different : Kind::None;
        if (size == 3 || x == 15)
            return Kind::None;
        if (x == 9 || x == 11 || x == 13)
            return !u && size != 0 ? Kind::Different : Kind::None;
        return !scalar ? Kind::Different : Kind::None;
    }
    if ((i & 0x8f3e0c00) == 0x0e200800) {
        const unsigned x = (i >> 12) & 31;
        if (x >= 24 || (x >= 12 && x <= 15)) {
            // FP32/64 arithmetic/conversion, plus unsigned estimate op 28.
            if (!scalar && (size & 1) && !q)
                return Kind::None;
            if (x == 12 || x == 13 || x == 14 || x == 15) {
                if (!(size & 2) || (x == 14 && u) || (scalar && x == 15))
                    return Kind::None;
                return Kind::Misc;
            }
            if (x == 24) return !scalar && (!u || !(size & 2)) ? Kind::Misc : Kind::None;
            if (x == 25) return !scalar ? Kind::Misc : Kind::None;
            if (x == 26 || x == 27 || x == 29) return Kind::Misc;
            if (x == 28) return !(size & 2) || (!scalar && size == 2) ? Kind::Misc : Kind::None;
            if (x == 31) return (size & 2) && (scalar ? !u : u) ? Kind::Misc : Kind::None;
            return Kind::None;
        }
        if (x == 22)
            return u ? (size == 1 ? Kind::Misc : Kind::None) : (!scalar && size < 2 ? Kind::Misc : Kind::None);
        if (x == 23)
            return !scalar && !u && size < 2 ? Kind::Misc : Kind::None;
        if (x == 18 || x == 20)
            return size < 3 && (x != 18 || !scalar || u) ? Kind::Misc : Kind::None;
        if (scalar) {
            switch (x) {
            case 3: case 7: return Kind::Misc;
            case 8: case 9: case 10: case 11:
                return size == 3 && (x != 10 || !u) ? Kind::Misc : Kind::None;
            default: return Kind::None;
            }
        }
        switch (x) {
        case 0: return size < (u ? 2u : 3u) ? Kind::Misc : Kind::None;
        case 1: return !u && size == 0 ? Kind::Misc : Kind::None;
        case 2: case 4: case 6: return size < 3 ? Kind::Misc : Kind::None;
        case 5: return u ? (size < 2 ? Kind::Misc : Kind::None) : (size == 0 ? Kind::Misc : Kind::None);
        case 3: case 7: case 8: case 9: case 11:
            return q || size != 3 ? Kind::Misc : Kind::None;
        case 10: return !u && (q || size != 3) ? Kind::Misc : Kind::None;
        case 19: return u && size < 3 ? Kind::Misc : Kind::None;
        default: return Kind::None;
        }
    }
    if ((i & 0x8f3e0c00) == 0x0e300800) {
        const unsigned x = (i >> 12) & 31;
        if (!u && !(size & 1) && (x == 12 || x == 15 || (scalar && x == 13 && !(size & 2))))
            return Kind::HalfReduce;
        if (scalar) {
            if (x == 27 && !u && size == 3)
                return Kind::Reduce;
            return u && (x == 12 || x == 15 || (x == 13 && !(size & 2))) ? Kind::Reduce : Kind::None;
        }
        if (x == 12 || x == 15)
            return u && q && !(size & 1) ? Kind::Reduce : Kind::None;
        if (size == 3 || (!q && size == 2))
            return Kind::None;
        return x == 3 || x == 10 || x == 26 || (x == 27 && !u) ? Kind::Reduce : Kind::None;
    }
    if ((i & 0x8f7e0c00) == 0x0e780800) {
        const unsigned x = (i >> 12) & 31;
        if (!((x >= 12 && x <= 15) || x >= 24)) return Kind::None;
        if (x == 28 && (size & 2)) return Kind::None;
        return Decode((i & ~0x007e0000u) | 0x00200000u) == Kind::Misc ? Kind::HalfMisc : Kind::None;
    }
    if ((i & 0x8f600400) == 0x0e400400 && op < 8)
        return Decode((i & ~0x00400000u) | 0x0020c000u) == Kind::Same ? Kind::HalfSame : Kind::None;
    if ((i & 0x8f200400) == 0x0e000400) {
        if (!scalar && u && ((op >= 24 && op <= 28) || op == 30))
            return size != 0 && (q || size != 3) ? Kind::Complex : Kind::None;
        if (op == 18)
            return !scalar && size == 2 ? Kind::Extra : Kind::None;
        return u && (op == 16 || op == 17) && (size == 1 || size == 2) ? Kind::Extra : Kind::None;
    }
    return Kind::None;
}

// Load/copy through memcpy: architectural lanes are little-endian, and this
// backend is only built on little-endian x86-64. No aliasing-violating casts.
struct Vector {
    std::array<u64, 2> word{};
    u64 Get(unsigned bits, unsigned lane) const {
        const unsigned at = bits * lane;
        return (word[at / 64] >> (at % 64)) & Mask(bits);
    }
    void Set(unsigned bits, unsigned lane, u64 value) {
        const unsigned at = bits * lane;
        const u64 mask = Mask(bits) << (at % 64);
        word[at / 64] = (word[at / 64] & ~mask) | ((value & Mask(bits)) << (at % 64));
    }
};
Vector Read(const A64JitState* s, unsigned reg) {
    return {{s->vec[reg * 2], s->vec[reg * 2 + 1]}};
}
void Write(A64JitState* s, unsigned reg, Vector value, unsigned bits = 128) {
    if (bits < 64) value.word[0] &= Mask(bits);
    if (bits <= 64) value.word[1] = 0;
    s->vec[reg * 2] = value.word[0];
    s->vec[reg * 2 + 1] = value.word[1];
}
u64 Gpr(const A64JitState* s, unsigned reg, unsigned bits) {
    return reg == 31 ? 0 : s->reg[reg] & Mask(bits);
}
void SetGpr(A64JitState* s, unsigned reg, unsigned bits, u64 value) {
    if (reg != 31) s->reg[reg] = value & Mask(bits);
}

bool Condition(u32 nzcv, unsigned cond) {
    const bool n = nzcv >> 31, z = (nzcv >> 30) & 1;
    const bool c = (nzcv >> 29) & 1, v = (nzcv >> 28) & 1;
    bool pass = false;
    switch (cond >> 1) {
    case 0: pass = z; break;
    case 1: pass = c; break;
    case 2: pass = n; break;
    case 3: pass = v; break;
    case 4: pass = c && !z; break;
    case 5: pass = n == v; break;
    case 6: pass = n == v && !z; break;
    case 7: return true;
    }
    return pass != bool(cond & 1);
}

u64 SatAdd(u64 a, u64 b, unsigned bits, bool uns, bool sub, bool& qc) {
    const u64 mask = Mask(bits), sign = u64{1} << (bits - 1);
    const u64 r = (sub ? a - b : a + b) & mask;
    if (uns) {
        if (sub ? a < b : r < a) { qc = true; return sub ? 0 : mask; }
    } else if ((sub ? (a ^ b) & (a ^ r) : ~(a ^ b) & (a ^ r)) & sign) {
        qc = true;
        return a & sign ? sign : sign - 1;
    }
    return r;
}
u64 Saturate(s64 value, unsigned bits, bool uns, bool& qc) {
    if (uns) {
        if (value < 0) { qc = true; return 0; }
        if (u64(value) > Mask(bits)) { qc = true; return Mask(bits); }
    } else if (bits < 64) {
        const s64 min = -(s64{1} << (bits - 1)), max = -min - 1;
        if (value < min) { qc = true; return u64(min) & Mask(bits); }
        if (value > max) { qc = true; return u64(max); }
    }
    return u64(value) & Mask(bits);
}
u64 Narrow(u64 value, unsigned from, bool input_unsigned, bool output_unsigned, bool& qc) {
    const unsigned to = from / 2;
    if (input_unsigned) {
        if (value > Mask(to)) { qc = true; return Mask(to); }
        return value;
    }
    return Saturate(Signed(value, from), to, output_unsigned, qc);
}
u64 ShiftRight(u64 value, unsigned bits, unsigned amount, bool uns, bool round) {
    if (!amount) return value;
    if (amount >= bits) {
        if (round) return uns && amount == bits ? value >> (bits - 1) : 0;
        return !uns && Signed(value, bits) < 0 ? Mask(bits) : 0;
    }
    u64 result = uns ? value >> amount : u64(Signed(value, bits) >> amount);
    if (round) result += (value >> (amount - 1)) & 1;
    return result & Mask(bits);
}
u64 ShiftLeft(u64 value, unsigned bits, int shift, bool uns, bool round, bool sat, bool& qc) {
    if (shift < 0) return ShiftRight(value, bits, unsigned(-shift), uns, round);
    if (!shift || !value) return value;
    if (!sat) return shift >= int(bits) ? 0 : (value << shift) & Mask(bits);
    if (shift < int(bits)) {
        const u64 r = (value << shift) & Mask(bits);
        const u64 back = uns ? r >> shift : u64(Signed(r, bits) >> shift) & Mask(bits);
        if (back == value) return r;
    }
    qc = true;
    return uns ? Mask(bits) : Signed(value, bits) < 0 ? u64{1} << (bits - 1) : Mask(bits - 1);
}
u64 DoublingMultiply(u64 a, u64 b, unsigned bits, bool high, bool round, bool& qc) {
    const s64 x = Signed(a, bits), y = Signed(b, bits);
    const s64 minimum = -(s64{1} << (bits - 1));
    if (x == minimum && y == minimum) {
        qc = true;
        return high ? Mask(bits - 1) : Mask(bits * 2 - 1);
    }
    u64 result = u64(x * y) * 2;
    if (round) result += u64{1} << (bits - 1);
    return high ? u64(std::bit_cast<s64>(result) >> bits) & Mask(bits) : result & Mask(bits * 2);
}

u64 RoundingMultiplyAccumulate(u64 d, u64 a, u64 b, unsigned bits, bool sub, bool& qc) {
    // SQRDMLAH/SH round the accumulated exact product once. In particular,
    // round(-product) is not -round(product), and a min*min product must not
    // saturate before an opposing accumulator can bring it back into range.
    s64 product = Signed(a, bits) * Signed(b, bits);
    if (sub) product = -product;
    product += s64{1} << (bits - 2);
    return Saturate(Signed(d, bits) + (product >> (bits - 1)), bits, false, qc);
}

template<typename T>
bool IsZero(T value) { return (value & ~FP::FPInfo<T>::sign_mask) == 0; }

template<typename T>
T Input(T value, FP::FPCR fpcr, FP::FPSR& fpsr) {
    constexpr T exp = FP::FPInfo<T>::exponent_mask, frac = FP::FPInfo<T>::mantissa_mask;
    if ((sizeof(T) == 2 ? fpcr.FZ16() : fpcr.FZ()) && !(value & exp) && (value & frac)) {
        if constexpr (sizeof(T) != 2) FP::FPProcessException(FP::FPExc::InputDenorm, fpcr, fpsr);
        return value & FP::FPInfo<T>::sign_mask;
    }
    return value;
}
template<typename T>
T NaN(T a, T b, FP::FPCR fpcr, FP::FPSR& fpsr) {
    if (FP::IsSNaN(a) || FP::IsSNaN(b)) FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
    return fpcr.DN() ? FP::FPInfo<T>::DefaultNaN() : *FP::ProcessNaNs(a, b);
}

enum class FloatOp { Add, Sub, Mul, Div, Max, Min, MaxNum, MinNum, MulX, Sqrt };
template<typename T>
T Arithmetic(T aa, T bb, FloatOp op, FP::FPCR fpcr, FP::FPSR& fpsr) {
    using Info = FP::FPInfo<T>;
    T a = Input(aa, fpcr, fpsr), b = Input(bb, fpcr, fpsr);
    if ((op == FloatOp::MaxNum || op == FloatOp::MinNum) && !FP::IsSNaN(a) && !FP::IsSNaN(b)) {
        if (FP::IsQNaN(a) && !FP::IsNaN(b)) return b;
        if (FP::IsQNaN(b) && !FP::IsNaN(a)) return a;
    }
    if (FP::IsNaN(a) || FP::IsNaN(b)) return NaN(a, b, fpcr, fpsr);
    if (op == FloatOp::Add || op == FloatOp::Sub)
        return FP::FPMulAdd<T>(a, op == FloatOp::Sub ? T(b ^ Info::sign_mask) : b,
                              FP::FPValue<T, false, 0, T{1}>(), fpcr, fpsr);
    if (op == FloatOp::Mul || op == FloatOp::MulX) {
        const bool sign = bool((a ^ b) & Info::sign_mask);
        if (op == FloatOp::MulX && ((FP::IsInf(a) && IsZero(b)) ||
                                    (IsZero(a) && FP::IsInf(b))))
            return T((FP::FPValue<T, false, 0, T{1}>() | Info::Zero(sign)) + (T{1} << Info::explicit_mantissa_width));
        return FP::FPMulAdd<T>(Info::Zero(sign), a, b, fpcr, fpsr);
    }
    if (op == FloatOp::Max || op == FloatOp::Min || op == FloatOp::MaxNum || op == FloatOp::MinNum) {
        const bool minimum = op == FloatOp::Min || op == FloatOp::MinNum;
        if (IsZero(a) && IsZero(b))
            return minimum ? a | b : a & b;
        const auto order = [](T value) { return (value & Info::sign_mask) ? T(~value) : T(value | Info::sign_mask); };
        return (order(a) < order(b)) == minimum ? a : b;
    }
    // Exact integer division/square-root is used only by the semantic veneer;
    // ordinary finite S/D operations have their SSE2 path in the emitter.
    // Keeping guard bits plus a sticky remainder avoids double rounding and
    // preserves ARM's before-rounding tininess and exact-tiny UFE behavior.
    // Arithmetic uses FPRound (ignores AHP, observes FZ16), unlike FCVT's
    // FPRoundCV (observes AHP, ignores FZ16).
    const bool negative = bool((a ^ b) & Info::sign_mask);
    if (op == FloatOp::Div) {
        if ((IsZero(a) && IsZero(b)) || (FP::IsInf(a) && FP::IsInf(b))) {
            FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
            return Info::DefaultNaN();
        }
        if (FP::IsInf(a)) return Info::Infinity(negative);
        if (IsZero(b)) {
            FP::FPProcessException(FP::FPExc::DivideByZero, fpcr, fpsr);
            return Info::Infinity(negative);
        }
        if (IsZero(a) || FP::IsInf(b)) return Info::Zero(negative);
        const auto [type_a, sign_a, value_a] = FP::FPUnpack(a, fpcr, fpsr);
        const auto [type_b, sign_b, value_b] = FP::FPUnpack(b, fpcr, fpsr);
        u64 quotient = 0, remainder = value_a.mantissa;
        for (int bit = 62; bit >= 0; --bit) {
            if (remainder >= value_b.mantissa) {
                remainder -= value_b.mantissa;
                quotient |= u64{1} << bit;
            }
            if (bit) remainder <<= 1;
        }
        quotient |= remainder != 0;
        return FP::FPRound<T>({negative, value_a.exponent - value_b.exponent, quotient}, fpcr, fpsr);
    }
    if (IsZero(a)) return a;
    if (a & Info::sign_mask) {
        FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return Info::DefaultNaN();
    }
    if (FP::IsInf(a)) return a;
    const auto [type, sign, value] = FP::FPUnpack(a, fpcr, fpsr);
    const unsigned odd = unsigned(value.exponent) & 1;
    const u128 radicand = u128(value.mantissa) << (62 + odd);
    u64 root = 0;
    for (int bit = 62; bit >= 0; --bit) {
        const u64 candidate = root | (u64{1} << bit);
        if (Multiply64To128(candidate, candidate) <= radicand) root = candidate;
    }
    const bool remainder = Multiply64To128(root, root) != radicand;
    return FP::FPRound<T>({false, (value.exponent - int(odd)) / 2, root | u64(remainder)}, fpcr, fpsr);
}

template<typename T>
u32 Compare(T a, T b, bool signaling, FP::FPCR fpcr, FP::FPSR& fpsr) {
    a = Input(a, fpcr, fpsr); b = Input(b, fpcr, fpsr);
    if (FP::IsNaN(a) || FP::IsNaN(b)) {
        if (signaling || FP::IsSNaN(a) || FP::IsSNaN(b)) FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return 0x30000000;
    }
    if (a == b || (IsZero(a) && IsZero(b))) return 0x60000000;
    const T sign = FP::FPInfo<T>::sign_mask;
    const T x = a & sign ? ~a : a | sign, y = b & sign ? ~b : b | sign;
    return x < y ? 0x80000000 : 0x20000000;
}

template<typename T>
T FromFixed(u64 value, unsigned bits, unsigned fbits, bool uns, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const bool negative = !uns && Signed(value, bits) < 0;
    const u64 magnitude = negative ? (0 - value) & Mask(bits) : value;
    if (!magnitude) return 0;
    const unsigned highest = 63 - std::countl_zero(magnitude);
    const u64 mantissa = highest == 63 ? (magnitude >> 1) | (magnitude & 1) : magnitude << (62 - highest);
    return FP::FPRound<T>({negative, int(highest) - int(fbits), mantissa}, fpcr, fpsr);
}

template<typename T>
u64 ToFixed(unsigned bits, T source, unsigned fractional, bool uns, FP::FPCR fpcr,
            FP::RoundingMode rounding, FP::FPSR& fpsr) {
    // Apply the unsigned range check after rounding. A small negative input
    // that rounds to zero is inexact, not invalid. Older FPToFixed helpers
    // rejected every negative unsigned conversion before this rounding step.
    const auto [type, negative, value] = FP::FPUnpack(source, fpcr, fpsr);
    if (type == FP::FPType::SNaN || type == FP::FPType::QNaN) {
        FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return 0;
    }
    const u64 limit = uns ? Mask(bits) : negative ? u64{1} << (bits - 1) : Mask(bits - 1);
    if (type == FP::FPType::Infinity) {
        FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return negative ? (uns ? 0 : limit) : limit;
    }
    if (!value.mantissa) return 0;
    const int shift = 62 - value.exponent - int(fractional);
    u64 magnitude = 0;
    bool inexact = false, above_half = false, equal_half = false, overflow = false;
    if (shift <= 0) {
        overflow = value.exponent + int(fractional) >= 64;
        if (!overflow) magnitude = value.mantissa << (-shift);
    } else if (shift >= 64) {
        inexact = true;
    } else {
        magnitude = value.mantissa >> shift;
        const u64 remainder = value.mantissa & Mask(unsigned(shift));
        const u64 half = u64{1} << (shift - 1);
        inexact = remainder != 0;
        above_half = remainder > half;
        equal_half = remainder == half;
    }
    bool increment = false;
    switch (rounding) {
    case FP::RoundingMode::ToNearest_TieEven: increment = above_half || (equal_half && (magnitude & 1)); break;
    case FP::RoundingMode::ToNearest_TieAwayFromZero: increment = above_half || equal_half; break;
    case FP::RoundingMode::TowardsPlusInfinity: increment = inexact && !negative; break;
    case FP::RoundingMode::TowardsMinusInfinity: increment = inexact && negative; break;
    default: break;
    }
    if (increment) { ++magnitude; overflow |= magnitude == 0; }
    if (overflow || magnitude > limit || (uns && negative && magnitude != 0)) {
        FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return uns && negative ? 0 : limit;
    }
    if (inexact) FP::FPProcessException(FP::FPExc::Inexact, fpcr, fpsr);
    return (negative ? 0 - magnitude : magnitude) & Mask(bits);
}

template<typename Fn>
u64 WithFloat(unsigned bits, Fn&& fn) {
    if (bits == 16) return fn.template operator()<u16>();
    if (bits == 32) return fn.template operator()<u32>();
    return fn.template operator()<u64>();
}

template<typename T>
T RoundRange(T source, unsigned integer_bits, FP::RoundingMode rounding, FP::FPCR fpcr, FP::FPSR& fpsr) {
    using Info = FP::FPInfo<T>;
    const T limit = T(integer_bits - 1 + Info::exponent_bias) << Info::explicit_mantissa_width;
    const T minimum = limit | Info::sign_mask;
    const T value = Input(source, fpcr, fpsr);
    if (FP::IsNaN(value) || FP::IsInf(value)) {
        FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return minimum;
    }
    // Check the rounded integer range before raising Inexact. FRINT32/64 X
    // and Z forms all report Inexact on a valid fractional input; out-of-range
    // results report only Invalid and encode the most-negative integer.
    const T rounded = FP::FPRoundInt(value, fpcr, rounding, false, fpsr);
    const T magnitude = rounded & ~Info::sign_mask;
    if (magnitude > limit || (magnitude == limit && !(rounded & Info::sign_mask))) {
        FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        return minimum;
    }
    if (rounded != value)
        FP::FPProcessException(FP::FPExc::Inexact, fpcr, fpsr);
    return rounded;
}

void RoundRanges(A64JitState* s, u32 i, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = i & 31, n = (i >> 5) & 31;
    const bool scalar = (i >> 28) & 1;
    const unsigned bits = 32 << ((i >> 22) & 1);
    const unsigned width = scalar ? bits : (i & 0x40000000) ? 128 : 64;
    const unsigned integer_bits = 32 << ((i >> (scalar ? 16 : 12)) & 1);
    const bool current_rounding = (i >> (scalar ? 15 : 29)) & 1;
    const auto mode = current_rounding ? fpcr.RMode() : FP::RoundingMode::TowardsZero;
    const Vector input = Read(s, n);
    Vector result;
    for (unsigned lane = 0; lane < width / bits; ++lane)
        result.Set(bits, lane, WithFloat(bits, [&]<typename T> {
            return RoundRange<T>(T(input.Get(bits, lane)), integer_bits, mode, fpcr, fpsr);
        }));
    Write(s, d, result, width);
}

void JavascriptConvert(A64JitState* s, u32 i, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = i & 31, n = (i >> 5) & 31;
    const u64 original = Read(s, n).word[0];
    const auto [type, negative, value] = FP::FPUnpack(original, fpcr, fpsr);
    u32 result = 0;
    bool exact = false;
    if (type == FP::FPType::SNaN || type == FP::FPType::QNaN || type == FP::FPType::Infinity) {
        FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
    } else if (!value.mantissa) {
        // Negative zero and positive denormals flushed to zero are not exact
        // JavaScript integer inputs, but do not raise an Inexact exception.
        exact = !negative && !(original & FP::FPInfo<u64>::mantissa_mask);
    } else {
        const int shift = 62 - value.exponent;
        u64 magnitude = 0;
        bool fractional = false;
        if (shift >= 64) fractional = true;
        else if (shift >= 0) {
            magnitude = value.mantissa >> shift;
            fractional = (value.mantissa & Mask(unsigned(shift))) != 0;
            result = u32(magnitude);
        } else {
            if (shift > -32) result = u32(value.mantissa) << -shift;
            if (value.exponent < 64) magnitude = value.mantissa << -shift;
        }
        if (negative) result = 0u - result;
        const bool invalid = value.exponent >= 64 || magnitude > (negative ? 0x80000000ull : 0x7fffffffull);
        if (invalid) FP::FPProcessException(FP::FPExc::InvalidOp, fpcr, fpsr);
        else if (fractional) FP::FPProcessException(FP::FPExc::Inexact, fpcr, fpsr);
        else exact = true;
    }
    SetGpr(s, d, 32, result);
    s->SetPstate(exact ? 0x40000000 : 0);
}

u32 WidenHalf(u16 half) {
    // Exact bit conversion without quieting an SNaN: FPMulAdd must select and
    // process all three NaNs together, preserving architectural NaN priority.
    const u32 sign = u32(half & 0x8000) << 16;
    const u32 exponent = (half >> 10) & 31, fraction = half & 1023;
    if (exponent == 31) return sign | 0x7f800000 | (fraction << 13);
    if (exponent) return sign | ((exponent + 112) << 23) | (fraction << 13);
    if (!fraction) return sign;
    const unsigned shift = std::countl_zero(fraction) - 21;
    return sign | ((113 - shift) << 23) | (((fraction << shift) & 1023) << 13);
}

void LongFloat(A64JitState* s, u32 i, bool indexed, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & (indexed ? 15 : 31);
    const unsigned width = (i & 0x40000000) ? 128 : 64;
    const unsigned start = ((i >> 29) & 1) * (width / 32);
    const unsigned index = (((i >> 11) & 1) << 2) | (((i >> 21) & 1) << 1) | ((i >> 20) & 1);
    const bool subtract = (i >> (indexed ? 14 : 23)) & 1;
    const Vector a = Read(s, n), b = Read(s, m), old = Read(s, d);
    Vector result;
    for (unsigned lane = 0; lane < width / 32; ++lane) {
        const u16 aa = u16(a.Get(16, start + lane)) ^ (subtract ? 0x8000 : 0);
        const u16 bb = u16(b.Get(16, indexed ? index : start + lane));
        const u32 x = WidenHalf(Input(aa, fpcr, fpsr)), y = WidenHalf(Input(bb, fpcr, fpsr));
        result.Set(32, lane, FP::FPMulAdd<u32>(u32(old.Get(32, lane)), x, y, fpcr, fpsr));
    }
    Write(s, d, result, width);
}

u64 ExpandFloat(unsigned imm, unsigned bits) {
    const unsigned exponent_bits = bits == 16 ? 5 : bits == 32 ? 8 : 11;
    const unsigned fraction_bits = bits - exponent_bits - 1;
    const u64 b = (imm >> 6) & 1;
    const u64 exponent = ((!b) << (exponent_bits - 1)) | (b ? Mask(exponent_bits - 3) << 2 : 0) | ((imm >> 4) & 3);
    return (u64(imm >> 7) << (bits - 1)) | (exponent << fraction_bits) | (u64(imm & 15) << (fraction_bits - 4));
}

void ScalarFp(A64JitState* s, u32 i, Kind kind, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31;
    const unsigned type = (i >> 22) & 3, bits = type == 2 ? 64 : FpBits(type);
    const unsigned ibits = i >> 31 ? 64 : 32;
    const unsigned op = (i >> 16) & 31;
    const u64 a = Read(s, n).Get(bits, 0), b = Read(s, m).Get(bits, 0);
    Vector result;
    if (kind == Kind::FpSelect) {
        result.word[0] = Condition(s->GetPstate(), (i >> 12) & 15) ? a : b;
        Write(s, d, result, bits);
        return;
    }
    if (kind == Kind::FpImmediate) {
        result.word[0] = ExpandFloat((i >> 13) & 255, bits);
        Write(s, d, result, bits);
        return;
    }
    if (kind == Kind::FpCompare || kind == Kind::FpCondCompare) {
        if (kind == Kind::FpCondCompare && !Condition(s->GetPstate(), (i >> 12) & 15)) {
            s->SetPstate((i & 15) << 28);
        } else {
            const u64 rhs = kind == Kind::FpCompare && (i & 8) ? 0 : b;
            s->SetPstate(u32(WithFloat(bits, [&]<typename T> { return Compare<T>(T(a), T(rhs), i & 16, fpcr, fpsr); })));
        }
        return;
    }
    if (kind == Kind::FpInteger && (op == 6 || op == 7 || op == 14 || op == 15)) {
        if (op & 1) {
            if (op == 15) {
                s->vec[d * 2 + 1] = Gpr(s, n, 64);
            } else {
                result.word[0] = Gpr(s, n, ibits);
                Write(s, d, result, bits);
            }
        } else {
            SetGpr(s, d, ibits, op == 14 ? s->vec[n * 2 + 1] : a);
        }
        return;
    }
    if (kind == Kind::FpInteger || kind == Kind::FpFixed) {
        const unsigned fbits = kind == Kind::FpFixed ? 64 - ((i >> 10) & 63) : 0;
        if (op == 2 || op == 3) {
            result.word[0] = WithFloat(bits, [&]<typename T> { return FromFixed<T>(Gpr(s, n, ibits), ibits, fbits, op & 1, fpcr, fpsr); });
            Write(s, d, result, bits);
        } else {
            const FP::RoundingMode rounding = op == 4 || op == 5 ? FP::RoundingMode::ToNearest_TieAwayFromZero : FP::RoundingMode(op >> 3);
            SetGpr(s, d, ibits, WithFloat(bits, [&]<typename T> { return ToFixed(ibits, T(a), fbits, op & 1, fpcr, rounding, fpsr); }));
        }
        return;
    }
    if (kind == Kind::FpUnary) {
        const unsigned unary = (i >> 15) & 63;
        if (unary == 4 || unary == 5 || unary == 7) {
            const unsigned to = FpBits(unary & 3);
            result.word[0] = WithFloat(to, [&]<typename T> { return WithFloat(bits, [&]<typename S> {
                if constexpr (std::is_same_v<T, S>) return T(a);
                else return FP::FPConvert<T>(S(a), fpcr, fpcr.RMode(), fpsr);
            }); });
            Write(s, d, result, to);
            return;
        }
        result.word[0] = WithFloat(bits, [&]<typename T>() -> u64 {
            const T value = T(a), sign = FP::FPInfo<T>::sign_mask;
            if (unary == 0) return value;
            if (unary == 1) return value & ~sign;
            if (unary == 2) return value ^ sign;
            if (unary == 3) return Arithmetic<T>(value, T(0), FloatOp::Sqrt, fpcr, fpsr);
            const FP::RoundingMode mode = unary <= 11 ? FP::RoundingMode(unary - 8) :
                                          unary == 12 ? FP::RoundingMode::ToNearest_TieAwayFromZero : fpcr.RMode();
            return FP::FPRoundInt(value, fpcr, mode, unary == 14, fpsr);
        });
    } else if (kind == Kind::FpBinary) {
        constexpr FloatOp ops[] = {FloatOp::Mul, FloatOp::Div, FloatOp::Add, FloatOp::Sub,
                                  FloatOp::Max, FloatOp::Min, FloatOp::MaxNum, FloatOp::MinNum, FloatOp::Mul};
        const unsigned binary = (i >> 12) & 15;
        result.word[0] = WithFloat(bits, [&]<typename T> {
            const T value = Arithmetic<T>(T(a), T(b), ops[binary], fpcr, fpsr);
            return binary == 8 ? T(value ^ FP::FPInfo<T>::sign_mask) : value;
        });
    } else {
        const u64 addend = Read(s, (i >> 10) & 31).Get(bits, 0);
        const bool negate_addend = (i >> 21) & 1;
        const bool negate_product = ((i >> 15) ^ (i >> 21)) & 1;
        result.word[0] = WithFloat(bits, [&]<typename T> {
            const T sign = FP::FPInfo<T>::sign_mask;
            return FP::FPMulAdd<T>(T(addend) ^ (negate_addend ? sign : 0), T(a) ^ (negate_product ? sign : 0), T(b), fpcr, fpsr);
        });
    }
    Write(s, d, result, bits);
}

void CopyAndPermute(A64JitState* s, u32 i, Kind kind) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const unsigned width = q ? 128 : 64;
    const Vector a = Read(s, n), b = Read(s, m);
    Vector result;
    if (kind == Kind::Copy) {
        const unsigned imm5 = (i >> 16) & 31, size = std::countr_zero(imm5);
        const unsigned bits = 8 << size, index = imm5 >> (size + 1), op = (i >> 11) & 15;
        if (scalar) {
            result.word[0] = a.Get(bits, index);
            Write(s, d, result, bits);
        } else if (u) {
            result = Read(s, d);
            result.Set(bits, index, a.Get(bits, op >> size));
            Write(s, d, result);
        } else if (op == 0 || op == 1) {
            const u64 lane = op == 0 ? a.Get(bits, index) : Gpr(s, n, bits);
            for (unsigned at = 0; at < width / bits; ++at) result.Set(bits, at, lane);
            Write(s, d, result, width);
        } else if (op == 3) {
            result = Read(s, d);
            result.Set(bits, index, Gpr(s, n, bits));
            Write(s, d, result);
        } else {
            const u64 value = a.Get(bits, index);
            SetGpr(s, d, q ? 64 : 32, op == 5 ? u64(Signed(value, bits)) : value);
        }
        return;
    }
    if (kind == Kind::Table) {
        const unsigned bytes = (((i >> 13) & 3) + 1) * 16;
        const bool extend = i & 0x1000;
        if (extend) result = Read(s, d);
        for (unsigned lane = 0; lane < width / 8; ++lane) {
            const unsigned at = unsigned(b.Get(8, lane));
            if (at < bytes) result.Set(8, lane, Read(s, (n + at / 16) & 31).Get(8, at % 16));
            else if (!extend) result.Set(8, lane, 0);
        }
    } else if (kind == Kind::Extract) {
        const unsigned offset = (i >> 11) & 15, count = width / 8;
        for (unsigned lane = 0; lane < count; ++lane)
            result.Set(8, lane, lane + offset < count ? a.Get(8, lane + offset) : b.Get(8, lane + offset - count));
    } else {
        const unsigned bits = 8 << ((i >> 22) & 3), count = width / bits;
        const unsigned op = (i >> 12) & 7, part = op >> 2;
        for (unsigned lane = 0; lane < count; ++lane) {
            switch (op & 3) {
            case 1: result.Set(bits, lane, (lane < count / 2 ? a : b).Get(bits, (lane % (count / 2)) * 2 + part)); break;
            case 2: result.Set(bits, lane, (lane & 1 ? b : a).Get(bits, (lane & ~1u) + part)); break;
            case 3: result.Set(bits, lane, (lane & 1 ? b : a).Get(bits, lane / 2 + part * count / 2)); break;
            }
        }
    }
    Write(s, d, result, width);
}

u64 PolynomialMultiply(u64 a, u64 b, unsigned bits) {
    u64 result = 0;
    for (unsigned k = 0; k < bits; ++k) if ((b >> k) & 1) result ^= a << k;
    return result;
}

void SameInteger(A64JitState* s, u32 i, bool& qc) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const unsigned bits = 8 << ((i >> 22) & 3), op = (i >> 11) & 31;
    const unsigned width = scalar ? bits : q ? 128 : 64, count = width / bits;
    const Vector a = Read(s, n), b = Read(s, m), old = Read(s, d);
    Vector result;
    if (op == 3) {
        const unsigned x = (i >> 22) & 3;
        for (unsigned lane = 0; lane < (q ? 2u : 1u); ++lane) {
            const u64 aa = a.word[lane], bb = b.word[lane], dd = old.word[lane];
            if (!u) {
                switch (x) {
                case 0: result.word[lane] = aa & bb; break;
                case 1: result.word[lane] = aa & ~bb; break;
                case 2: result.word[lane] = aa | bb; break;
                case 3: result.word[lane] = aa | ~bb; break;
                }
            } else {
                switch (x) {
                case 0: result.word[lane] = aa ^ bb; break;
                case 1: result.word[lane] = (aa & dd) | (bb & ~dd); break;
                case 2: result.word[lane] = (dd & ~bb) | (aa & bb); break;
                case 3: result.word[lane] = (dd & bb) | (aa & ~bb); break;
                }
            }
        }
        Write(s, d, result, q ? 128 : 64);
        return;
    }
    for (unsigned lane = 0; lane < count; ++lane) {
        u64 aa = a.Get(bits, lane), bb = b.Get(bits, lane), dd = old.Get(bits, lane), value = 0;
        if (op == 20 || op == 21 || op == 23) {
            const Vector& source = lane < count / 2 ? a : b;
            aa = source.Get(bits, 2 * (lane % (count / 2)));
            bb = source.Get(bits, 2 * (lane % (count / 2)) + 1);
        }
        const s64 sa = Signed(aa, bits), sb = Signed(bb, bits);
        switch (op) {
        case 0: value = u ? (aa + bb) >> 1 : u64((sa + sb) >> 1); break;
        case 1: value = SatAdd(aa, bb, bits, u, false, qc); break;
        case 2: value = u ? (aa + bb + 1) >> 1 : u64((sa + sb + 1) >> 1); break;
        case 4: value = u ? (aa - bb) >> 1 : u64((sa - sb) >> 1); break;
        case 5: value = SatAdd(aa, bb, bits, u, true, qc); break;
        case 6: value = (u ? aa > bb : sa > sb) ? Mask(bits) : 0; break;
        case 7: value = (u ? aa >= bb : sa >= sb) ? Mask(bits) : 0; break;
        case 8: case 9: case 10: case 11:
            value = ShiftLeft(aa, bits, std::bit_cast<s8>(u8(bb)), u, op >= 10, op & 1, qc);
            break;
        case 12: case 20: value = (u ? aa > bb : sa > sb) ? aa : bb; break;
        case 13: case 21: value = (u ? aa < bb : sa < sb) ? aa : bb; break;
        case 14: case 15:
            value = u ? (aa > bb ? aa - bb : bb - aa) : (sa > sb ? u64(sa - sb) : u64(sb - sa));
            if (op == 15) value += dd;
            break;
        case 16: value = u ? aa - bb : aa + bb; break;
        case 17: value = (u ? aa == bb : (aa & bb) != 0) ? Mask(bits) : 0; break;
        case 18: value = u ? dd - aa * bb : dd + aa * bb; break;
        case 19: value = u ? PolynomialMultiply(aa, bb, bits) : aa * bb; break;
        case 22: value = DoublingMultiply(aa, bb, bits, true, u, qc); break;
        case 23: value = aa + bb; break;
        }
        result.Set(bits, lane, value);
    }
    Write(s, d, result, width);
}

template<typename T>
T VectorFloatLane(T a, T b, T old, unsigned op, bool u, bool upper, FP::FPCR fpcr, FP::FPSR& fpsr) {
    switch (op) {
    case 24: return Arithmetic<T>(a, b, upper ? FloatOp::MinNum : FloatOp::MaxNum, fpcr, fpsr);
    case 25: return FP::FPMulAdd<T>(old, T(a ^ (upper ? FP::FPInfo<T>::sign_mask : 0)), b, fpcr, fpsr);
    case 26: {
        const T value = Arithmetic<T>(a, b, upper ? FloatOp::Sub : FloatOp::Add, fpcr, fpsr);
        return u && upper ? T(value & ~FP::FPInfo<T>::sign_mask) : value;
    }
    case 27: return Arithmetic<T>(a, b, u ? FloatOp::Mul : FloatOp::MulX, fpcr, fpsr);
    case 28: case 29: {
        if (op == 29) { a &= ~FP::FPInfo<T>::sign_mask; b &= ~FP::FPInfo<T>::sign_mask; }
        const u32 cmp = Compare<T>(a, b, u, fpcr, fpsr);
        return (!u ? cmp == 0x60000000 : upper ? cmp == 0x20000000 : (cmp & 0x90000000) == 0) ? T(~T{0}) : 0;
    }
    case 30: return Arithmetic<T>(a, b, upper ? FloatOp::Min : FloatOp::Max, fpcr, fpsr);
    case 31:
        if (u) return Arithmetic<T>(a, b, FloatOp::Div, fpcr, fpsr);
        return upper ? FP::FPRSqrtStepFused<T>(a, b, fpcr, fpsr) : FP::FPRecipStepFused<T>(a, b, fpcr, fpsr);
    }
    return 0;
}

void SameFloat(A64JitState* s, u32 i, FP::FPCR fpcr, FP::FPSR& fpsr, bool half = false) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1, upper = (i >> 23) & 1;
    const unsigned bits = half ? 16 : (i >> 22) & 1 ? 64 : 32, op = ((i >> 11) & 31) + (half ? 24 : 0);
    const unsigned width = scalar ? bits : q ? 128 : 64, count = width / bits;
    const bool pair = u && (op == 24 || op == 30 || (op == 26 && !upper));
    const Vector a = Read(s, n), b = Read(s, m), old = Read(s, d);
    Vector result;
    for (unsigned lane = 0; lane < count; ++lane) {
        u64 aa = a.Get(bits, lane), bb = b.Get(bits, lane);
        if (pair) {
            const Vector& source = lane < count / 2 ? a : b;
            aa = source.Get(bits, (lane % (count / 2)) * 2);
            bb = source.Get(bits, (lane % (count / 2)) * 2 + 1);
        }
        result.Set(bits, lane, WithFloat(bits, [&]<typename T> {
            return VectorFloatLane<T>(T(aa), T(bb), T(old.Get(bits, lane)), op, u, upper, fpcr, fpsr);
        }));
    }
    Write(s, d, result, width);
}

void Different(A64JitState* s, u32 i, bool& qc) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const unsigned bits = 8 << ((i >> 22) & 3), op = (i >> 12) & 15;
    const Vector a = Read(s, n), b = Read(s, m), old = Read(s, d);
    Vector result;
    if (bits == 64) {
        // PMULL, including polynomial64 multiplication, needs no PCLMUL host
        // feature and preserves all 128 product bits.
        const u64 aa = a.word[q], bb = b.word[q];
        for (unsigned bit = 0; bit < 64; ++bit) {
            if ((bb >> bit) & 1) {
                result.word[0] ^= aa << bit;
                if (bit) result.word[1] ^= aa >> (64 - bit);
            }
        }
        Write(s, d, result);
        return;
    }
    const bool narrow = op == 4 || op == 6;
    if (narrow && q) result = old;
    const unsigned count = scalar ? 1 : 64 / bits, wide = bits * 2, part = !scalar && q ? count : 0;
    for (unsigned lane = 0; lane < count; ++lane) {
        u64 aa = a.Get(bits, part + lane), bb = b.Get(bits, part + lane);
        const u64 dd = old.Get(wide, lane);
        u64 value = 0;
        if (narrow) {
            const u64 x = a.Get(wide, lane), y = b.Get(wide, lane);
            value = (op == 4 ? x + y : x - y) + (u ? u64{1} << (bits - 1) : 0);
            value = (value & Mask(wide)) >> bits;
            result.Set(bits, (q ? count : 0) + lane, value);
            continue;
        }
        const s64 sa = Signed(aa, bits), sb = Signed(bb, bits);
        if (!u) { aa = u64(sa); bb = u64(sb); }
        switch (op) {
        case 0: value = aa + bb; break;
        case 1: value = a.Get(wide, lane) + bb; break;
        case 2: value = aa - bb; break;
        case 3: value = a.Get(wide, lane) - bb; break;
        case 5: case 7:
            value = u ? (aa > bb ? aa - bb : bb - aa) : sa > sb ? u64(sa - sb) : u64(sb - sa);
            if (op == 5) value += dd;
            break;
        case 8: value = dd + aa * bb; break;
        case 10: value = dd - aa * bb; break;
        case 12: value = aa * bb; break;
        case 9: case 11: case 13:
            value = DoublingMultiply(a.Get(bits, part + lane), b.Get(bits, part + lane), bits, false, false, qc);
            if (op != 13) value = SatAdd(dd, value, wide, false, op == 11, qc);
            break;
        case 14: value = PolynomialMultiply(a.Get(bits, part + lane), b.Get(bits, part + lane), bits); break;
        }
        result.Set(wide, lane, value);
    }
    Write(s, d, result, scalar ? wide : narrow && !q ? 64 : 128);
}

u32 UnsignedEstimate(u32 value, bool sqrt) {
    // ARM's integer estimates are table-defined; these integer algorithms
    // produce the same eight-bit table entries without host FP dependence.
    if (!sqrt) {
        if (!(value & 0x80000000)) return 0xffffffff;
        const u32 a = value >> 23;
        const u32 b = (1u << 19) / (2 * a + 1);
        return ((b + 1) / 2) << 23;
    }
    if (!(value & 0xc0000000)) return 0xffffffff;
    const unsigned a = value >> 23;
    const unsigned scaled = a < 256 ? a * 2 + 1 : ((a >> 1) * 2 + 1) * 2;
    unsigned b = 512;
    while (u64(scaled) * (b + 1) * (b + 1) < (u64{1} << 28)) ++b;
    return ((b + 1) / 2) << 23;
}

void Misc(A64JitState* s, u32 i, bool& qc, FP::FPCR fpcr, FP::FPSR& fpsr, bool half = false) {
    const unsigned d = i & 31, n = (i >> 5) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const unsigned size = (i >> 22) & 3, op = (i >> 12) & 31;
    const unsigned bits = 8 << size, width = scalar ? bits : q ? 128 : 64;
    const Vector a = Read(s, n), old = Read(s, d);
    Vector result;
    if (op == 18 || op == 20) {
        const unsigned count = scalar ? 1 : 64 / bits;
        if (!scalar && q) result = old;
        for (unsigned lane = 0; lane < count; ++lane) {
            const u64 value = a.Get(bits * 2, lane);
            result.Set(bits, (!scalar && q ? count : 0) + lane,
                       op == 18 && !u ? value : Narrow(value, bits * 2, u && op == 20, u, qc));
        }
        Write(s, d, result, scalar ? bits : q ? 128 : 64);
        return;
    }
    if (op == 19) {
        const unsigned count = 64 / bits;
        for (unsigned lane = 0; lane < count; ++lane) result.Set(bits * 2, lane, a.Get(bits, (q ? count : 0) + lane) << bits);
        Write(s, d, result);
        return;
    }
    if (op == 22 || op == 23) {
        // FCVTN/FCVTXN write a lower/upper destination half; FCVTL reads one.
        const unsigned small = 16 << (size & 1), big = small * 2, count = scalar ? 1 : 64 / small;
        const bool widening = op == 23;
        if (!widening && q && !scalar) result = old;
        for (unsigned lane = 0; lane < count; ++lane) {
            const unsigned from = widening ? small : big, to = widening ? big : small;
            const u64 value = a.Get(from, lane + (widening && q ? count : 0));
            const u64 converted = WithFloat(to, [&]<typename T> {
                return WithFloat(from, [&]<typename S> {
                    if constexpr (std::is_same_v<T, S>) return T(value);
                    else return FP::FPConvert<T>(S(value), fpcr, u ? FP::RoundingMode::ToOdd : fpcr.RMode(), fpsr);
                });
            });
            result.Set(to, lane + (!widening && q && !scalar ? count : 0), converted);
        }
        Write(s, d, result, scalar ? small : widening || q ? 128 : 64);
        return;
    }
    if (op >= 24 || (op >= 12 && op <= 15)) {
        const unsigned fbits = half ? 16 : size & 1 ? 64 : 32, fwidth = scalar ? fbits : q ? 128 : 64;
        for (unsigned lane = 0; lane < fwidth / fbits; ++lane) {
            const u64 aa = a.Get(fbits, lane);
            result.Set(fbits, lane, WithFloat(fbits, [&]<typename T>() -> u64 {
                const T x = T(aa);
                const bool upper = size & 2;
                if (op == 15) return u ? x ^ FP::FPInfo<T>::sign_mask : x & ~FP::FPInfo<T>::sign_mask;
                if (op == 12 || op == 13 || op == 14) {
                    const u32 cmp = Compare<T>(x, T{0}, op != 13 || u, fpcr, fpsr);
                    bool yes;
                    if (op == 12) yes = u ? (cmp & 0x90000000) == 0 : cmp == 0x20000000;
                    else if (op == 13) yes = u ? cmp == 0x60000000 || cmp == 0x80000000 : cmp == 0x60000000;
                    else yes = cmp == 0x80000000;
                    return yes ? u64(~T{0}) : 0;
                }
                if (op == 24 || op == 25) {
                    const FP::RoundingMode mode = op == 24 ? (u ? FP::RoundingMode::ToNearest_TieAwayFromZero :
                                                              upper ? FP::RoundingMode::TowardsPlusInfinity : FP::RoundingMode::ToNearest_TieEven) :
                                                          u ? fpcr.RMode() : upper ? FP::RoundingMode::TowardsZero : FP::RoundingMode::TowardsMinusInfinity;
                    return FP::FPRoundInt(x, fpcr, mode, op == 25 && u && !upper, fpsr);
                }
                if (op == 26 || op == 27 || (op == 28 && !upper)) {
                    const FP::RoundingMode mode = op == 26 ? (upper ? FP::RoundingMode::TowardsPlusInfinity : FP::RoundingMode::ToNearest_TieEven) :
                                                  op == 27 ? (upper ? FP::RoundingMode::TowardsZero : FP::RoundingMode::TowardsMinusInfinity) :
                                                             FP::RoundingMode::ToNearest_TieAwayFromZero;
                    return ToFixed(fbits, x, 0, u, fpcr, mode, fpsr);
                }
                if (op == 28) return UnsignedEstimate(u32(x), u);
                if (op == 29) {
                    if (!upper) return FromFixed<T>(x, fbits, 0, u, fpcr, fpsr);
                    return u ? FP::FPRSqrtEstimate(x, fpcr, fpsr) : FP::FPRecipEstimate(x, fpcr, fpsr);
                }
                if (op == 31) return scalar ? FP::FPRecipExponent(x, fpcr, fpsr) : Arithmetic<T>(x, T{0}, FloatOp::Sqrt, fpcr, fpsr);
                return 0;
            }));
        }
        Write(s, d, result, fwidth);
        return;
    }
    for (unsigned lane = 0; lane < width / bits; ++lane) {
        const u64 aa = a.Get(bits, lane), dd = old.Get(bits, lane);
        const s64 sa = Signed(aa, bits);
        u64 value = 0;
        switch (op) {
        case 0: case 1: {
            const unsigned container = op == 1 ? 16 : u ? 32 : 64;
            const unsigned lane_mask = container / bits - 1;
            value = a.Get(bits, lane ^ lane_mask);
            break;
        }
        case 2: case 6:
            // These instructions have half as many, twice as wide results.
            if (lane >= width / (2 * bits)) continue;
            value = u ? a.Get(bits, 2 * lane) + a.Get(bits, 2 * lane + 1) :
                        u64(Signed(a.Get(bits, 2 * lane), bits) + Signed(a.Get(bits, 2 * lane + 1), bits));
            if (op == 6) value += old.Get(bits * 2, lane);
            result.Set(bits * 2, lane, value);
            continue;
        case 3:
            // SUQADD: signed destination plus unsigned source. USQADD has
            // unsigned destination plus signed source. Split by sign so no
            // signed64 overflow or conversion-dependent behavior is used.
            if (!u) {
                const s64 sd = Signed(dd, bits);
                const u64 maximum = Mask(bits - 1);
                if (sd >= 0) {
                    if (aa > maximum - dd) { qc = true; value = maximum; }
                    else value = aa + dd;
                } else {
                    const u64 magnitude = (0 - dd) & Mask(bits);
                    if (aa >= magnitude && aa - magnitude > maximum) { qc = true; value = maximum; }
                    else value = aa + dd;
                }
            } else if (sa < 0) {
                const u64 magnitude = (0 - aa) & Mask(bits);
                if (dd < magnitude) { qc = true; value = 0; }
                else value = dd - magnitude;
            } else {
                value = SatAdd(dd, aa, bits, true, false, qc);
            }
            break;
        case 4: {
            u64 x = aa;
            if (!u && sa < 0) x ^= Mask(bits);
            value = std::countl_zero(x) - (64 - bits) - (u ? 0 : 1);
            break;
        }
        case 5:
            if (!u) value = std::popcount(aa);
            else if (size == 0) value = ~aa;
            else {
                // RBIT operates on bytes despite the size=01 opcode field.
                for (unsigned byte = 0; byte < width / 8; ++byte) {
                    const u8 x = u8(a.Get(8, byte));
                    u8 y = 0;
                    for (unsigned bit = 0; bit < 8; ++bit) y |= ((x >> bit) & 1) << (7 - bit);
                    result.Set(8, byte, y);
                }
                Write(s, d, result, width);
                return;
            }
            break;
        case 7:
            if (aa == (u64{1} << (bits - 1))) { qc = true; value = Mask(bits - 1); }
            else value = u || sa < 0 ? 0 - aa : aa;
            break;
        case 8: value = (u ? sa >= 0 : sa > 0) ? Mask(bits) : 0; break;
        case 9: value = (u ? sa <= 0 : aa == 0) ? Mask(bits) : 0; break;
        case 10: value = sa < 0 ? Mask(bits) : 0; break;
        case 11: value = u || sa < 0 ? 0 - aa : aa; break;
        }
        result.Set(bits, lane, value);
    }
    Write(s, d, result, width);
}

void Reduce(A64JitState* s, u32 i, FP::FPCR fpcr, FP::FPSR& fpsr, bool half = false) {
    const unsigned d = i & 31, n = (i >> 5) & 31, op = (i >> 12) & 31, size = (i >> 22) & 3;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const Vector a = Read(s, n);
    Vector result;
    if (op == 12 || op == 13 || op == 15) {
        const unsigned bits = half ? 16 : (size & 1) ? 64 : 32;
        const FloatOp operation = op == 13 ? FloatOp::Add : op == 12 ? (size & 2 ? FloatOp::MinNum : FloatOp::MaxNum) :
                                                                                      (size & 2 ? FloatOp::Min : FloatOp::Max);
        result.word[0] = WithFloat(bits, [&]<typename T> {
            const T low = Arithmetic<T>(T(a.Get(bits, 0)), T(a.Get(bits, 1)), operation, fpcr, fpsr);
            if (scalar) return low;
            const T high = Arithmetic<T>(T(a.Get(bits, 2)), T(a.Get(bits, 3)), operation, fpcr, fpsr);
            const T lower_half = Arithmetic<T>(low, high, operation, fpcr, fpsr);
            if (!half || !q) return lower_half;
            const T third = Arithmetic<T>(T(a.Get(bits, 4)), T(a.Get(bits, 5)), operation, fpcr, fpsr);
            const T fourth = Arithmetic<T>(T(a.Get(bits, 6)), T(a.Get(bits, 7)), operation, fpcr, fpsr);
            return Arithmetic<T>(lower_half, Arithmetic<T>(third, fourth, operation, fpcr, fpsr), operation, fpcr, fpsr);
        });
        Write(s, d, result, bits);
        return;
    }
    const unsigned bits = 8 << size, count = scalar ? 2 : (q ? 128 : 64) / bits;
    u64 value = op == 10 || op == 26 ? a.Get(bits, 0) : 0;
    for (unsigned lane = 0; lane < count; ++lane) {
        const u64 x = a.Get(bits, lane);
        if (op == 3 || op == 27) value += op == 3 && !u ? u64(Signed(x, bits)) : x;
        else if ((u ? x > value : Signed(x, bits) > Signed(value, bits)) == (op == 10)) value = x;
    }
    result.word[0] = value;
    Write(s, d, result, op == 3 ? bits * 2 : bits);
}

void Immediate(A64JitState* s, u32 i) {
    const unsigned d = i & 31, cmode = (i >> 12) & 15;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, half = i & 0x800;
    const unsigned imm = ((i >> 11) & 0xe0) | ((i >> 5) & 0x1f);
    unsigned bits;
    u64 value;
    bool merge = false;
    if (cmode < 8) { bits = 32; value = u64(imm) << ((cmode >> 1) * 8); merge = cmode & 1; }
    else if (cmode < 12) { bits = 16; value = u64(imm) << (((cmode >> 1) & 1) * 8); merge = cmode & 1; }
    else if (cmode < 14) { bits = 32; const unsigned shift = cmode == 12 ? 8 : 16; value = (u64(imm) << shift) | Mask(shift); }
    else if (cmode == 14 && !u) { bits = 8; value = imm; }
    else if (cmode == 14) {
        bits = 64; value = 0;
        for (unsigned k = 0; k < 8; ++k) if ((imm >> k) & 1) value |= u64{255} << (k * 8);
    } else { bits = half ? 16 : u ? 64 : 32; value = ExpandFloat(imm, bits); }
    if (cmode < 14 && u) value = ~value;
    Vector result = merge ? Read(s, d) : Vector{};
    for (unsigned lane = 0; lane < (q ? 128u : 64u) / bits; ++lane) {
        const u64 v = merge ? (u ? result.Get(bits, lane) & value : result.Get(bits, lane) | value) : value;
        result.Set(bits, lane, v);
    }
    Write(s, d, result, q ? 128 : 64);
}

void Shift(A64JitState* s, u32 i, bool& qc, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = i & 31, n = (i >> 5) & 31, op = (i >> 11) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const unsigned imm = (i >> 16) & 127, bits = 8u << (31 - std::countl_zero(imm >> 3));
    const unsigned right = bits * 2 - imm, left = imm - bits;
    const bool narrow = op >= 16 && op <= 19, widen = op == 20;
    const unsigned width = scalar ? bits : q ? 128 : 64;
    const Vector a = Read(s, n), old = Read(s, d);
    Vector result = narrow && !scalar && q ? old : Vector{};
    const unsigned count = scalar ? 1 : (narrow || widen ? 64 : width) / bits;
    for (unsigned lane = 0; lane < count; ++lane) {
        const u64 aa = a.Get(narrow ? bits * 2 : bits, lane + (widen && q ? count : 0));
        const u64 dd = old.Get(bits, lane);
        u64 value = 0;
        switch (op) {
        case 0: case 2: case 4: case 6:
            value = ShiftRight(aa, bits, right, u, op & 4);
            if (op & 2) value += dd;
            break;
        case 8:
            value = (dd & (Mask(bits) ^ Mask(bits - right))) | ShiftRight(aa, bits, right, true, false);
            break;
        case 10:
            value = aa << left;
            if (u) value |= dd & Mask(left);
            break;
        case 12:
            if (Signed(aa, bits) < 0) { qc = true; value = 0; }
            else value = ShiftLeft(aa, bits, int(left), true, false, true, qc);
            break;
        case 14: value = ShiftLeft(aa, bits, int(left), u, false, true, qc); break;
        case 16: case 17: case 18: case 19: {
            const bool input_unsigned = u && op >= 18;
            value = ShiftRight(aa, bits * 2, right, input_unsigned || (!u && op < 18), op & 1);
            if (u || op >= 18) value = Narrow(value, bits * 2, input_unsigned, u, qc);
            break;
        }
        case 20: value = (u ? aa : u64(Signed(aa, bits))) << left; break;
        case 28:
            value = WithFloat(bits, [&]<typename T> { return FromFixed<T>(aa, bits, right, u, fpcr, fpsr); });
            break;
        case 31:
            value = WithFloat(bits, [&]<typename T> { return ToFixed(bits, T(aa), right, u, fpcr, FP::RoundingMode::TowardsZero, fpsr); });
            break;
        }
        result.Set(widen ? bits * 2 : bits, lane + (narrow && !scalar && q ? count : 0), value);
    }
    Write(s, d, result, scalar ? bits : widen || (narrow && q) ? 128 : narrow ? 64 : width);
}

void Indexed(A64JitState* s, u32 i, bool& qc, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = i & 31, n = (i >> 5) & 31, size = (i >> 22) & 3, op = (i >> 12) & 15;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const unsigned h = (i >> 11) & 1, l = (i >> 21) & 1, mm = (i >> 20) & 1;
    const bool fp = op == 1 || op == 5 || op == 9;
    const unsigned bits = fp && size == 0 ? 16 : 8 << size;
    const unsigned m = ((i >> 16) & 15) | (bits == 16 ? 0 : mm << 4);
    const unsigned index = bits == 16 ? h * 4 + l * 2 + mm : bits == 32 ? h * 2 + l : h;
    const bool wide = !fp && (op == 2 || op == 3 || op == 6 || op == 7 || op == 10 || op == 11);
    const unsigned width = scalar ? bits : q ? 128 : 64, count = scalar ? 1 : (wide ? 64 : width) / bits;
    const Vector a = Read(s, n), b = Read(s, m), old = Read(s, d);
    Vector result;
    if (op == 14) {
        for (unsigned lane = 0; lane < width / 32; ++lane) {
            u32 value = u32(old.Get(32, lane));
            for (unsigned k = 0; k < 4; ++k) {
                const u8 aa = u8(a.Get(8, lane * 4 + k)), bb = u8(b.Get(8, index * 4 + k));
                value += u ? unsigned(aa) * bb : u32(int(std::bit_cast<s8>(aa)) * int(std::bit_cast<s8>(bb)));
            }
            result.Set(32, lane, value);
        }
        Write(s, d, result, width);
        return;
    }
    for (unsigned lane = 0; lane < count; ++lane) {
        const u64 aa = a.Get(bits, lane + (wide && q && !scalar ? count : 0)), bb = b.Get(bits, index);
        const unsigned dest_bits = wide ? bits * 2 : bits;
        const u64 dd = old.Get(dest_bits, lane);
        u64 value = 0;
        if (fp) {
            value = WithFloat(bits, [&]<typename T> {
                if (op == 9) return Arithmetic<T>(T(aa), T(bb), u ? FloatOp::MulX : FloatOp::Mul, fpcr, fpsr);
                return FP::FPMulAdd<T>(T(dd), T(aa) ^ (op == 5 ? FP::FPInfo<T>::sign_mask : 0), T(bb), fpcr, fpsr);
            });
        } else {
            const u64 product = (u ? aa : u64(Signed(aa, bits))) * (u ? bb : u64(Signed(bb, bits)));
            switch (op) {
            case 0: case 2: value = dd + product; break;
            case 4: case 6: value = dd - product; break;
            case 8: case 10: value = product; break;
            case 3: case 7: case 11:
                value = DoublingMultiply(aa, bb, bits, false, false, qc);
                if (op != 11) value = SatAdd(dd, value, dest_bits, false, op == 7, qc);
                break;
            case 12: case 13: case 15:
                value = u ? RoundingMultiplyAccumulate(dd, aa, bb, bits, op == 15, qc) : DoublingMultiply(aa, bb, bits, true, op != 12, qc);
                break;
            }
        }
        result.Set(dest_bits, lane, value);
    }
    Write(s, d, result, scalar ? (wide ? bits * 2 : bits) : wide ? 128 : width);
}

void Extra(A64JitState* s, u32 i, bool& qc) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31, op = (i >> 11) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const unsigned bits = 8 << ((i >> 22) & 3), width = scalar ? bits : q ? 128 : 64;
    const Vector a = Read(s, n), b = Read(s, m), old = Read(s, d);
    Vector result;
    for (unsigned lane = 0; lane < width / bits; ++lane) {
        u64 value = old.Get(bits, lane);
        if (op == 18) {
            for (unsigned k = 0; k < 4; ++k) {
                const u8 aa = u8(a.Get(8, lane * 4 + k)), bb = u8(b.Get(8, lane * 4 + k));
                value += u ? unsigned(aa) * bb : u64(int(std::bit_cast<s8>(aa)) * int(std::bit_cast<s8>(bb)));
            }
        } else {
            value = RoundingMultiplyAccumulate(value, a.Get(bits, lane), b.Get(bits, lane), bits, op == 17, qc);
        }
        result.Set(bits, lane, value);
    }
    Write(s, d, result, width);
}

void Complex(A64JitState* s, u32 i, bool indexed, FP::FPCR fpcr, FP::FPSR& fpsr) {
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31, size = (i >> 22) & 3;
    const unsigned bits = 8 << size, width = (i >> 30) & 1 ? 128 : 64;
    const unsigned op = (i >> 11) & 31;
    const bool add = !indexed && (op == 28 || op == 30);
    const unsigned rotation = indexed ? (i >> 13) & 3 : add ? 1 + ((i >> 12) & 1) * 2 : (i >> 11) & 3;
    const unsigned index = size == 1 ? (((i >> 11) & 1) * 2 + ((i >> 21) & 1)) : (i >> 11) & 1;
    const Vector a = Read(s, n), b = Read(s, m), old = Read(s, d);
    Vector result;
    for (unsigned pair = 0; pair < width / bits / 2; ++pair) {
        const unsigned source_pair = indexed ? index : pair;
        for (unsigned component = 0; component < 2; ++component) {
            const u64 aa = a.Get(bits, pair * 2 + (add ? component : rotation & 1));
            const u64 bb = b.Get(bits, source_pair * 2 + (component ^ (rotation & 1)));
            const bool negative = rotation == 2 || (rotation == 1 && component == 0) || (rotation == 3 && component == 1);
            result.Set(bits, pair * 2 + component, WithFloat(bits, [&]<typename T> {
                const T multiplier = T(bb) ^ (negative ? FP::FPInfo<T>::sign_mask : 0);
                return add ? Arithmetic<T>(T(aa), multiplier, FloatOp::Add, fpcr, fpsr) :
                             FP::FPMulAdd<T>(T(old.Get(bits, pair * 2 + component)), T(aa), multiplier, fpcr, fpsr);
            }));
        }
    }
    Write(s, d, result, width);
}

void Execute(A64JitState* state, u32 instruction, Kind kind, FP::FPCR fpcr, FP::FPSR& fpsr) {
    bool qc = false;
    switch (kind) {
    case Kind::FpUnary: case Kind::FpBinary: case Kind::FpFma: case Kind::FpCompare:
    case Kind::FpCondCompare: case Kind::FpSelect: case Kind::FpImmediate:
    case Kind::FpInteger: case Kind::FpFixed:
        ScalarFp(state, instruction, kind, fpcr, fpsr); break;
    case Kind::FpRoundRange: RoundRanges(state, instruction, fpcr, fpsr); break;
    case Kind::FpJavascript: JavascriptConvert(state, instruction, fpcr, fpsr); break;
    case Kind::FpLong: LongFloat(state, instruction, false, fpcr, fpsr); break;
    case Kind::FpLongIndexed: LongFloat(state, instruction, true, fpcr, fpsr); break;
    case Kind::Copy: case Kind::Table: case Kind::Extract: case Kind::Permute:
        CopyAndPermute(state, instruction, kind); break;
    case Kind::Same:
        if (((instruction >> 11) & 31) >= 24) SameFloat(state, instruction, fpcr, fpsr);
        else SameInteger(state, instruction, qc);
        break;
    case Kind::HalfSame: SameFloat(state, instruction, fpcr, fpsr, true); break;
    case Kind::HalfMisc: Misc(state, instruction, qc, fpcr, fpsr, true); break;
    case Kind::HalfReduce: Reduce(state, instruction, fpcr, fpsr, true); break;
    case Kind::Complex: Complex(state, instruction, false, fpcr, fpsr); break;
    case Kind::ComplexIndexed: Complex(state, instruction, true, fpcr, fpsr); break;
    case Kind::Different: Different(state, instruction, qc); break;
    case Kind::Misc: Misc(state, instruction, qc, fpcr, fpsr); break;
    case Kind::Reduce: Reduce(state, instruction, fpcr, fpsr); break;
    case Kind::Immediate: Immediate(state, instruction); break;
    case Kind::Shift: Shift(state, instruction, qc, fpcr, fpsr); break;
    case Kind::Indexed: Indexed(state, instruction, qc, fpcr, fpsr); break;
    case Kind::Extra: Extra(state, instruction, qc); break;
    case Kind::Extension: ExecuteBaselineSimdExtra(state, instruction, fpcr, fpsr); break;
    case Kind::None: return;
    }
    state->fpsr_exc |= fpsr.Value() & 0x9f;
    if (qc) state->fpsr_qc = 1;
}

// C++ unwinding is contained in this veneer, never across generated code.
// All Common::FP routines involved are ordinary, non-noexcept C++ functions.
// Their existing FPCR enable tests call this handler before setting a trapped
// cumulative bit. Untrapped earlier exceptions remain visible in FPSR.
struct FloatingTrap { FP::FPExc exception; };
struct ScopedTrapHandler {
    FP::BaselineTrapHandler saved = FP::baseline_trap_handler;
    ScopedTrapHandler() {
        FP::baseline_trap_handler = [](FP::FPExc exception) { throw FloatingTrap{exception}; };
    }
    ~ScopedTrapHandler() { FP::baseline_trap_handler = saved; }
};

bool ExecuteWithTraps(A64JitState* state, u32 instruction, Kind kind) {
    // Only the uncommon enabled-trap path snapshots architectural destinations.
    // Do not snapshot/restore halt_reason: another thread may request a halt.
    const auto registers = state->reg;
    const auto vectors = state->vec;
    const u32 nzcv = state->cpsr_nzcv;
    const u32 qc = state->fpsr_qc;
    FP::FPSR fpsr{state->fpsr_exc};
    try {
        ScopedTrapHandler handler;
        Execute(state, instruction, kind, FP::FPCR{state->fpcr}, fpsr);
        return true;
    } catch (const FloatingTrap&) {
        state->reg = registers;
        state->vec = vectors;
        state->cpsr_nzcv = nzcv;
        state->fpsr_qc = qc;
        state->fpsr_exc = fpsr.Value() & 0x9f;
        if (state->runtime_callbacks)
            state->runtime_callbacks->ExceptionRaised(state->pc, A64::Exception::FloatingPointTrap);
        Atomic::Or(&state->halt_reason, u32(HaltReason::UserDefined1));
        return false;
    }
}

}  // namespace

bool BaselineSimdSupported(u32 instruction) {
    return Decode(instruction) != Kind::None;
}

bool ExecuteBaselineSimd(A64JitState* state, u32 instruction) {
    const Kind kind = Decode(instruction);
    if (kind == Kind::None) {
        if (state->runtime_callbacks)
            state->runtime_callbacks->ExceptionRaised(state->pc, A64::Exception::UnallocatedEncoding);
        Atomic::Or(&state->halt_reason, u32(HaltReason::UserDefined1));
        return false;
    }
    if (state->fpcr & 0x9f00)
        return ExecuteWithTraps(state, instruction, kind);
    FP::FPSR fpsr{state->fpsr_exc};
    Execute(state, instruction, kind, FP::FPCR{state->fpcr}, fpsr);
    return true;
}

}  // namespace Dynarmic::Backend::X64
