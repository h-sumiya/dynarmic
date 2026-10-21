// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
// Independent crypto instruction semantics. Pure AES/SM4 math utilities are
// reused; no guest decoder, IR construction or optimizing compiler is called.

#include "dynarmic/backend/x64/a64_baseline_crypto.h"
#include "dynarmic/backend/x64/a64_baseline_system.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/common/crypto/aes.h"
#include "dynarmic/common/crypto/sm4.h"

#include <array>
#include <bit>
#include <cstring>

namespace Dynarmic::Backend::X64 {
namespace {
enum class Op {
    None, AESE, AESD, AESMC, AESIMC,
    SHA1C, SHA1P, SHA1M, SHA1SU0, SHA1H, SHA1SU1,
    SHA256H, SHA256H2, SHA256SU0, SHA256SU1,
    SHA512H, SHA512H2, SHA512SU0, SHA512SU1,
    RAX1, XAR, EOR3, BCAX,
    SM3SS1, SM3TT1A, SM3TT1B, SM3TT2A, SM3TT2B, SM3PARTW1, SM3PARTW2,
    SM4E, SM4EKEY,
};
struct Encoding {
    Op op;
    u32 mask = 0, value = 0;
    template<std::size_t N>
    consteval Encoding(Op op_, const char (&bits)[N]) : op(op_) {
        static_assert(N == 33);
        for (unsigned i = 0; i < 32; ++i) {
            mask = (mask << 1) | (bits[i] == '0' || bits[i] == '1');
            value = (value << 1) | (bits[i] == '1');
        }
    }
};
constexpr Encoding encodings[] = {
    {Op::AESE, "0100111000101000010010nnnnnddddd"},
    {Op::AESD, "0100111000101000010110nnnnnddddd"},
    {Op::AESMC, "0100111000101000011010nnnnnddddd"},
    {Op::AESIMC, "0100111000101000011110nnnnnddddd"},
    {Op::SHA1C, "01011110000mmmmm000000nnnnnddddd"},
    {Op::SHA1P, "01011110000mmmmm000100nnnnnddddd"},
    {Op::SHA1M, "01011110000mmmmm001000nnnnnddddd"},
    {Op::SHA1SU0, "01011110000mmmmm001100nnnnnddddd"},
    {Op::SHA256H, "01011110000mmmmm010000nnnnnddddd"},
    {Op::SHA256H2, "01011110000mmmmm010100nnnnnddddd"},
    {Op::SHA256SU1, "01011110000mmmmm011000nnnnnddddd"},
    {Op::SHA1H, "0101111000101000000010nnnnnddddd"},
    {Op::SHA1SU1, "0101111000101000000110nnnnnddddd"},
    {Op::SHA256SU0, "0101111000101000001010nnnnnddddd"},
    {Op::SM3TT1A, "11001110010mmmmm10ii00nnnnnddddd"},
    {Op::SM3TT1B, "11001110010mmmmm10ii01nnnnnddddd"},
    {Op::SM3TT2A, "11001110010mmmmm10ii10nnnnnddddd"},
    {Op::SM3TT2B, "11001110010mmmmm10ii11nnnnnddddd"},
    {Op::SHA512H, "11001110011mmmmm100000nnnnnddddd"},
    {Op::SHA512H2, "11001110011mmmmm100001nnnnnddddd"},
    {Op::SHA512SU1, "11001110011mmmmm100010nnnnnddddd"},
    {Op::RAX1, "11001110011mmmmm100011nnnnnddddd"},
    {Op::SM3PARTW1, "11001110011mmmmm110000nnnnnddddd"},
    {Op::SM3PARTW2, "11001110011mmmmm110001nnnnnddddd"},
    {Op::SM4EKEY, "11001110011mmmmm110010nnnnnddddd"},
    {Op::XAR, "11001110100mmmmmiiiiiinnnnnddddd"},
    {Op::EOR3, "11001110000mmmmm0aaaaannnnnddddd"},
    {Op::BCAX, "11001110001mmmmm0aaaaannnnnddddd"},
    {Op::SM3SS1, "11001110010mmmmm0aaaaannnnnddddd"},
    {Op::SHA512SU0, "1100111011000000100000nnnnnddddd"},
    {Op::SM4E, "1100111011000000100001nnnnnddddd"},
};
Op Decode(u32 instruction) {
    const unsigned prefix = instruction >> 24;
    if (prefix != 0x4e && prefix != 0x5e && prefix != 0xce) return Op::None;
    for (const auto& e : encodings)
        if ((instruction & e.mask) == e.value) return e.op;
    return Op::None;
}

using V32 = std::array<u32, 4>;
using V64 = std::array<u64, 2>;
V64 Read(const A64JitState& state, unsigned r) { return {state.vec[r * 2], state.vec[r * 2 + 1]}; }
void Write(A64JitState& state, unsigned r, V64 value) {
    state.vec[r * 2] = value[0]; state.vec[r * 2 + 1] = value[1];
}
template<class T> T Choose(T x, T y, T z) { return (x & y) ^ (~x & z); }
template<class T> T Majority(T x, T y, T z) { return (x & y) ^ (x & z) ^ (y & z); }
u32 Sigma0(u32 x) { return std::rotr(x, 2) ^ std::rotr(x, 13) ^ std::rotr(x, 22); }
u32 Sigma1(u32 x) { return std::rotr(x, 6) ^ std::rotr(x, 11) ^ std::rotr(x, 25); }
u32 Small0(u32 x) { return std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3); }
u32 Small1(u32 x) { return std::rotr(x, 17) ^ std::rotr(x, 19) ^ (x >> 10); }
u64 Sigma0(u64 x) { return std::rotr(x, 28) ^ std::rotr(x, 34) ^ std::rotr(x, 39); }
u64 Sigma1(u64 x) { return std::rotr(x, 14) ^ std::rotr(x, 18) ^ std::rotr(x, 41); }
u64 Small0(u64 x) { return std::rotr(x, 1) ^ std::rotr(x, 8) ^ (x >> 7); }
u64 Small1(u64 x) { return std::rotr(x, 19) ^ std::rotr(x, 61) ^ (x >> 6); }

V32 Sha1Hash(V32 d, const V32& n, const V32& m, Op op) {
    u32 y = n[0];
    for (unsigned i = 0; i < 4; ++i) {
        const u32 f = op == Op::SHA1C ? Choose(d[1], d[2], d[3]) :
                      op == Op::SHA1M ? Majority(d[1], d[2], d[3]) : d[1] ^ d[2] ^ d[3];
        const u32 t = y + std::rotl(d[0], 5) + f + m[i];
        y = d[3];
        d = {t, d[0], std::rotr(d[1], 2), d[2]};
    }
    return d;
}
V32 Sha256Hash(V32 x, V32 y, const V32& m, bool first) {
    for (unsigned i = 0; i < 4; ++i) {
        const u32 t = y[3] + Sigma1(y[0]) + Choose(y[0], y[1], y[2]) + m[i];
        const u32 nx = t + Sigma0(x[0]) + Majority(x[0], x[1], x[2]);
        y = {t + x[3], y[0], y[1], y[2]};
        x = {nx, x[0], x[1], x[2]};
    }
    return first ? x : y;
}
V32 Sm4Hash(V32 d, const V32& keys, bool schedule) {
    for (unsigned i = 0; i < 4; ++i) {
        const u32 input = d[1] ^ d[2] ^ d[3] ^ keys[i];
        u32 t = 0;
        for (unsigned byte = 0; byte < 4; ++byte)
            t |= u32(Common::Crypto::SM4::AccessSubstitutionBox(u8(input >> (byte * 8)))) << (byte * 8);
        const u32 value = schedule ? d[0] ^ t ^ std::rotl(t, 13) ^ std::rotl(t, 23) :
            d[0] ^ t ^ std::rotl(t, 2) ^ std::rotl(t, 10) ^ std::rotl(t, 18) ^ std::rotl(t, 24);
        d = {d[1], d[2], d[3], value};
    }
    return d;
}
}  // namespace

bool BaselineCryptoSupported(u32 instruction) { return Decode(instruction) != Op::None; }

bool ExecuteBaselineCrypto(A64JitState* state, u32 instruction) {
    const Op op = Decode(instruction);
    const unsigned rd = instruction & 31, rn = (instruction >> 5) & 31, rm = (instruction >> 16) & 31;
    // Snapshot all operands before modifying the destination; every register
    // alias combination must have the same result as separate source registers.
    const V64 d64 = Read(*state, rd), n64 = Read(*state, rn), m64 = Read(*state, rm);
    const V32 d = std::bit_cast<V32>(d64), n = std::bit_cast<V32>(n64), m = std::bit_cast<V32>(m64);
    V32 result{};
    V64 result64{};
    bool wide = false;
    switch (op) {
    case Op::AESE: case Op::AESD: case Op::AESMC: case Op::AESIMC: {
        using namespace Common::Crypto::AES;
        auto value = std::bit_cast<State>(n64);
        State out{};
        if (op == Op::AESE || op == Op::AESD) {
            const auto dest = std::bit_cast<State>(d64);
            for (unsigned i = 0; i < 16; ++i) value[i] ^= dest[i];
            if (op == Op::AESE) EncryptSingleRound(out, value); else DecryptSingleRound(out, value);
        } else if (op == Op::AESMC) MixColumns(out, value);
        else InverseMixColumns(out, value);
        result64 = std::bit_cast<V64>(out); wide = true; break;
    }
    case Op::SHA1C: case Op::SHA1P: case Op::SHA1M:
        result = Sha1Hash(d, n, m, op); break;
    case Op::SHA1SU0:
        result = {d[2] ^ d[0] ^ m[0], d[3] ^ d[1] ^ m[1], n[0] ^ d[2] ^ m[2], n[1] ^ d[3] ^ m[3]}; break;
    case Op::SHA1H: result = {std::rotr(n[0], 2), 0, 0, 0}; break;
    case Op::SHA1SU1:
        result = {std::rotl(d[0] ^ n[1], 1), std::rotl(d[1] ^ n[2], 1), std::rotl(d[2] ^ n[3], 1), std::rotl(d[3], 1)};
        result[3] ^= std::rotl(result[0], 1); break;
    case Op::SHA256H: result = Sha256Hash(d, n, m, true); break;
    case Op::SHA256H2: result = Sha256Hash(n, d, m, false); break;
    case Op::SHA256SU0:
        result = {d[0] + Small0(d[1]), d[1] + Small0(d[2]), d[2] + Small0(d[3]), d[3] + Small0(n[0])}; break;
    case Op::SHA256SU1:
        result[0] = d[0] + n[1] + Small1(m[2]);
        result[1] = d[1] + n[2] + Small1(m[3]);
        result[2] = d[2] + n[3] + Small1(result[0]);
        result[3] = d[3] + m[0] + Small1(result[1]); break;
    case Op::SHA512SU0:
        result64 = {d64[0] + Small0(d64[1]), d64[1] + Small0(n64[0])}; wide = true; break;
    case Op::SHA512SU1:
        result64 = {d64[0] + m64[0] + Small1(n64[0]), d64[1] + m64[1] + Small1(n64[1])}; wide = true; break;
    case Op::SHA512H: {
        const u64 t = Choose(m64[1], n64[0], n64[1]) + Sigma1(m64[1]) + d64[1];
        const u64 e = t + m64[0];
        result64 = {Choose(e, m64[1], n64[0]) + Sigma1(e) + d64[0], t}; wide = true; break;
    }
    case Op::SHA512H2: {
        const u64 t = Majority(n64[0], m64[1], m64[0]) + Sigma0(m64[0]) + d64[1];
        result64 = {Majority(t, m64[0], m64[1]) + Sigma0(t) + d64[0], t}; wide = true; break;
    }
    case Op::RAX1:
        result64 = {n64[0] ^ std::rotl(m64[0], 1), n64[1] ^ std::rotl(m64[1], 1)}; wide = true; break;
    case Op::XAR:
        result64 = {std::rotr(n64[0] ^ m64[0], int((instruction >> 10) & 63)), std::rotr(n64[1] ^ m64[1], int((instruction >> 10) & 63))}; wide = true; break;
    case Op::EOR3: case Op::BCAX: {
        const auto a = Read(*state, (instruction >> 10) & 31);
        for (unsigned i = 0; i < 2; ++i)
            result64[i] = n64[i] ^ (op == Op::EOR3 ? m64[i] ^ a[i] : m64[i] & ~a[i]);
        wide = true; break;
    }
    case Op::SM3SS1: {
        const auto a = std::bit_cast<V32>(Read(*state, (instruction >> 10) & 31));
        result[3] = std::rotl(std::rotl(n[3], 12) + m[3] + a[3], 7); break;
    }
    case Op::SM3TT1A: case Op::SM3TT1B: case Op::SM3TT2A: case Op::SM3TT2B: {
        const bool first = op == Op::SM3TT1A || op == Op::SM3TT1B;
        const bool parity = op == Op::SM3TT1A || op == Op::SM3TT2A;
        u32 t = parity ? d[1] ^ d[2] ^ d[3] : first ? Majority(d[3], d[1], d[2]) : Choose(d[3], d[2], d[1]);
        t += d[0] + (first ? n[3] ^ std::rotl(d[3], 12) : n[3]) + m[(instruction >> 12) & 3];
        result = {d[1], std::rotl(d[2], first ? 9 : 19), d[3], first ? t : t ^ std::rotl(t, 9) ^ std::rotl(t, 17)}; break;
    }
    case Op::SM3PARTW1:
        for (unsigned i = 0; i < 4; ++i) {
            const u32 t = d[i] ^ n[i] ^ std::rotl(i == 3 ? result[0] : m[i + 1], 15);
            result[i] = t ^ std::rotl(t, 15) ^ std::rotl(t, 23);
        }
        break;
    case Op::SM3PARTW2: {
        V32 t{};
        for (unsigned i = 0; i < 4; ++i) { t[i] = n[i] ^ std::rotl(m[i], 7); result[i] = d[i] ^ t[i]; }
        const u32 a = std::rotl(t[0], 15);
        result[3] ^= a ^ std::rotl(a, 15) ^ std::rotl(a, 23); break;
    }
    case Op::SM4E: result = Sm4Hash(d, n, false); break;
    case Op::SM4EKEY: result = Sm4Hash(n, m, true); break;
    default: return ExecuteBaselineUndefined(state, instruction);
    }
    Write(*state, rd, wide ? result64 : std::bit_cast<V64>(result));
    return true;
}
}  // namespace Dynarmic::Backend::X64
