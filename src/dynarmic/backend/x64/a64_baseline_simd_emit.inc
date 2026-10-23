// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT

Emitter::Result Emitter::SimdInstruction(u32 instruction) {
    if (!BaselineSimdSupported(instruction))
        return Result::Unsupported;
    if (BaselineSimdExtraSupported(instruction)) {
        EmitHelper(offsetof(A64JitState, baseline_simd_instruction), instruction);
        return Result::Continue;
    }
    const u32 i = instruction;
    const unsigned d = i & 31, n = (i >> 5) & 31, m = (i >> 16) & 31;
    const bool q = (i >> 30) & 1, u = (i >> 29) & 1, scalar = (i >> 28) & 1;
    const bool simd = !scalar || q;
    const unsigned size = (i >> 22) & 3, bits = 8 << size, op = (i >> 11) & 31;
    const auto constant = [&](const Xbyak::Xmm& reg, u64 value) {
        mov(rax, value);
        movq(reg, rax);
        punpcklqdq(reg, reg);
    };
    const auto repeated = [](u64 value, unsigned width) {
        u64 result = 0;
        for (unsigned at = 0; at < 64; at += width) result |= value << at;
        return result;
    };
    const auto add_lanes = [&](const Xbyak::Xmm& a, const Xbyak::Xmm& b, unsigned width, bool subtract) {
        if (subtract) {
            if (width == 8) psubb(a, b); else if (width == 16) psubw(a, b);
            else if (width == 32) psubd(a, b); else psubq(a, b);
        } else {
            if (width == 8) paddb(a, b); else if (width == 16) paddw(a, b);
            else if (width == 32) paddd(a, b); else paddq(a, b);
        }
    };
    const auto equal_lanes = [&](const Xbyak::Xmm& a, const Xbyak::Xmm& b, unsigned width) {
        if (width == 8) pcmpeqb(a, b); else if (width == 16) pcmpeqw(a, b);
        else {
            pcmpeqd(a, b);
            if (width == 64) { pshufd(xmm5, a, 0xb1); pand(a, xmm5); }
        }
    };
    const auto greater_lanes = [&](const Xbyak::Xmm& a, const Xbyak::Xmm& b, unsigned width) {
        if (width == 8) pcmpgtb(a, b); else if (width == 16) pcmpgtw(a, b); else pcmpgtd(a, b);
    };
    const auto shift_lanes = [&](const Xbyak::Xmm& a, unsigned width, unsigned count, bool left, bool arithmetic = false) {
        if (left) {
            if (width == 16) psllw(a, count); else if (width == 32) pslld(a, count); else psllq(a, count);
        } else if (arithmetic) {
            if (width == 16) psraw(a, count); else psrad(a, count);
        } else {
            if (width == 16) psrlw(a, count); else if (width == 32) psrld(a, count); else psrlq(a, count);
        }
    };
    const auto isolate = [&](const Xbyak::Xmm& a, unsigned bytes) {
        if (bytes == 8) movq(a, a);
        else { pslldq(a, 16 - bytes); psrldq(a, 16 - bytes); }
    };
    const auto fp_immediate = [](unsigned imm, unsigned width) {
        const unsigned exponent_bits = width == 16 ? 5 : width == 32 ? 8 : 11;
        const unsigned fraction_bits = width - exponent_bits - 1;
        const u64 b = (imm >> 6) & 1;
        const u64 exponent = ((!b) << (exponent_bits - 1)) | (b ? ((u64{1} << (exponent_bits - 3)) - 1) << 2 : 0) | ((imm >> 4) & 3);
        return (u64(imm >> 7) << (width - 1)) | (exponent << fraction_bits) | (u64(imm & 15) << (fraction_bits - 4));
    };
    const auto narrow = [&](unsigned width) {
        if (width == 8) {
            constant(xmm1, 0x00ff00ff00ff00ffull); pand(xmm0, xmm1); packuswb(xmm0, xmm0);
        } else if (width == 16) {
            pslld(xmm0, 16); psrad(xmm0, 16); packssdw(xmm0, xmm0);
        } else pshufd(xmm0, xmm0, 0x88);
        if (q) { ReadVector(d, xmm1); punpcklqdq(xmm1, xmm0); movdqa(xmm0, xmm1); }
        WriteVector(d, xmm0, q);
    };

    if ((i & 0xff201fe0) == 0x1e201000) {
        const unsigned width = size == 3 ? 16 : 32 << size;
        mov(rax, fp_immediate((i >> 13) & 255, width)); movq(xmm0, rax);
        WriteVector(d, xmm0);
        return Result::Continue;
    }
    if ((i & 0x9ff80400) == 0x0f000400) {
        const unsigned cmode = (i >> 12) & 15, imm = ((i >> 11) & 0xe0) | ((i >> 5) & 31);
        unsigned width;
        u64 value;
        bool merge = false;
        if (cmode < 8) { width = 32; value = u64(imm) << ((cmode >> 1) * 8); merge = cmode & 1; }
        else if (cmode < 12) { width = 16; value = u64(imm) << (((cmode >> 1) & 1) * 8); merge = cmode & 1; }
        else if (cmode < 14) { width = 32; const unsigned shift = cmode == 12 ? 8 : 16; value = (u64(imm) << shift) | Ones(shift); }
        else if (cmode == 14 && !u) { width = 8; value = imm; }
        else if (cmode == 14) {
            width = 64; value = 0;
            for (unsigned k = 0; k < 8; ++k) if ((imm >> k) & 1) value |= u64{255} << (k * 8);
        } else { width = i & 0x800 ? 16 : u ? 64 : 32; value = fp_immediate(imm, width); }
        value = repeated(value, width);
        if (cmode < 14 && u) value = ~value;
        constant(xmm0, value);
        if (merge) {
            ReadVector(d, xmm1, q);
            if (u) pand(xmm0, xmm1); else por(xmm0, xmm1);
        }
        WriteVector(d, xmm0, q);
        return Result::Continue;
    }

    // FMOV/FABS/FNEG are bit operations and never consult the FP environment.
    if ((i & 0xff207c00) == 0x1e204000 && ((i >> 15) & 63) <= 2) {
        const unsigned width = size == 3 ? 16 : 32 << size;
        const unsigned unary = (i >> 15) & 63;
        ReadVector(n, xmm0);
        if (unary) {
            mov(rax, u64{1} << (width - 1)); movq(xmm1, rax);
            if (unary == 1) { pandn(xmm1, xmm0); movdqa(xmm0, xmm1); }
            else pxor(xmm0, xmm1);
        }
        isolate(xmm0, width / 8);
        WriteVector(d, xmm0);
        return Result::Continue;
    }

    // FCSEL and half-precision general-register FMOV transfer raw bits.
    // They never consult FPCR or FPSR and need no arithmetic helper boundary.
    if ((i & 0xff200c00) == 0x1e200c00) {
        const unsigned width = size == 3 ? 16 : 32 << size;
        ReadVector(n, xmm0); ReadVector(m, xmm1);
        movq(rbx, xmm0); movq(rcx, xmm1);
        const unsigned condition = (i >> 12) & 15;
        LoadFlags(condition); Select(condition, rcx, rbx);
        movq(xmm0, rcx); isolate(xmm0, width / 8);
        WriteVector(d, xmm0);
        return Result::Continue;
    }
    if ((i & 0x7fe0fc00) == 0x1ee00000) {
        const unsigned transfer = (i >> 16) & 31;
        if (transfer == 6 || transfer == 7) {
            const bool wide = i >> 31;
            if (transfer == 7) {
                Read(n, rax, wide); movzx(eax, ax); movq(xmm0, rax);
                WriteVector(d, xmm0);
            } else {
                ReadVector(n, xmm0); movq(rax, xmm0); movzx(eax, ax);
                Write(d, rax, wide);
            }
            return Result::Continue;
        }
    }

    // Ordered S/D comparisons of normal numbers and signed zero have identical
    // ARM/SSE results and cannot raise FP exceptions. Keep exceptional values
    // in the exact helper so NaN signaling, FZ and enabled traps remain precise.
    if ((i & 0xff20fc07) == 0x1e202000 && size <= 1) {
        ReadVector(n, xmm0);
        const bool with_zero = i & 8;
        if (with_zero) pxor(xmm1, xmm1); else ReadVector(m, xmm1);
        Xbyak::Label slow, done;
        const auto ordered = [&](const Xbyak::Xmm& value) {
            Xbyak::Label valid;
            if (size == 0) {
                movd(eax, value); add(eax, eax);
                mov(ecx, eax); and_(ecx, 0xff000000u);
                cmp(ecx, 0xff000000u); je(slow, T_NEAR);
                test(ecx, ecx); jnz(valid, T_NEAR);
                test(eax, eax); jnz(slow, T_NEAR);
            } else {
                movq(rax, value); add(rax, rax);
                mov(rcx, rax); shr(rcx, 53);
                cmp(ecx, 0x7ff); je(slow, T_NEAR);
                test(ecx, ecx); jnz(valid, T_NEAR);
                test(rax, rax); jnz(slow, T_NEAR);
            }
            L(valid);
        };
        ordered(xmm0); if (!with_zero) ordered(xmm1);
        if (size == 0) ucomiss(xmm0, xmm1); else ucomisd(xmm0, xmm1);
        mov(eax, NZCV::ToX64(0x20000000));
        mov(ecx, NZCV::ToX64(0x80000000));
        mov(edx, NZCV::ToX64(0x60000000));
        cmovb(eax, ecx); cmove(eax, edx);
        mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], eax);
        jmp(done, T_NEAR);
        L(slow);
        const auto saved_slots = slots;
        const auto saved_vectors = vector_slots;
        EmitHelper(offsetof(A64JitState, baseline_simd_instruction), instruction);
        slots = saved_slots; vector_slots = saved_vectors;
        ReloadAfterMemoryCallback();
        L(done);
        return Result::Continue;
    }

    // Register copies, broadcasts, insertion, and extraction are independent
    // of FPSR. Every source is read before any aliasing destination is written.
    if (simd && (i & 0x8fe08400) == 0x0e000400) {
        const unsigned imm5 = (i >> 16) & 31, esize = std::countr_zero(imm5);
        const unsigned width = 8 << esize, index = imm5 >> (esize + 1), copy_op = (i >> 11) & 15;
        const bool general = !u && (copy_op == 1 || copy_op == 3);
        if (general) {
            Read(n, rax, width == 64);
            movq(xmm0, rax);
        } else {
            ReadVector(n, xmm0);
            const unsigned source_lane = u ? copy_op >> esize : index;
            if (source_lane) psrldq(xmm0, source_lane * width / 8);
        }
        if (scalar) {
            isolate(xmm0, width / 8);
            WriteVector(d, xmm0);
        } else if (u || copy_op == 3) {
            isolate(xmm0, width / 8);
            if (index) pslldq(xmm0, index * width / 8);
            ReadVector(d, xmm1);
            mov(rax, Ones(width)); movq(xmm2, rax);
            if (index) pslldq(xmm2, index * width / 8);
            pandn(xmm2, xmm1); por(xmm0, xmm2);
            WriteVector(d, xmm0);
        } else if (copy_op == 0 || copy_op == 1) {
            if (width == 8) punpcklbw(xmm0, xmm0);
            if (width <= 16) pshuflw(xmm0, xmm0, 0);
            if (width <= 32) pshufd(xmm0, xmm0, 0);
            else punpcklqdq(xmm0, xmm0);
            WriteVector(d, xmm0, q);
        } else {
            movq(rax, xmm0);
            if (copy_op == 5) {
                if (width == 8) movsx(rax, al); else if (width == 16) movsx(rax, ax); else movsxd(rax, eax);
            } else {
                if (width == 8) movzx(eax, al); else if (width == 16) movzx(eax, ax);
            }
            Write(d, rax, q);
        }
        return Result::Continue;
    }

    if ((i & 0xbfe08400) == 0x2e000000) {
        const unsigned offset = (i >> 11) & 15, length = q ? 16 : 8;
        ReadVector(n, xmm0, q); ReadVector(m, xmm1, q);
        if (offset) psrldq(xmm0, offset);
        pslldq(xmm1, length - offset); por(xmm0, xmm1);
        WriteVector(d, xmm0, q);
        return Result::Continue;
    }

    if ((i & 0xbf208c00) == 0x0e000800) {
        const unsigned permutation = (i >> 12) & 7, part = permutation >> 2;
        if ((permutation & 3) == 3 || (permutation & 3) == 1 || bits == 64) {
            ReadVector(n, xmm0, q); ReadVector(m, xmm1, q);
            if ((permutation & 3) == 3) {
                if (part && !q) { psrldq(xmm0, 4); psrldq(xmm1, 4); }
                if (part && q) {
                    if (bits == 8) punpckhbw(xmm0, xmm1); else if (bits == 16) punpckhwd(xmm0, xmm1);
                    else if (bits == 32) punpckhdq(xmm0, xmm1); else punpckhqdq(xmm0, xmm1);
                } else {
                    if (bits == 8) punpcklbw(xmm0, xmm1); else if (bits == 16) punpcklwd(xmm0, xmm1);
                    else if (bits == 32) punpckldq(xmm0, xmm1); else punpcklqdq(xmm0, xmm1);
                }
            } else if (bits == 64) shufpd(xmm0, xmm1, part ? 3 : 0);
            else if (bits == 32) {
                pshufd(xmm0, xmm0, part ? 0xdd : 0x88); pshufd(xmm1, xmm1, part ? 0xdd : 0x88);
                if (q) punpcklqdq(xmm0, xmm1); else punpckldq(xmm0, xmm1);
            } else if (bits == 16) {
                pshuflw(xmm0, xmm0, part ? 0xdd : 0x88); pshufhw(xmm0, xmm0, part ? 0xdd : 0x88);
                pshuflw(xmm1, xmm1, part ? 0xdd : 0x88); pshufhw(xmm1, xmm1, part ? 0xdd : 0x88);
                pshufd(xmm0, xmm0, 0x88); pshufd(xmm1, xmm1, 0x88);
                if (q) punpcklqdq(xmm0, xmm1); else punpckldq(xmm0, xmm1);
            } else {
                if (part) { psrlw(xmm0, 8); psrlw(xmm1, 8); }
                else { constant(xmm2, 0x00ff00ff00ff00ffull); pand(xmm0, xmm2); pand(xmm1, xmm2); }
                packuswb(xmm0, xmm1);
                if (!q) { movdqa(xmm1, xmm0); psrldq(xmm1, 8); punpckldq(xmm0, xmm1); }
            }
            WriteVector(d, xmm0, q);
            return Result::Continue;
        }
    }

    // Common integer arithmetic uses SSE2 on every x86-64 host. Saturating
    // instructions compare against modular arithmetic to set sticky FPSR.QC.
    if (simd && (i & 0x8f200400) == 0x0e200400 && op < 24 && op != 3) {
        const bool compare = (op == 6 || op == 7 || op == 12 || op == 13) && bits <= 32;
        const bool multiply = (op == 18 || (op == 19 && !u)) && (bits == 16 || bits == 32);
        const bool saturate = (op == 1 || op == 5) && bits <= 16;
        if (op == 16 || op == 17 || compare || multiply || saturate || (op == 2 && u && bits <= 16)) {
            ReadVector(n, xmm0, q); ReadVector(m, xmm1, q);
            if (op == 16) {
                add_lanes(xmm0, xmm1, bits, u);
            } else if (op == 17) {
                if (!u) {
                    pand(xmm0, xmm1); pxor(xmm1, xmm1);
                    equal_lanes(xmm0, xmm1, bits);
                    pcmpeqd(xmm1, xmm1); pxor(xmm0, xmm1);
                } else equal_lanes(xmm0, xmm1, bits);
            } else if (compare) {
                movdqa(xmm2, xmm0); movdqa(xmm3, xmm1);
                if (u) {
                    constant(xmm4, repeated(u64{1} << (bits - 1), bits));
                    pxor(xmm2, xmm4); pxor(xmm3, xmm4);
                }
                if (op == 7) {
                    greater_lanes(xmm3, xmm2, bits);
                    pcmpeqd(xmm2, xmm2); pxor(xmm3, xmm2); movdqa(xmm0, xmm3);
                } else {
                    greater_lanes(xmm2, xmm3, bits);
                    if (op == 6) movdqa(xmm0, xmm2);
                    else {
                        if (op == 13) { pcmpeqd(xmm3, xmm3); pxor(xmm2, xmm3); }
                        pand(xmm0, xmm2); pandn(xmm2, xmm1); por(xmm0, xmm2);
                    }
                }
            } else if (multiply) {
                if (bits == 16) pmullw(xmm0, xmm1);
                else {
                    movdqa(xmm2, xmm0); movdqa(xmm3, xmm1);
                    psrlq(xmm2, 32); psrlq(xmm3, 32);
                    pmuludq(xmm0, xmm1); pmuludq(xmm2, xmm3);
                    pshufd(xmm0, xmm0, 0x88); pshufd(xmm2, xmm2, 0x88);
                    punpckldq(xmm0, xmm2);
                }
                if (op == 18) {
                    ReadVector(d, xmm1, q);
                    if (u) { add_lanes(xmm1, xmm0, bits, true); movdqa(xmm0, xmm1); }
                    else add_lanes(xmm0, xmm1, bits, false);
                }
            } else if (saturate) {
                movdqa(xmm2, xmm0); add_lanes(xmm2, xmm1, bits, op == 5);
                if (op == 1) {
                    if (bits == 8) { if (u) paddusb(xmm0, xmm1); else paddsb(xmm0, xmm1); }
                    else { if (u) paddusw(xmm0, xmm1); else paddsw(xmm0, xmm1); }
                } else {
                    if (bits == 8) { if (u) psubusb(xmm0, xmm1); else psubsb(xmm0, xmm1); }
                    else { if (u) psubusw(xmm0, xmm1); else psubsw(xmm0, xmm1); }
                }
                pcmpeqb(xmm2, xmm0); pmovmskb(eax, xmm2);
                const unsigned bytes = scalar ? bits / 8 : q ? 16 : 8;
                const unsigned valid = (1u << bytes) - 1;
                and_(eax, valid); cmp(eax, valid);
                Xbyak::Label not_saturated;
                je(not_saturated, T_NEAR);
                mov(dword[r15 + offsetof(A64JitState, fpsr_qc)], 1);
                L(not_saturated);
            } else {
                if (bits == 8) pavgb(xmm0, xmm1); else pavgw(xmm0, xmm1);
            }
            if (scalar) isolate(xmm0, bits / 8);
            WriteVector(d, xmm0, scalar || q);
            return Result::Continue;
        }
    }

    if (simd && (i & 0x8f3e0c00) == 0x0e200800) {
        const unsigned misc = (i >> 12) & 31;
        if (misc == 18 && !u) {
            ReadVector(n, xmm0); narrow(bits);
            return Result::Continue;
        }
        if (misc == 15 && !scalar) {
            const unsigned width = 32 << (size & 1);
            ReadVector(n, xmm0, q); constant(xmm1, repeated(u64{1} << (width - 1), width));
            if (u) pxor(xmm0, xmm1); else { pandn(xmm1, xmm0); movdqa(xmm0, xmm1); }
            WriteVector(d, xmm0, q);
            return Result::Continue;
        }
        if ((misc == 11 && (u || bits <= 32)) || (misc == 5 && u && size == 0)) {
            ReadVector(n, xmm0, q);
            if (misc == 5) { pcmpeqd(xmm1, xmm1); pxor(xmm0, xmm1); }
            else if (u) { pxor(xmm1, xmm1); add_lanes(xmm1, xmm0, bits, true); movdqa(xmm0, xmm1); }
            else {
                pxor(xmm1, xmm1); greater_lanes(xmm1, xmm0, bits);
                pxor(xmm0, xmm1); add_lanes(xmm0, xmm1, bits, true);
            }
            if (scalar) isolate(xmm0, bits / 8);
            WriteVector(d, xmm0, scalar || q);
            return Result::Continue;
        }
    }

    const bool shift_group = simd && (i & 0x8f800400) == 0x0f000400 && ((i >> 19) & 15);
    if (shift_group) {
        const unsigned imm = (i >> 16) & 127;
        const unsigned width = 8u << (31 - std::countl_zero(imm >> 3));
        const unsigned right = width * 2 - imm, left = imm - width;
        const bool right_op = op == 0 || op == 2 || op == 4 || op == 6 || op == 8;
        if (width >= 16 && (op == 10 || (right_op && (u || width != 64)))) {
            ReadVector(n, xmm0, q);
            if (op == 10) {
                shift_lanes(xmm0, width, left, true);
                if (u) {
                    ReadVector(d, xmm1, q); constant(xmm2, repeated(Ones(left), width));
                    pand(xmm1, xmm2); por(xmm0, xmm1);
                }
            } else {
                if (op & 4) {
                    movdqa(xmm1, xmm0);
                    shift_lanes(xmm1, width, right - 1, false);
                    constant(xmm2, repeated(1, width)); pand(xmm1, xmm2);
                }
                shift_lanes(xmm0, width, right, false, !u);
                if (op & 4) add_lanes(xmm0, xmm1, width, false);
                if (op & 2) { ReadVector(d, xmm1, q); add_lanes(xmm0, xmm1, width, false); }
                if (op == 8) {
                    ReadVector(d, xmm1, q);
                    constant(xmm2, repeated(Ones(width) ^ Ones(width - right), width));
                    pand(xmm1, xmm2); por(xmm0, xmm1);
                }
            }
            if (scalar) isolate(xmm0, width / 8);
            WriteVector(d, xmm0, scalar || q);
            return Result::Continue;
        }
        if (op == 20) {  // SSHLL/USHLL, including SXTL/UXTL aliases.
            ReadVector(n, xmm0);
            if (q) psrldq(xmm0, 8);
            pxor(xmm1, xmm1);
            if (!u) greater_lanes(xmm1, xmm0, width);
            if (width == 8) punpcklbw(xmm0, xmm1);
            else if (width == 16) punpcklwd(xmm0, xmm1);
            else punpckldq(xmm0, xmm1);
            if (left) shift_lanes(xmm0, width * 2, left, true);
            WriteVector(d, xmm0);
            return Result::Continue;
        }
        if (!u && !scalar && (op == 16 || op == 17)) {
            ReadVector(n, xmm0);
            if (op == 17) {
                constant(xmm1, repeated(u64{1} << (right - 1), width * 2));
                add_lanes(xmm0, xmm1, width * 2, false);
            }
            shift_lanes(xmm0, width * 2, right, false);
            narrow(width);
            return Result::Continue;
        }
    }

    // Common S/D floating-point operations have a straight SSE2 normal-value
    // path. Exceptional inputs/results go to our exact arithmetic veneer.
    // Restore the pre-operation MXCSR before that path so speculative x86
    // underflow/NaN flags never leak into the architectural ARM status.
    const bool fp_scalar_binary = (i & 0xff200c00) == 0x1e200800 && size <= 1 && ((i >> 12) & 15) <= 3;
    const bool fp_scalar_sqrt = (i & 0xff207c00) == 0x1e204000 && size <= 1 && ((i >> 15) & 63) == 3;
    const bool fp_vector = (i & 0x9f200400) == 0x0e200400 &&
                           ((op == 26 && !u) || (op == 27 && u && !(size & 2)) || (op == 31 && u && !(size & 2)));
    if (fp_scalar_binary || fp_scalar_sqrt || fp_vector) {
        const unsigned width = 32 << (size & 1), lanes = fp_vector ? (q ? 128 : 64) / width : 1;
        const unsigned operation = fp_scalar_sqrt ? 4 : fp_scalar_binary ? (i >> 12) & 15 : op == 26 ? ((size & 2) ? 3 : 2) : op == 27 ? 0 : 1;
        const bool scalar_add_sub = fp_scalar_binary && operation >= 2;
        ReadVector(n, xmm0); if (operation != 4) ReadVector(m, xmm1);
        Xbyak::Label slow, restore, done, left_zero, right_zero, left_ready, right_ready;
        test(dword[r15 + offsetof(A64JitState, fpcr)], 0x9f00);
        jnz(slow, T_NEAR);
        constant(xmm5, width == 32 ? 0x7f8000007f800000ull : 0x7ff0000000000000ull);
        pxor(xmm4, xmm4);
        const auto normal = [&](const Xbyak::Xmm& value, Xbyak::Label& exceptional) {
            movdqa(xmm2, value);
            pand(xmm2, xmm5);
            movdqa(xmm3, xmm2);
            pcmpeqd(xmm2, xmm4);
            pcmpeqd(xmm3, xmm5);
            por(xmm2, xmm3);
            pmovmskb(eax, xmm2);
            // A double exponent occupies the upper dword of each lane. The
            // zero low dword must not participate in this allocation check.
            const unsigned active = width == 32 ? (1u << (lanes * 4)) - 1 : lanes == 2 ? 0xf0f0 : 0x00f0;
            test(eax, active);
            jnz(exceptional, T_NEAR);
        };
        normal(xmm0, scalar_add_sub || fp_scalar_sqrt ? left_zero : slow);
        L(left_ready);
        if (operation != 4) normal(xmm1, scalar_add_sub ? right_zero : slow);
        L(right_ready);
        if (fp_vector && !q) {
            // Packed SSE processes all 128 bits. Fill unused high lanes with
            // 1.0 so they cannot add exceptions (particularly for division).
            constant(xmm3, 0x3f8000003f800000ull);
            movlhps(xmm0, xmm3); movlhps(xmm1, xmm3);
        }
        stmxcsr(dword[r15 + offsetof(A64JitState, guest_MXCSR)]);
        if (fp_vector) {
            if (width == 32) {
                if (operation == 0) mulps(xmm0, xmm1); else if (operation == 1) divps(xmm0, xmm1);
                else if (operation == 2) addps(xmm0, xmm1); else subps(xmm0, xmm1);
            } else {
                if (operation == 0) mulpd(xmm0, xmm1); else if (operation == 1) divpd(xmm0, xmm1);
                else if (operation == 2) addpd(xmm0, xmm1); else subpd(xmm0, xmm1);
            }
        } else if (width == 32) {
            if (operation == 0) mulss(xmm0, xmm1); else if (operation == 1) divss(xmm0, xmm1);
            else if (operation == 2) addss(xmm0, xmm1); else if (operation == 3) subss(xmm0, xmm1); else sqrtss(xmm0, xmm0);
        } else {
            if (operation == 0) mulsd(xmm0, xmm1); else if (operation == 1) divsd(xmm0, xmm1);
            else if (operation == 2) addsd(xmm0, xmm1); else if (operation == 3) subsd(xmm0, xmm1); else sqrtsd(xmm0, xmm0);
        }
        // Also reject exponent==1: rounding a tiny exact value up to the
        // minimum normal differs in ARM/x86 underflow detection. Ignoring the
        // exponent's low bit conservatively sends both boundary bins to the
        // exact veneer, without adding another vector comparison.
        constant(xmm5, width == 32 ? 0x7f0000007f000000ull : 0x7fe0000000000000ull);
        normal(xmm0, restore);
        if (!fp_vector) isolate(xmm0, width / 8);
        else if (!q) movq(xmm0, xmm0);
        jmp(done, T_NEAR);
        // Signed zeros use a cold check without extending the normal-input
        // path. Subnormal and non-finite values still need the exact veneer.
        const auto exact_zero = [&](const Xbyak::Xmm& value) {
            if (width == 32) { movd(eax, value); shl(eax, 1); }
            else { movq(rax, value); shl(rax, 1); }
            jnz(slow, T_NEAR);
        };
        if (scalar_add_sub || fp_scalar_sqrt) {
            L(left_zero);
            exact_zero(xmm0);
            if (fp_scalar_sqrt) {
                isolate(xmm0, width / 8);
                jmp(done, T_NEAR);
            } else {
                jmp(left_ready, T_NEAR);
                L(right_zero);
                exact_zero(xmm1);
                jmp(right_ready, T_NEAR);
            }
        }
        L(restore); ldmxcsr(dword[r15 + offsetof(A64JitState, guest_MXCSR)]);
        L(slow);
        // Preserve the compile-time cache map across a conditional helper.
        // Only the cold branch spills/reloads it; the native branch retains
        // live integer/vector values and joins at one architectural write.
        const auto saved_slots = slots;
        const auto saved_vectors = vector_slots;
        EmitHelper(offsetof(A64JitState, baseline_simd_instruction), instruction);
        slots = saved_slots;
        vector_slots = saved_vectors;
        ReloadAfterMemoryCallback();
        ReadVector(d, xmm0);
        L(done);
        WriteVector(d, xmm0, !fp_vector || q);
        return Result::Continue;
    }
    EmitHelper(offsetof(A64JitState, baseline_simd_instruction), instruction);
    return Result::Continue;
}
