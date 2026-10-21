#if A2E_DYNARMIC_SHARDED_MONITOR
// A2E_SHARDED_EXCLUSIVE_MEMORY_V1
template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveReadMemory(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor != nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    if constexpr (bitsize != 128) {
        using T = mcl::unsigned_integer_of_size<bitsize>;

        ctx.reg_alloc.HostCall(inst, {}, args[1]);
        EmitExclusiveAlignmentCheck(ctx, inst, code.ABI_PARAM2, bitsize);

        code.mov(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(1));
        DYNARMIC_RUNTIME_CONFIG(code.ABI_PARAM1);
        if (ordered) {
            code.mfence();
        }
        code.CallLambda(
            [](AxxUserConfig& conf, Axx::VAddr vaddr) -> T {
                return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
                    return (conf.callbacks->*callback)(vaddr);
                });
            });
        code.ZeroExtendFrom(bitsize, code.ABI_RETURN);
    } else {
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
        ctx.reg_alloc.Use(args[1], ABI_PARAM2);
        ctx.reg_alloc.EndOfAllocScope();
        ctx.reg_alloc.HostCall(nullptr);
        EmitExclusiveAlignmentCheck(ctx, inst, code.ABI_PARAM2, bitsize);

        code.mov(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(1));
        DYNARMIC_RUNTIME_CONFIG(code.ABI_PARAM1);
        ctx.reg_alloc.AllocStackSpace(16 + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
        if (ordered) {
            code.mfence();
        }
        code.CallLambda(
            [](AxxUserConfig& conf, Axx::VAddr vaddr, Vector& ret) {
                ret = conf.global_monitor->ReadAndMark<Vector>(conf.processor_id, vaddr, [&]() -> Vector {
                    return (conf.callbacks->*callback)(vaddr);
                });
            });
        code.movups(result, xword[rsp + ABI_SHADOW_SPACE]);
        ctx.reg_alloc.ReleaseStackSpace(16 + ABI_SHADOW_SPACE);

        ctx.reg_alloc.DefineValue(inst, result);
    }

    // A returned memory-abort callback must not leave a usable reservation.
    Xbyak::Label read_ok;
    code.test(dword[r15 + offsetof(AxxJitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
    code.jz(read_ok);
    code.mov(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(0));
    DYNARMIC_RUNTIME_RESERVATION(code.ABI_PARAM1);
    code.mov(qword[code.ABI_PARAM1 + offsetof(ExclusiveMonitor::Reservation, epoch)], 0);
    code.L(read_ok);
    EmitCheckMemoryAbort(ctx, inst);
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveWriteMemory(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor != nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    if constexpr (bitsize != 128) {
        ctx.reg_alloc.HostCall(inst, {}, args[1], args[2]);
    } else {
        ctx.reg_alloc.Use(args[1], ABI_PARAM2);
        ctx.reg_alloc.Use(args[2], HostLoc::XMM1);
        ctx.reg_alloc.EndOfAllocScope();
        ctx.reg_alloc.HostCall(inst);
    }

    EmitExclusiveAlignmentCheck(ctx, inst, code.ABI_PARAM2, bitsize);
    Xbyak::Label end;

    code.mov(code.ABI_RETURN, u32(1));
    code.cmp(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(0));
    code.mov(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(0));
    code.je(end);
    DYNARMIC_RUNTIME_CONFIG(code.ABI_PARAM1);
    if constexpr (bitsize != 128) {
        using T = mcl::unsigned_integer_of_size<bitsize>;

        code.CallLambda(
            [](AxxUserConfig& conf, Axx::VAddr vaddr, T value) -> u32 {
                return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr,
                                                                    [&](T expected) -> bool {
                                                                        return (conf.callbacks->*callback)(vaddr, value, expected);
                                                                    })
                         ? 0
                         : 1;
            });
        if (ordered) {
            code.mfence();
        }
    } else {
        ctx.reg_alloc.AllocStackSpace(16 + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
        code.movaps(xword[code.ABI_PARAM3], xmm1);
        code.CallLambda(
            [](AxxUserConfig& conf, Axx::VAddr vaddr, Vector& value) -> u32 {
                return conf.global_monitor->DoExclusiveOperation<Vector>(conf.processor_id, vaddr,
                                                                         [&](Vector expected) -> bool {
                                                                             return (conf.callbacks->*callback)(vaddr, value, expected);
                                                                         })
                         ? 0
                         : 1;
            });
        if (ordered) {
            code.mfence();
        }
        ctx.reg_alloc.ReleaseStackSpace(16 + ABI_SHADOW_SPACE);
    }
    code.L(end);

    EmitCheckMemoryAbort(ctx, inst);
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveReadMemoryInline(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor && conf.fastmem_pointer);
    if (!exception_handler.SupportsFastmem()) {
        EmitExclusiveReadMemory<bitsize, callback>(ctx, inst);
        return;
    }

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    constexpr bool ordered = true;

    if constexpr (ordered && bitsize == 128) {
        // Required for atomic 128-bit loads/stores
        ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
    }

    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[1]);
    const int value_idx = bitsize == 128 ? ctx.reg_alloc.ScratchXmm().getIdx() : ctx.reg_alloc.ScratchGpr().getIdx();
    const Xbyak::Reg64 tmp = ctx.reg_alloc.ScratchGpr();
    const Xbyak::Reg64 tmp2 = ctx.reg_alloc.ScratchGpr();
    const Xbyak::Reg64 reservation = ctx.reg_alloc.ScratchGpr();

    const auto wrapped_fn = read_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value_idx)];

    EmitExclusiveAlignmentCheck(ctx, inst, vaddr, bitsize);
    EmitExclusiveLock(code, conf, vaddr, tmp, tmp2.cvt32());

    DYNARMIC_RUNTIME_RESERVATION(tmp2);
    code.mov(qword[tmp2 + offsetof(ExclusiveMonitor::Reservation, epoch)], 0);
    code.mov(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(0));

    const auto fastmem_marker = ShouldFastmem(ctx, inst);
    if (fastmem_marker) {
        SharedLabel abort = GenSharedLabel(), end = GenSharedLabel();
        bool require_abort_handling = false;

        const auto src_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

        const auto location = EmitReadMemoryMov<bitsize>(code, value_idx, src_ptr, ordered);

        DYNARMIC_FASTMEM_RECORD(
            mcl::bit_cast<u64>(location),
            FastmemPatchInfo{
                mcl::bit_cast<u64>(code.getCurr()),
                mcl::bit_cast<u64>(wrapped_fn),
                *fastmem_marker,
                conf.recompile_on_exclusive_fastmem_failure,
            });

        code.L(*end);

        if (require_abort_handling) {
            ctx.deferred_emits.emplace_back([=, this] {
                code.L(*abort);
                code.call(wrapped_fn);
                code.jmp(*end, code.T_NEAR);
            });
        }
    } else {
        code.call(wrapped_fn);
    }

    DYNARMIC_RUNTIME_RESERVATION(tmp2);
    EmitWriteMemoryMov<bitsize>(code, tmp2 + offsetof(ExclusiveMonitor::Reservation, value), value_idx, false);
    code.mov(qword[tmp2 + offsetof(ExclusiveMonitor::Reservation, address)], vaddr);
    code.mov(qword[tmp2 + offsetof(ExclusiveMonitor::Reservation, bytes)], bitsize / 8);
    // tmp is the bucket pointer, which the fault stubs preserve. Store the
    // saved epoch only if the read callback did not report a memory abort.
    Xbyak::Label read_fault;
    code.test(dword[r15 + offsetof(AxxJitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
    code.jnz(read_fault);
    // Re-use tmp2 after saving the guest value, without clobbering that value.
    code.mov(tmp2, qword[tmp + offsetof(ExclusiveMonitor::Bucket, epoch)]);
    DYNARMIC_RUNTIME_RESERVATION(reservation);
    code.mov(qword[reservation + offsetof(ExclusiveMonitor::Reservation, epoch)], tmp2);
    code.mov(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(1));
    code.L(read_fault);
    EmitExclusiveUnlock(code, tmp, tmp2.cvt32());

    if constexpr (bitsize == 128) {
        ctx.reg_alloc.DefineValue(inst, Xbyak::Xmm{value_idx});
    } else {
        ctx.reg_alloc.DefineValue(inst, Xbyak::Reg64{value_idx});
    }

    EmitCheckMemoryAbort(ctx, inst);
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveWriteMemoryInline(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor && conf.fastmem_pointer);
    if (!exception_handler.SupportsFastmem()) {
        EmitExclusiveWriteMemory<bitsize, callback>(ctx, inst);
        return;
    }

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    constexpr bool ordered = true;

    const auto value = [&] {
        if constexpr (bitsize == 128) {
            ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
            ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
            ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
            ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
            return ctx.reg_alloc.UseXmm(args[2]);
        } else {
            ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
            return ctx.reg_alloc.UseGpr(args[2]);
        }
    }();
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[1]);
    const Xbyak::Reg32 status = ctx.reg_alloc.ScratchGpr().cvt32();
    const Xbyak::Reg64 tmp = ctx.reg_alloc.ScratchGpr();
    const Xbyak::Reg64 reservation = ctx.reg_alloc.ScratchGpr();

    const auto wrapped_fn = exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value.getIdx())];

    EmitExclusiveAlignmentCheck(ctx, inst, vaddr, bitsize);
    EmitExclusiveLock(code, conf, vaddr, tmp, eax);

    SharedLabel end = GenSharedLabel();
    DYNARMIC_RUNTIME_RESERVATION(reservation);
    code.mov(status, u32(1));
    code.cmp(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(0));
    code.mov(code.byte[r15 + offsetof(AxxJitState, exclusive_state)], u8(0));
    code.je(*end, code.T_NEAR);
    code.mov(rax, qword[reservation + offsetof(ExclusiveMonitor::Reservation, epoch)]);
    code.mov(qword[reservation + offsetof(ExclusiveMonitor::Reservation, epoch)], 0);
    code.cmp(qword[reservation + offsetof(ExclusiveMonitor::Reservation, address)], vaddr);
    code.jne(*end, code.T_NEAR);
    code.cmp(qword[reservation + offsetof(ExclusiveMonitor::Reservation, bytes)], bitsize / 8);
    code.jne(*end, code.T_NEAR);
    code.test(rax, rax);
    code.je(*end, code.T_NEAR);
    code.cmp(rax, qword[tmp + offsetof(ExclusiveMonitor::Bucket, epoch)]);
    code.jne(*end, code.T_NEAR);

    EmitExclusiveBumpEpoch(code, conf, tmp);

    if constexpr (bitsize == 128) {
        code.mov(rax, qword[reservation + offsetof(ExclusiveMonitor::Reservation, value)]);
        code.mov(rdx, qword[reservation + offsetof(ExclusiveMonitor::Reservation, value) + 8]);
        if (code.HasHostFeature(HostFeature::SSE41)) {
            code.movq(rbx, value);
            code.pextrq(rcx, value, 1);
        } else {
            code.movaps(xmm0, value);
            code.movq(rbx, xmm0);
            code.punpckhqdq(xmm0, xmm0);
            code.movq(rcx, xmm0);
        }
    } else {
        EmitReadMemoryMov<bitsize>(code, rax.getIdx(), reservation + offsetof(ExclusiveMonitor::Reservation, value), false);
    }

    const auto fastmem_marker = ShouldFastmem(ctx, inst);
    if (fastmem_marker) {
        SharedLabel abort = GenSharedLabel();
        bool require_abort_handling = false;

        const auto dest_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling, reservation);

        const auto location = code.getCurr();

        if constexpr (bitsize == 128) {
            code.lock();
            code.cmpxchg16b(ptr[dest_ptr]);
        } else {
            switch (bitsize) {
            case 8:
                code.lock();
                code.cmpxchg(code.byte[dest_ptr], value.cvt8());
                break;
            case 16:
                code.lock();
                code.cmpxchg(word[dest_ptr], value.cvt16());
                break;
            case 32:
                code.lock();
                code.cmpxchg(dword[dest_ptr], value.cvt32());
                break;
            case 64:
                code.lock();
                code.cmpxchg(qword[dest_ptr], value.cvt64());
                break;
            default:
                UNREACHABLE();
            }
        }

        code.setnz(status.cvt8());

        ctx.deferred_emits.emplace_back([=, this] {
            code.L(*abort);
            code.call(wrapped_fn);

            DYNARMIC_FASTMEM_RECORD(
                mcl::bit_cast<u64>(location),
                FastmemPatchInfo{
                    mcl::bit_cast<u64>(code.getCurr()),
                    mcl::bit_cast<u64>(wrapped_fn),
                    *fastmem_marker,
                    conf.recompile_on_exclusive_fastmem_failure,
                });

            code.cmp(al, 0);
            code.setz(status.cvt8());
            code.movzx(status.cvt32(), status.cvt8());
            code.jmp(*end, code.T_NEAR);
        });
    } else {
        code.call(wrapped_fn);
        code.cmp(al, 0);
        code.setz(status.cvt8());
        code.movzx(status.cvt32(), status.cvt8());
    }

    code.L(*end);

    EmitExclusiveUnlock(code, tmp, eax);

    ctx.reg_alloc.DefineValue(inst, status);

    EmitCheckMemoryAbort(ctx, inst);
}


#else
#include "dynarmic/backend/x64/emit_exclusive_memory_original.inc"
#endif
