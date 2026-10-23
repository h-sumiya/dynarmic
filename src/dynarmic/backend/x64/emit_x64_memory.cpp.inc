/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/macro/concatenate_tokens.hpp>

#define AxxEmitX64 CONCATENATE_TOKENS(Axx, EmitX64)
#define AxxEmitContext CONCATENATE_TOKENS(Axx, EmitContext)
#define AxxJitState CONCATENATE_TOKENS(Axx, JitState)
#define AxxUserConfig Axx::UserConfig

#ifdef DYNARMIC_A64_SHARED_EMITTER
#    define DYNARMIC_MEMORY_CALLBACK(member) GetUserCallback<member>()
#    define DYNARMIC_FASTMEM_RECORD(...) RegisterFastmemPatch(__VA_ARGS__)
#    define DYNARMIC_RUNTIME_CONFIG(target) EmitRuntimeConfigPointer(target)
#    define DYNARMIC_RUNTIME_RESERVATION(target) EmitRuntimeReservationPointer(target)
#else
#    define DYNARMIC_MEMORY_CALLBACK(member) Devirtualize<member>(conf.callbacks)
#    define DYNARMIC_FASTMEM_RECORD(...) fastmem_patch_info.emplace(__VA_ARGS__)
#    define DYNARMIC_RUNTIME_CONFIG(target) code.mov(target, reinterpret_cast<u64>(&conf))
#    define DYNARMIC_RUNTIME_RESERVATION(target) code.mov(target, mcl::bit_cast<u64>(GetExclusiveMonitorReservation(conf.global_monitor, conf.processor_id)))
#endif

namespace {
using Vector = std::array<u64, 2>;
}

std::optional<AxxEmitX64::DoNotFastmemMarker> AxxEmitX64::ShouldFastmem(AxxEmitContext& ctx, IR::Inst* inst) const {
    if (!conf.fastmem_pointer || !exception_handler.SupportsFastmem()) {
        return std::nullopt;
    }

    const auto marker = std::make_tuple(ctx.Location(), inst->GetName());
    if (do_not_fastmem.count(marker) > 0) {
        return std::nullopt;
    }
    return marker;
}

FakeCall AxxEmitX64::FastmemCallback(u64 rip_) {
#ifdef DYNARMIC_A64_SHARED_EMITTER
    if (shared_optimized) {
        return SharedFastmemCallback(rip_);
    }
#endif
    const auto iter = fastmem_patch_info.find(rip_);

    if (iter == fastmem_patch_info.end()) {
        fmt::print("dynarmic: Segfault happened within JITted code at rip = {:016x}\n", rip_);
        fmt::print("Segfault wasn't at a fastmem patch location!\n");
        fmt::print("Now dumping code.......\n\n");
        Common::DumpDisassembledX64((void*)(rip_ & ~u64(0xFFF)), 0x1000);
        ASSERT_FALSE("iter != fastmem_patch_info.end()");
    }

    FakeCall result{
        .call_rip = iter->second.callback,
        .ret_rip = iter->second.resume_rip,
    };

    if (iter->second.recompile) {
        const auto marker = iter->second.marker;
        do_not_fastmem.insert(marker);
        InvalidateBasicBlocks({std::get<0>(marker)});
    }

    return result;
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitMemoryRead(AxxEmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!conf.page_table && !fastmem_marker) {
        // Neither fastmem nor page table: Use callbacks
        if constexpr (bitsize == 128) {
            ctx.reg_alloc.HostCall(nullptr, {}, args[1]);
            if (ordered) {
                code.mfence();
            }
            code.CallFunction(memory_read_128);
            ctx.reg_alloc.DefineValue(inst, xmm1);
        } else {
            ctx.reg_alloc.HostCall(inst, {}, args[1]);
            if (ordered) {
                code.mfence();
            }
            DYNARMIC_MEMORY_CALLBACK(callback).EmitCall(code);
            code.ZeroExtendFrom(bitsize, code.ABI_RETURN);
        }
        EmitCheckMemoryAbort(ctx, inst);
        return;
    }

    if (ordered && bitsize == 128) {
        // Required for atomic 128-bit loads/stores
        ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
    }

    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[1]);
    const int value_idx = bitsize == 128 ? ctx.reg_alloc.ScratchXmm().getIdx() : ctx.reg_alloc.ScratchGpr().getIdx();

    const auto wrapped_fn = read_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value_idx)];

    SharedLabel abort = GenSharedLabel(), end = GenSharedLabel();

    if (fastmem_marker) {
        // Use fastmem
        bool require_abort_handling;
        const auto src_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

        const auto location = EmitReadMemoryMov<bitsize>(code, value_idx, src_ptr, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);

            DYNARMIC_FASTMEM_RECORD(
                mcl::bit_cast<u64>(location),
                FastmemPatchInfo{
                    mcl::bit_cast<u64>(code.getCurr()),
                    mcl::bit_cast<u64>(wrapped_fn),
                    *fastmem_marker,
                    conf.recompile_on_fastmem_failure,
                });

            EmitCheckMemoryAbort(ctx, inst, end.get());
            code.jmp(*end, code.T_NEAR);
        });
    } else {
        // Use page table
        ASSERT(conf.page_table);
        const auto src_ptr = EmitVAddrLookup(code, ctx, bitsize, *abort, vaddr);
        EmitReadMemoryMov<bitsize>(code, value_idx, src_ptr, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);
            EmitCheckMemoryAbort(ctx, inst, end.get());
            code.jmp(*end, code.T_NEAR);
        });
    }
    code.L(*end);

    if constexpr (bitsize == 128) {
        ctx.reg_alloc.DefineValue(inst, Xbyak::Xmm{value_idx});
    } else {
        ctx.reg_alloc.DefineValue(inst, Xbyak::Reg64{value_idx});
    }
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitMemoryWrite(AxxEmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!conf.page_table && !fastmem_marker) {
        // Neither fastmem nor page table: Use callbacks
        if constexpr (bitsize == 128) {
            ctx.reg_alloc.Use(args[1], ABI_PARAM2);
            ctx.reg_alloc.Use(args[2], HostLoc::XMM1);
            ctx.reg_alloc.EndOfAllocScope();
            ctx.reg_alloc.HostCall(nullptr);
            code.CallFunction(memory_write_128);
        } else {
            ctx.reg_alloc.HostCall(nullptr, {}, args[1], args[2]);
            DYNARMIC_MEMORY_CALLBACK(callback).EmitCall(code);
        }
        if (ordered) {
            code.mfence();
        }
        EmitCheckMemoryAbort(ctx, inst);
        return;
    }

    if (ordered && bitsize == 128) {
        // Required for atomic 128-bit loads/stores
        ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
    }

    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(args[1]);
    const int value_idx = bitsize == 128
                            ? ctx.reg_alloc.UseXmm(args[2]).getIdx()
                            : (ordered ? ctx.reg_alloc.UseScratchGpr(args[2]).getIdx() : ctx.reg_alloc.UseGpr(args[2]).getIdx());

    const auto wrapped_fn = write_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value_idx)];

    SharedLabel abort = GenSharedLabel(), end = GenSharedLabel();

    if (fastmem_marker) {
        // Use fastmem
        bool require_abort_handling;
        const auto dest_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

        const auto location = EmitWriteMemoryMov<bitsize>(code, dest_ptr, value_idx, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);

            DYNARMIC_FASTMEM_RECORD(
                mcl::bit_cast<u64>(location),
                FastmemPatchInfo{
                    mcl::bit_cast<u64>(code.getCurr()),
                    mcl::bit_cast<u64>(wrapped_fn),
                    *fastmem_marker,
                    conf.recompile_on_fastmem_failure,
                });

            EmitCheckMemoryAbort(ctx, inst, end.get());
            code.jmp(*end, code.T_NEAR);
        });
    } else {
        // Use page table
        ASSERT(conf.page_table);
        const auto dest_ptr = EmitVAddrLookup(code, ctx, bitsize, *abort, vaddr);
        EmitWriteMemoryMov<bitsize>(code, dest_ptr, value_idx, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);
            EmitCheckMemoryAbort(ctx, inst, end.get());
            code.jmp(*end, code.T_NEAR);
        });
    }
    code.L(*end);
}

#include "dynarmic/backend/x64/emit_exclusive_memory_sharded.inc"
#undef AxxEmitX64
#undef AxxEmitContext
#undef AxxJitState
#undef AxxUserConfig

#undef DYNARMIC_MEMORY_CALLBACK
#undef DYNARMIC_FASTMEM_RECORD
#undef DYNARMIC_RUNTIME_CONFIG
#undef DYNARMIC_RUNTIME_RESERVATION
