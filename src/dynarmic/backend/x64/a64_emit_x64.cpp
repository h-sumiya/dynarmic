/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/a64_emit_x64.h"

#include <algorithm>
#include <utility>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <mcl/assert.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>
#include <mcl/type_traits/integer_of_size.hpp>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/a64_types.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

// TODO: Have ARM flags in host flags and not have them use up GPR registers unless necessary.
// TODO: Actually implement that proper instruction selector you've always wanted to sweetheart.

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

A64EmitContext::A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block)
        : EmitContext(reg_alloc, block), conf(conf) {}

A64::LocationDescriptor A64EmitContext::Location() const {
    return A64::LocationDescriptor{block.Location()};
}

bool A64EmitContext::IsSingleStep() const {
    return Location().SingleStepping();
}

FP::FPCR A64EmitContext::FPCR(bool fpcr_controlled) const {
    return fpcr_controlled ? Location().FPCR() : Location().FPCR().ASIMDStandardValue();
}

A64EmitX64::A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface,
                     bool shared_optimized)
        : EmitX64(code), shared_optimized(shared_optimized), conf(conf), jit_interface{jit_interface}
        , fast_dispatch_table(shared_optimized ? nullptr : std::make_unique<FastDispatchEntry[]>(fast_dispatch_table_size))
        , shared_fastmem_buckets(shared_optimized ? std::make_unique<std::atomic<const SharedFastmemRecord*>[]>(shared_fastmem_bucket_count) : nullptr) {
    GenMemory128Accessors();
    GenFastmemFallbacks();
    if (!shared_optimized) {
        GenTerminalHandlers();
        GenBaselineReturn();
    }
    // Windows stores its exception callback selector in the generated unwind
    // data. Install it before a concurrent region seals the prelude read-only.
    exception_handler.SetFastmemCallback([this](u64 rip_) {
        return FastmemCallback(rip_);
    });
    code.PreludeComplete();
    ClearFastDispatchTable();
}

A64EmitX64::~A64EmitX64() = default;

void A64EmitX64::EmitSharedDispatch() {
    EmitSharedLoopFlush();
    EmitSharedPCFlush();
    // The destination is the executing thread's permanent prelude. No return
    // address in a reclaimable shared region is left on the native stack.
    code.jmp(code.qword[code.r15 + offsetof(A64JitState, shared_dispatch)]);
}

void A64EmitX64::EmitSharedPCValue(Xbyak::Reg64 target) {
    if (shared_pc_in_r11)
        code.mov(target, r11);
    else
        code.mov(target, qword[r15 + offsetof(A64JitState, pc)]);
}

void A64EmitX64::EmitSharedPCFlush() {
    if (shared_pc_in_r11)
        code.mov(qword[r15 + offsetof(A64JitState, pc)], r11);
}

void A64EmitX64::EmitSharedGeneration(Xbyak::Reg64 target) {
    if (shared_generation) {
        // The cache owns both this counter and every executable region. x64's
        // accumulator moffs form loads it directly, without a dependent load
        // of the same immutable pointer from every executing Jit's state.
        code.mov(rax, qword[reinterpret_cast<u64>(shared_generation)]);
        if (target.getIdx() != rax.getIdx())
            code.mov(target, rax);
    } else {
        code.mov(target, qword[r15 + offsetof(A64JitState, shared_generation)]);
        code.mov(target, qword[target]);
    }
}

void A64EmitX64::PrepareSharedDispatch(IR::Block& block) {
    shared_static_dispatch.reset();
    shared_static_pc_inst = nullptr;
    shared_dynamic_pc_inst = nullptr;
    shared_pc_in_r11 = false;
    shared_static_fpcr = shared_optimized;
    const auto terminal = block.GetTerminal();
    const bool constant_terminal = boost::get<IR::Term::FastDispatchHint>(&terminal) != nullptr;
    if (!shared_optimized)
        return;
    if (!conf.very_verbose_debugging_output && !block.empty()
        && block.back().GetOpcode() == IR::Opcode::A64SetPC
        && (constant_terminal || boost::get<IR::Term::PopRSBHint>(&terminal))) {
        shared_dynamic_pc_inst = &block.back();
    }

    // Constant propagation can resolve ADR/BR and constant function pointers.
    // Keep that information through the terminal instead of rebuilding the
    // descriptor and hashing it again at run time. Callbacks and FPCR writes
    // require the general dispatcher because they can change the descriptor.
    for (const auto& inst : block) {
        const auto op = inst.GetOpcode();
        if (inst.WritesToFPCR() || inst.IsMemoryRead() || inst.IsMemoryWrite()
            || inst.CausesCPUException() || op == IR::Opcode::CallHostFunction
            || op == IR::Opcode::A64DataCacheOperationRaised
            || op == IR::Opcode::A64InstructionCacheOperationRaised
            || op == IR::Opcode::A64GetCNTPCT
            || (op == IR::Opcode::A64InstructionSynchronizationBarrier && conf.hook_isb)) {
            shared_static_dispatch.reset();
            shared_static_pc_inst = nullptr;
            shared_static_fpcr = false;
            return;
        }
        if (constant_terminal && op == IR::Opcode::A64SetPC) {
            shared_static_dispatch.reset();
            shared_static_pc_inst = nullptr;
            if (inst.GetArg(0).IsImmediate()) {
                shared_static_dispatch = A64::LocationDescriptor{block.Location()}
                    .SetSingleStepping(false).SetPC(inst.GetArg(0).GetU64());
                shared_static_pc_inst = &inst;
            }
        }
    }
}

void A64EmitX64::EmitSharedLinkFallback(IR::LocationDescriptor target) {
    code.mov(rax, A64::LocationDescriptor{target}.PC());
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    // A return-link cell can enter this veneer from a different block. Its
    // caller has already materialized guest registers: do not emit a
    // source-block-specific register flush here.
    code.jmp(qword[r15 + offsetof(A64JitState, shared_dispatch)]);
}

void A64EmitX64::EmitSharedLink(IR::LocationDescriptor next,
                               IR::LocationDescriptor initial_location, bool is_single_step) {
    const bool self_edge = next == initial_location;
    std::atomic<const void*>* cell = nullptr;
    if (!self_edge && !is_single_step && shared_link_resolver
        && conf.HasOptimization(OptimizationFlag::BlockLinking)) {
        cell = std::construct_at(static_cast<std::atomic<const void*>*>(
            code.AllocateMutableData(sizeof(std::atomic<const void*>))), nullptr);
    }

    if (!is_single_step && (self_edge || cell)) {
        // Mutable link data belongs to the source region. A published target
        // replaces only this atomic cell, never executable instruction bytes.
        // All edges still test the reclamation generation and halt state.
        Xbyak::Label fallback;
        code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
        code.jne(fallback, code.T_NEAR);
        if (conf.enable_cycle_counting) {
            code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
            code.jle(fallback, code.T_NEAR);
        }
        EmitSharedGeneration(rax);
        code.cmp(rax, qword[r15 + offsetof(A64JitState, shared_observed_generation)]);
        code.jne(fallback, code.T_NEAR);
        if (self_edge) {
            ASSERT(shared_block_entrypoint);
            code.jmp(shared_block_entrypoint);
        } else {
            EmitSharedLoopFlush();
            code.jmp(qword[rip + cell]);
        }
        code.L(fallback);
        if (cell) {
            const void* const fallback_entrypoint = code.getCurr();
            cell->store(fallback_entrypoint, std::memory_order_relaxed);
            shared_link_resolver(next.Value(), cell, fallback_entrypoint);
        }
    }
    code.mov(rax, A64::LocationDescriptor{next}.PC());
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    EmitSharedDispatch();
}

void A64EmitX64::EmitSharedDispatchCacheIndex(Xbyak::Reg64 index,
                                            Xbyak::Reg64 location, Xbyak::Reg64 scratch) {
    if (code.HasHostFeature(HostFeature::SSE42)) {
        code.xor_(index.cvt32(), index.cvt32());
        code.crc32(index, location);
    } else {
        code.mov(index, location);
        code.shr(index, 2);
        code.mov(scratch, location);
        code.shr(scratch, 12);
        code.xor_(index, scratch);
        code.shl(index, 5);
    }
    code.and_(index.cvt32(), u32(A64JitState::SharedDispatchCacheByteMask));
    code.lea(index, ptr[r15 + offsetof(A64JitState, shared_dispatch_cache) + index]);
}

void A64EmitX64::EmitSharedDispatchCacheFill(Xbyak::Reg64 entrypoint,
                                           Xbyak::Reg64 location) {
    EmitSharedDispatchCacheIndex(r9, location, r10);
    code.mov(r10, qword[r15 + offsetof(A64JitState, shared_observed_generation)]);
    code.mov(qword[r9 + offsetof(A64JitState::SharedDispatchCacheEntry, location)], location);
    code.mov(qword[r9 + offsetof(A64JitState::SharedDispatchCacheEntry, entrypoint)], entrypoint);
    code.mov(qword[r9 + offsetof(A64JitState::SharedDispatchCacheEntry, generation)], r10);
}

void A64EmitX64::EmitSharedIndirect(IR::LocationDescriptor initial_location, bool is_single_step) {
    EmitSharedLoopFlush();
    if (is_single_step || !conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        EmitSharedDispatch();
        return;
    }
    // Each branch site chooses two fixed private slots at compile time. The
    // slots are only hints: descriptor and generation remain the authority.
    // It shares the bounded private table with the general dispatcher, so a
    // collision needs neither allocation nor shared executable-code patching.
    const u64 source = initial_location.Value();
    const auto slot = (((source >> 2) ^ (source >> 12) ^ (source >> 32))
        & (A64JitState::SharedDispatchCacheSize / 2 - 1)) * 2;
    const auto offset = offsetof(A64JitState, shared_dispatch_cache)
        + slot * sizeof(A64JitState::SharedDispatchCacheEntry);
    Xbyak::Label dispatch;
    code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(dispatch, code.T_NEAR);
    if (conf.enable_cycle_counting) {
        code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
        code.jle(dispatch, code.T_NEAR);
    }
    EmitSharedGeneration(rdx);
    code.cmp(rdx, qword[r15 + offsetof(A64JitState, shared_observed_generation)]);
    code.jne(dispatch, code.T_NEAR);
    EmitSharedPCValue(rax);
    code.mov(rcx, A64::LocationDescriptor::pc_mask);
    code.and_(rax, rcx);
    if (shared_static_fpcr) {
        const u64 fpcr = initial_location.Value() &
            (u64(A64::LocationDescriptor::fpcr_mask) << A64::LocationDescriptor::fpcr_shift);
        if (fpcr != 0) {
            code.mov(rcx, fpcr);
            code.or_(rax, rcx);
        }
    } else {
        code.mov(ecx, dword[r15 + offsetof(A64JitState, fpcr)]);
        code.and_(ecx, A64::LocationDescriptor::fpcr_mask);
        code.shl(rcx, A64::LocationDescriptor::fpcr_shift);
        code.or_(rax, rcx);
    }
    for (unsigned way = 0; way < 2; ++way) {
        Xbyak::Label next;
        const auto entry_offset = offset + way * sizeof(A64JitState::SharedDispatchCacheEntry);
        code.cmp(qword[r15 + entry_offset + offsetof(A64JitState::SharedDispatchCacheEntry, generation)], rdx);
        code.jne(next, code.T_SHORT);
        code.cmp(qword[r15 + entry_offset + offsetof(A64JitState::SharedDispatchCacheEntry, location)], rax);
        code.jne(next, code.T_SHORT);
        code.jmp(qword[r15 + entry_offset + offsetof(A64JitState::SharedDispatchCacheEntry, entrypoint)]);
        code.L(next);
    }
    // RBX is preserved by the C++ lookup callback. The tail jump is essential:
    // the owner prelude may refresh the reader epoch and reclaim this source.
    EmitSharedPCFlush();
    code.mov(ecx, dword[r15 + offset + offsetof(A64JitState::SharedDispatchCacheEntry, replacement)]);
    code.xor_(dword[r15 + offset + offsetof(A64JitState::SharedDispatchCacheEntry, replacement)],
              u32(sizeof(A64JitState::SharedDispatchCacheEntry)));
    code.lea(rbx, ptr[r15 + offset + rcx]);
    code.jmp(qword[r15 + offsetof(A64JitState, shared_indirect_lookup)]);
    code.L(dispatch);
    EmitSharedDispatch();
}

void A64EmitX64::EmitSharedReturn(IR::LocationDescriptor initial_location, bool is_single_step) {
    EmitSharedLoopFlush();
    Xbyak::Label fallback;
    if (!is_single_step && shared_link_resolver
        && conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
        code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
        code.jne(fallback, code.T_NEAR);
        if (conf.enable_cycle_counting) {
            code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
            code.jle(fallback, code.T_NEAR);
        }
        EmitSharedGeneration(rcx);
        code.cmp(rcx, qword[r15 + offsetof(A64JitState, shared_observed_generation)]);
        code.jne(fallback, code.T_NEAR);

        code.mov(eax, dword[r15 + offsetof(A64JitState, shared_rsb_ptr)]);
        code.sub(eax, 1);
        code.and_(eax, u32(A64JitState::RSBPtrMask));
        code.mov(dword[r15 + offsetof(A64JitState, shared_rsb_ptr)], eax);
        // Check before loading the cell address: its source region may have
        // retired while a callee passed through a reclamation safepoint.
        code.cmp(qword[r15 + offsetof(A64JitState, shared_rsb_generations) + rax * 8], rcx);
        code.jne(fallback, code.T_NEAR);
        EmitSharedPCValue(rbx);
        code.mov(rcx, A64::LocationDescriptor::pc_mask);
        code.and_(rbx, rcx);
        if (shared_static_fpcr) {
            const u64 fpcr = initial_location.Value() &
                (u64(A64::LocationDescriptor::fpcr_mask) << A64::LocationDescriptor::fpcr_shift);
            if (fpcr != 0) {
                code.mov(rcx, fpcr);
                code.or_(rbx, rcx);
            }
        } else {
            code.mov(ecx, dword[r15 + offsetof(A64JitState, fpcr)]);
            code.and_(ecx, A64::LocationDescriptor::fpcr_mask);
            code.shl(rcx, A64::LocationDescriptor::fpcr_shift);
            code.or_(rbx, rcx);
        }
        code.cmp(qword[r15 + offsetof(A64JitState, shared_rsb_locations) + rax * 8], rbx);
        code.jne(fallback, code.T_NEAR);
        code.mov(rax, qword[r15 + offsetof(A64JitState, shared_rsb_cells) + rax * 8]);
        code.jmp(qword[rax]);
        code.L(fallback);
    }
    EmitSharedIndirect(initial_location, is_single_step);
}

void A64EmitX64::RegisterFastmemPatch(u64 rip, FastmemPatchInfo info) {
    if (!shared_optimized) {
        fastmem_patch_info.emplace(rip, std::move(info));
        return;
    }
    info.recompile = false;
    const size_t bucket = ((rip >> 4) ^ (rip >> 16)) & (shared_fastmem_bucket_count - 1);
    auto& head = shared_fastmem_buckets[bucket];
    auto node = std::make_unique<SharedFastmemRecord>(SharedFastmemRecord{
        rip, std::move(info), head.load(std::memory_order_relaxed)});
    const auto* published = node.get();
    shared_fastmem_records.push_back(std::move(node));
    head.store(published, std::memory_order_release);
}

FakeCall A64EmitX64::SharedFastmemCallback(u64 rip) const {
    const size_t bucket = ((rip >> 4) ^ (rip >> 16)) & (shared_fastmem_bucket_count - 1);
    for (const auto* node = shared_fastmem_buckets[bucket].load(std::memory_order_acquire);
         node; node = node->next) {
        if (node->rip == rip) {
            return FakeCall{.call_rip = node->info.callback, .ret_rip = node->info.resume_rip};
        }
    }
    ASSERT_FALSE("Fault in shared JIT code is not a registered fastmem access: {:016x}", rip);
    return {};
}

#include "dynarmic/backend/x64/a64_baseline_emit.inc"
#include "dynarmic/backend/x64/a64_shared_loop.inc"

A64EmitX64::BlockDescriptor A64EmitX64::Emit(IR::Block& block) {
    if (conf.very_verbose_debugging_output) {
        std::puts(IR::DumpBlock(block).c_str());
    }

    code.EnableWriting();
    SCOPE_EXIT {
        if (!shared_optimized)
            code.DisableWriting();
    };

    PrepareSharedDispatch(block);
    std::vector<HostLoc> gpr_order = [this] {
        std::vector<HostLoc> gprs{any_gpr};
        if (conf.page_table) {
            gprs.erase(std::find(gprs.begin(), gprs.end(), HostLoc::R14));
        }
        if (conf.fastmem_pointer && (*conf.fastmem_pointer != 0 || !conf.optimize_zero_base_fastmem)) {
            gprs.erase(std::find(gprs.begin(), gprs.end(), HostLoc::R13));
        }
        return gprs;
    }();

    PrepareSharedLoop(block, gpr_order);
    RegAlloc reg_alloc{code, std::move(gpr_order), any_xmm};
    A64EmitContext ctx{conf, reg_alloc, block};

    // Start emitting.
    code.align();
    const u8* const entrypoint = code.getCurr();
    if (shared_optimized)
        EmitSharedLoopEntry();

    ASSERT(block.GetCondition() == IR::Cond::AL);

    for (auto iter = block.begin(); iter != block.end(); ++iter) {
        IR::Inst* inst = &*iter;

        // Call the relevant Emit* member function.
        switch (inst->GetOpcode()) {
#define OPCODE(name, type, ...)            \
    case IR::Opcode::name:                 \
        A64EmitX64::Emit##name(ctx, inst); \
        break;
#define A32OPC(...)
#define A64OPC(name, type, ...)               \
    case IR::Opcode::A64##name:               \
        A64EmitX64::EmitA64##name(ctx, inst); \
        break;
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

        default:
            ASSERT_MSG(false, "Invalid opcode: {}", inst->GetOpcode());
            break;
        }

        ctx.reg_alloc.EndOfAllocScope();

        if (conf.very_verbose_debugging_output) {
            EmitVerboseDebuggingOutput(reg_alloc);
        }
    }

    reg_alloc.AssertNoMoreUses();

    if (conf.enable_cycle_counting) {
        EmitAddCycles(block.CycleCount());
    }
    EmitX64::EmitTerminal(block.GetTerminal(), ctx.Location().SetSingleStepping(false), ctx.IsSingleStep());
    code.int3();

    for (auto& deferred_emit : ctx.deferred_emits) {
        deferred_emit();
    }
    code.int3();

    const size_t size = static_cast<size_t>(code.getCurr() - entrypoint);

    const A64::LocationDescriptor descriptor{block.Location()};
    if (shared_optimized) {
        // The shared cache owns immutable descriptors, invalidation and
        // publication. Do not install patchable private-cache bookkeeping;
        // the same location can be emitted again after a guest invalidation.
        // The worker seals the complete batch before publishing its entries.
        PerfMapRegister(entrypoint, entrypoint + size, LocationDescriptorToFriendlyName(descriptor));
        return BlockDescriptor{entrypoint, size};
    }
    const A64::LocationDescriptor end_location{block.EndLocation()};

    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);

    return RegisterBlock(descriptor, entrypoint, size);
}

void A64EmitX64::ClearCache() {
    ASSERT(!shared_optimized);  // Published shared code is retired as a region.
    EmitX64::ClearCache();
    block_ranges.ClearCache();
    ClearFastDispatchTable();
    fastmem_patch_info.clear();
}

void A64EmitX64::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    ASSERT(!shared_optimized);
    EmitX64::InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
}

void A64EmitX64::InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& locations) {
    ASSERT(!shared_optimized);
    const auto before = CachedBlockCount();
    block_ranges.RemoveLocations(locations);
    EmitX64::InvalidateBasicBlocks(locations);
    fastmem_invalidated_blocks += before - CachedBlockCount();
}


void A64EmitX64::ClearFastDispatchTable() {
    if (fast_dispatch_table && conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        std::fill_n(fast_dispatch_table.get(), fast_dispatch_table_size, FastDispatchEntry{});
    }
}

void A64EmitX64::GenTerminalHandlers() {
    // PC ends up in rbp, location_descriptor ends up in rbx
    const auto calculate_location_descriptor = [this] {
        // This calculation has to match up with A64::LocationDescriptor::UniqueHash
        // TODO: Optimization is available here based on known state of fpcr.
        code.mov(rbp, qword[r15 + offsetof(A64JitState, pc)]);
        code.mov(rcx, A64::LocationDescriptor::pc_mask);
        code.and_(rcx, rbp);
        code.mov(ebx, dword[r15 + offsetof(A64JitState, fpcr)]);
        code.and_(ebx, A64::LocationDescriptor::fpcr_mask);
        code.shl(rbx, A64::LocationDescriptor::fpcr_shift);
        code.or_(rbx, rcx);
    };

    Xbyak::Label fast_dispatch_cache_miss, rsb_cache_miss;

    code.align();
    terminal_handler_pop_rsb_hint = code.getCurr<const void*>();
    calculate_location_descriptor();
    code.mov(eax, dword[r15 + offsetof(A64JitState, rsb_ptr)]);
    code.sub(eax, 1);
    code.and_(eax, u32(A64JitState::RSBPtrMask));
    code.mov(dword[r15 + offsetof(A64JitState, rsb_ptr)], eax);
    code.cmp(rbx, qword[r15 + offsetof(A64JitState, rsb_location_descriptors) + rax * sizeof(u64)]);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jne(rsb_cache_miss);
    } else {
        code.jne(code.GetReturnFromRunCodeAddress());
    }
    code.mov(rax, qword[r15 + offsetof(A64JitState, rsb_codeptrs) + rax * sizeof(u64)]);
    code.jmp(rax);
    PerfMapRegister(terminal_handler_pop_rsb_hint, code.getCurr(), "a64_terminal_handler_pop_rsb_hint");

    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.align();
        terminal_handler_fast_dispatch_hint = code.getCurr<const void*>();
        calculate_location_descriptor();
        code.L(rsb_cache_miss);
        code.mov(r12, reinterpret_cast<u64>(fast_dispatch_table.get()));
        code.mov(rbp, rbx);
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(rbp, r12);
        }
        code.and_(ebp, fast_dispatch_table_mask);
        code.lea(rbp, ptr[r12 + rbp]);
        code.cmp(rbx, qword[rbp + offsetof(FastDispatchEntry, location_descriptor)]);
        code.jne(fast_dispatch_cache_miss);
        code.jmp(ptr[rbp + offsetof(FastDispatchEntry, code_ptr)]);
        code.L(fast_dispatch_cache_miss);
        code.LookupBlock();
        Xbyak::Label shared_fill, dispatch;
        code.cmp(byte[r15 + offsetof(A64JitState, last_lookup_is_shared)], 0);
        code.jne(shared_fill, code.T_NEAR);
        code.mov(qword[rbp + offsetof(FastDispatchEntry, location_descriptor)], rbx);
        code.mov(ptr[rbp + offsetof(FastDispatchEntry, code_ptr)], rax);
        code.jmp(dispatch, code.T_NEAR);
        code.L(shared_fill);
        if (conf.shared_code_cache && conf.enable_baseline_hot_optimization)
            EmitSharedDispatchCacheFill(rax, rbx);
        code.L(dispatch);
        code.jmp(rax);
        PerfMapRegister(terminal_handler_fast_dispatch_hint, code.getCurr(), "a64_terminal_handler_fast_dispatch_hint");

        code.align();
        fast_dispatch_table_lookup = code.getCurr<FastDispatchEntry& (*)(u64)>();
        code.mov(code.ABI_PARAM2, reinterpret_cast<u64>(fast_dispatch_table.get()));
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(code.ABI_PARAM1, code.ABI_PARAM2);
        }
        code.and_(code.ABI_PARAM1.cvt32(), fast_dispatch_table_mask);
        code.lea(code.ABI_RETURN, code.ptr[code.ABI_PARAM2 + code.ABI_PARAM1]);
        code.ret();
        PerfMapRegister(fast_dispatch_table_lookup, code.getCurr(), "a64_fast_dispatch_table_lookup");

    }
}


void A64EmitX64::EmitPushRSB(EmitContext& ctx, IR::Inst* inst) {
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
        return;
    }

    if (shared_optimized) {
        if (!shared_link_resolver)
            return;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[0].IsImmediate());
        const auto target = IR::LocationDescriptor{args[0].GetImmediateU64()};
        auto* const cell = std::construct_at(static_cast<std::atomic<const void*>*>(
            code.AllocateMutableData(sizeof(std::atomic<const void*>))), nullptr);
        const Xbyak::Reg64 index = ctx.reg_alloc.ScratchGpr();
        const Xbyak::Reg64 value = ctx.reg_alloc.ScratchGpr();
        code.mov(index.cvt32(), dword[r15 + offsetof(A64JitState, shared_rsb_ptr)]);
        code.mov(value, target.Value());
        code.mov(qword[r15 + offsetof(A64JitState, shared_rsb_locations) + index * 8], value);
        code.mov(value, reinterpret_cast<u64>(cell));
        code.mov(qword[r15 + offsetof(A64JitState, shared_rsb_cells) + index * 8], value);
        code.mov(value, qword[r15 + offsetof(A64JitState, shared_observed_generation)]);
        code.mov(qword[r15 + offsetof(A64JitState, shared_rsb_generations) + index * 8], value);
        code.add(index.cvt32(), 1);
        code.and_(index.cvt32(), u32(A64JitState::RSBPtrMask));
        code.mov(dword[r15 + offsetof(A64JitState, shared_rsb_ptr)], index.cvt32());
        ctx.deferred_emits.emplace_back([this, cell, target] {
            const void* const fallback = code.getCurr();
            EmitSharedLinkFallback(target);
            cell->store(fallback, std::memory_order_relaxed);
            shared_link_resolver(target.Value(), cell, fallback);
        });
        return;
    }

    EmitX64::EmitPushRSB(ctx, inst);
}

void A64EmitX64::EmitA64SetCheckBit(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg8 to_store = ctx.reg_alloc.UseGpr(args[0]).cvt8();
    code.mov(code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], to_store);
}

void A64EmitX64::EmitA64GetCFlag(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);
    code.shr(result, NZCV::x64_c_flag_bit);
    code.and_(result, 1);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.ScratchGpr().cvt32();

    code.mov(nzcv_raw, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);

    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr().cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pext(nzcv_raw, nzcv_raw, tmp);
        code.shl(nzcv_raw, 28);
    } else {
        code.and_(nzcv_raw, NZCV::x64_mask);
        code.imul(nzcv_raw, nzcv_raw, NZCV::from_x64_multiplier);
        code.and_(nzcv_raw, NZCV::arm_mask);
    }

    ctx.reg_alloc.DefineValue(inst, nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();

    code.shr(nzcv_raw, 28);
    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr().cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pdep(nzcv_raw, nzcv_raw, tmp);
    } else {
        code.imul(nzcv_raw, nzcv_raw, NZCV::to_x64_multiplier);
        code.and_(nzcv_raw, NZCV::x64_mask);
    }
    code.mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCV(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 to_store = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();
    if (shared_loop_flags)
        code.mov(r10d, to_store);
    else
        code.mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], to_store);
}

void A64EmitX64::EmitA64GetW(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();

    if (const auto host = shared_loop_registers[A64::RegNumber(reg)])
        code.mov(result, HostLocToReg64(*host).cvt32());
    else
        code.mov(result, dword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetX(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();

    if (const auto host = shared_loop_registers[A64::RegNumber(reg)])
        code.mov(result, HostLocToReg64(*host));
    else
        code.mov(result, qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetS(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movd(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetD(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movq(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetQ(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movaps(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetSP(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    code.mov(result, qword[r15 + offsetof(A64JitState, sp)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, dword[r15 + offsetof(A64JitState, fpcr)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

static u32 GetFPSRImpl(A64JitState* jit_state) {
    return jit_state->GetFpsr();
}

void A64EmitX64::EmitA64GetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(inst);
    code.mov(code.ABI_PARAM1, code.r15);
    code.stmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
    code.CallFunction(GetFPSRImpl);
}

void A64EmitX64::EmitA64SetW(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    if (const auto host = shared_loop_registers[A64::RegNumber(reg)]) {
        const auto destination = HostLocToReg64(*host).cvt32();
        if (args[1].IsImmediate())
            code.mov(destination, args[1].GetImmediateU32());
        else
            code.mov(destination, ctx.reg_alloc.UseGpr(args[1]).cvt32());
        return;
    }
    const auto addr = qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else {
        // TODO: zext tracking, xmm variant
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseScratchGpr(args[1]);
        code.mov(to_store.cvt32(), to_store.cvt32());
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64SetX(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    if (const auto host = shared_loop_registers[A64::RegNumber(reg)]) {
        const auto destination = HostLocToReg64(*host);
        if (args[1].IsImmediate())
            code.mov(destination, args[1].GetImmediateU64());
        else if (args[1].IsInXmm())
            code.movq(destination, ctx.reg_alloc.UseXmm(args[1]));
        else
            code.mov(destination, ctx.reg_alloc.UseGpr(args[1]));
        return;
    }
    const auto addr = qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else if (args[1].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[1]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64SetS(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
    const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm();
    // TODO: Optimize
    code.pxor(tmp, tmp);
    code.movss(tmp, to_store);
    code.movaps(addr, tmp);
}

void A64EmitX64::EmitA64SetD(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseScratchXmm(args[1]);
    code.movq(to_store, to_store);  // TODO: Remove when able
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetQ(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetSP(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[r15 + offsetof(A64JitState, sp)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[0]);
        code.mov(addr, to_store);
    }
}

static void SetFPCRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpcr(value);
}

void A64EmitX64::EmitA64SetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.r15);
    code.CallFunction(SetFPCRImpl);
    code.ldmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
}

static void SetFPSRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpsr(value);
}

void A64EmitX64::EmitA64SetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.r15);
    code.CallFunction(SetFPSRImpl);
    code.ldmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
}

void A64EmitX64::EmitA64SetPC(A64EmitContext& ctx, IR::Inst* inst) {
    if (inst == shared_static_pc_inst)
        return;  // The constant-edge fallback materializes this PC if needed.
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    if (inst == shared_dynamic_pc_inst) {
        // This is the last IR instruction of a plain indirect/return block.
        // Keep the target through the terminal to avoid a store-forwarding
        // dependency on every hit; slow exits materialize the guest PC.
        ctx.reg_alloc.Use(args[0], HostLoc::R11);
        shared_pc_in_r11 = true;
        return;
    }
    const auto addr = qword[r15 + offsetof(A64JitState, pc)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[0]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64CallSupervisor(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate());
    const u32 imm = args[0].GetImmediateU32();
    GetUserCallback<&A64::UserCallbacks::CallSVC>().EmitCall(code, [&](RegList param) {
        code.mov(param[0], imm);
    });
    // The kernel would have to execute ERET to get here, which would clear exclusive state.
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
}

void A64EmitX64::EmitA64ExceptionRaised(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate() && args[1].IsImmediate());
    const u64 pc = args[0].GetImmediateU64();
    const u64 exception = args[1].GetImmediateU64();
    GetUserCallback<&A64::UserCallbacks::ExceptionRaised>().EmitCall(code, [&](RegList param) {
        code.mov(param[0], pc);
        code.mov(param[1], exception);
    });
}

void A64EmitX64::EmitA64DataCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[1], args[2]);
    GetUserCallback<&A64::UserCallbacks::DataCacheOperationRaised>().EmitCall(code);
}

void A64EmitX64::EmitA64InstructionCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[0], args[1]);
    GetUserCallback<&A64::UserCallbacks::InstructionCacheOperationRaised>().EmitCall(code);
}

void A64EmitX64::EmitA64DataSynchronizationBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
    code.lfence();
}

void A64EmitX64::EmitA64DataMemoryBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
}

void A64EmitX64::EmitA64InstructionSynchronizationBarrier(A64EmitContext& ctx, IR::Inst*) {
    if (!conf.hook_isb) {
        return;
    }

    ctx.reg_alloc.HostCall(nullptr);
    GetUserCallback<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>().EmitCall(code);
}

void A64EmitX64::EmitA64GetCNTFRQ(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.cntfrq_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetCNTPCT(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(inst);
    if (!conf.wall_clock_cntpct) {
        code.UpdateTicks();
    }
    GetUserCallback<&A64::UserCallbacks::GetCNTPCT>().EmitCall(code);
}

void A64EmitX64::EmitA64GetCTR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.ctr_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetDCZID(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.dczid_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    if (shared_optimized) {
        Xbyak::Label absent;
        code.mov(result, qword[r15 + offsetof(A64JitState, runtime_tpidr_el0)]);
        code.test(result, result);
        code.jz(absent, code.T_SHORT);
        code.mov(result, qword[result]);
        code.L(absent);
    } else if (conf.tpidr_el0) {
        code.mov(result, u64(conf.tpidr_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetTPIDRRO(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    if (shared_optimized) {
        Xbyak::Label absent;
        code.mov(result, qword[r15 + offsetof(A64JitState, runtime_tpidrro_el0)]);
        code.test(result, result);
        code.jz(absent, code.T_SHORT);
        code.mov(result, qword[result]);
        code.L(absent);
    } else if (conf.tpidrro_el0) {
        code.mov(result, u64(conf.tpidrro_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64SetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg64 value = ctx.reg_alloc.UseGpr(args[0]);
    const Xbyak::Reg64 addr = ctx.reg_alloc.ScratchGpr();
    if (shared_optimized) {
        Xbyak::Label absent;
        code.mov(addr, qword[r15 + offsetof(A64JitState, runtime_tpidr_el0)]);
        code.test(addr, addr);
        code.jz(absent, code.T_SHORT);
        code.mov(qword[addr], value);
        code.L(absent);
    } else if (conf.tpidr_el0) {
        code.mov(addr, u64(conf.tpidr_el0));
        code.mov(qword[addr], value);
    }
}

std::string A64EmitX64::LocationDescriptorToFriendlyName(const IR::LocationDescriptor& ir_descriptor) const {
    const A64::LocationDescriptor descriptor{ir_descriptor};
    return fmt::format("a64_{:016X}_fpcr{:08X}",
                       descriptor.PC(),
                       descriptor.FPCR().Value());
}

void A64EmitX64::EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor, bool) {
    code.SwitchMxcsrOnExit();
    GetUserCallback<&A64::UserCallbacks::InterpreterFallback>().EmitCall(code, [&](RegList param) {
        code.mov(param[0], A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], param[0]);
        code.mov(param[1].cvt32(), terminal.num_instructions);
    });
    if (shared_optimized) {
        code.SwitchMxcsrOnEntry();
        EmitSharedDispatch();
    } else {
        code.ReturnFromRunCode(true);  // TODO: Check cycles
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::ReturnToDispatch, IR::LocationDescriptor, bool) {
    if (shared_optimized) {
        EmitSharedDispatch();
    } else {
        code.ReturnFromRunCode();
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (shared_optimized) {
        EmitSharedLink(terminal.next, initial_location, is_single_step);
        return;
    }
    if (!conf.HasOptimization(OptimizationFlag::BlockLinking) || is_single_step) {
        code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
        return;
    }

    if (conf.enable_cycle_counting) {
        code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);

        patch_information[terminal.next].jg.push_back(code.getCurr());
        if (const auto next_bb = GetBasicBlock(terminal.next)) {
            EmitPatchJg(terminal.next, next_bb->entrypoint);
        } else {
            EmitPatchJg(terminal.next);
        }
    } else {
        code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);

        patch_information[terminal.next].jz.push_back(code.getCurr());
        if (const auto next_bb = GetBasicBlock(terminal.next)) {
            EmitPatchJz(terminal.next, next_bb->entrypoint);
        } else {
            EmitPatchJz(terminal.next);
        }
    }

    code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
    code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
}

void A64EmitX64::EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (shared_optimized) {
        EmitSharedLink(terminal.next, initial_location, is_single_step);
        return;
    }
    if (!conf.HasOptimization(OptimizationFlag::BlockLinking) || is_single_step) {
        code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
        return;
    }

    patch_information[terminal.next].jmp.push_back(code.getCurr());
    if (auto next_bb = GetBasicBlock(terminal.next)) {
        EmitPatchJmp(terminal.next, next_bb->entrypoint);
    } else {
        EmitPatchJmp(terminal.next);
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::PopRSBHint, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (shared_optimized) {
        EmitSharedReturn(initial_location, is_single_step);
        return;
    }
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) || is_single_step) {
        code.ReturnFromRunCode();
        return;
    }

    code.jmp(terminal_handler_pop_rsb_hint);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::FastDispatchHint, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (shared_optimized) {
        if (shared_static_dispatch)
            EmitSharedLink(*shared_static_dispatch, initial_location, is_single_step);
        else
            EmitSharedIndirect(initial_location, is_single_step);
        return;
    }
    if (!conf.HasOptimization(OptimizationFlag::FastDispatch) || is_single_step) {
        code.ReturnFromRunCode();
        return;
    }

    code.jmp(terminal_handler_fast_dispatch_hint);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    switch (terminal.if_) {
    case IR::Cond::AL:
    case IR::Cond::NV:
        EmitTerminal(terminal.then_, initial_location, is_single_step);
        break;
    default:
        Xbyak::Label pass = EmitCond(terminal.if_, shared_loop_flags);
        EmitTerminal(terminal.else_, initial_location, is_single_step);
        code.L(pass);
        EmitTerminal(terminal.then_, initial_location, is_single_step);
        break;
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    Xbyak::Label fail;
    code.cmp(code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], u8(0));
    code.jz(fail);
    EmitTerminal(terminal.then_, initial_location, is_single_step);
    code.L(fail);
    EmitTerminal(terminal.else_, initial_location, is_single_step);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    EmitTerminal(terminal.else_, initial_location, is_single_step);
}

void A64EmitX64::EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jg(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jg(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jz(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jz(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (target_code_ptr) {
        code.jmp(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jmp(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 22);
}

void A64EmitX64::EmitPatchMovRcx(CodePtr target_code_ptr) {
    if (!target_code_ptr) {
        target_code_ptr = code.GetReturnFromRunCodeAddress();
    }
    const CodePtr patch_location = code.getCurr();
    code.mov(code.rcx, reinterpret_cast<u64>(target_code_ptr));
    code.EnsurePatchLocationSize(patch_location, 10);
}

void A64EmitX64::Unpatch(const IR::LocationDescriptor& location) {
    EmitX64::Unpatch(location);
    // Covers both range invalidation and fastmem fault-driven recompilation.
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.DisableWriting();
        (*fast_dispatch_table_lookup)(location.Value()) = {};
        code.EnableWriting();
    }
}

}  // namespace Dynarmic::Backend::X64
