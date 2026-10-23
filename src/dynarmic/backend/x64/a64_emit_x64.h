/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/x64/a64_baseline_cache.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/a64_runtime_callback.h"
#include "dynarmic/backend/x64/a64_baseline_compiler.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::Backend::X64 {

class RegAlloc;

struct A64EmitContext final : public EmitContext {
    A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block);

    A64::LocationDescriptor Location() const;
    bool IsSingleStep() const;
    FP::FPCR FPCR(bool fpcr_controlled = true) const override;

    bool HasOptimization(OptimizationFlag flag) const override {
        return conf.HasOptimization(flag);
    }

    const A64::UserConfig& conf;
};

class A64EmitX64 final : public EmitX64 {
public:
    A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface,
               bool shared_optimized = false);
    ~A64EmitX64() override;

    /**
     * Emit host machine code for a basic block with intermediate representation `block`.
     * @note block is modified.
     */
    BlockDescriptor Emit(IR::Block& block);
    // Register source-owned mutable link data; generated branch bytes remain
    // immutable and the cache updates only the pointed-to atomic cell.
    void SetSharedLinkResolver(std::function<void(u64, std::atomic<const void*>*, const void*)> resolver) {
        shared_link_resolver = std::move(resolver);
    }
    BlockDescriptor EmitBaseline(A64::LocationDescriptor location, const A64BaselineBlock& block);
    // Call from the immutable thread dispatcher before EmitBaseline. The
    // returned retirements require the owner to reset its RSB and hot samples.
    BaselineReclamation PrepareBaselineEmission(size_t body_size);
    BaselineCacheUsage GetBaselineCacheUsage() const { return baseline_cache.Usage(); }
    void ResetBaselineCache() { baseline_cache.Reset(); }
    const void* GetBaselineReturn() const { return baseline_return; }
    const void* GetBaselineHotReturn() const { return baseline_hot_return; }
    const void* GetSharedIndirectLookup() const { return shared_indirect_lookup; }

    void ClearCache() override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);
    // The fastmem fault path directly invalidates descriptors rather than a
    // guest range. Keep its metadata and statistics synchronized as well.
    void InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& locations);
    u64 FastmemInvalidatedBlockCount() const { return fastmem_invalidated_blocks; }

protected:
    const bool shared_optimized;
    const A64::UserConfig conf;
    A64::Jit* jit_interface;
    BlockRangeInformation<u64> block_ranges;
    u64 fastmem_invalidated_blocks = 0;

    struct FastDispatchEntry {
        u64 location_descriptor = 0xFFFF'FFFF'FFFF'FFFFull;
        const void* code_ptr = nullptr;
    };
    static_assert(sizeof(FastDispatchEntry) == 0x10);
    static constexpr u64 fast_dispatch_table_mask = 0xFFFFF0;
    static constexpr size_t fast_dispatch_table_size = 0x100000;
    // Shared emitters tail-dispatch through the executing thread's prelude and
    // therefore need no private 16 MiB dispatch table of their own.
    std::unique_ptr<FastDispatchEntry[]> fast_dispatch_table;
    void ClearFastDispatchTable();

    void (*memory_read_128)();
    void (*memory_write_128)();
    void (*memory_exclusive_write_128)();
    void GenMemory128Accessors();

    std::map<std::tuple<bool, size_t, int, int>, void (*)()> read_fallbacks;
    std::map<std::tuple<bool, size_t, int, int>, void (*)()> write_fallbacks;
    std::map<std::tuple<bool, size_t, int, int>, void (*)()> exclusive_write_fallbacks;
    void GenFastmemFallbacks();

    struct DeferredRegister {
        u8 index;
        bool word;
        RegAlloc::ValueSnapshot value;
    };
    void PrepareDeferredState(IR::Block& block);
    std::vector<DeferredRegister> CaptureDeferredState(A64EmitContext& ctx, IR::Inst* inst);
    std::pair<u64, u64> EmitDeferredMemoryFallback(A64EmitContext& ctx, IR::Inst* inst,
        const void* wrapped, Xbyak::Label& end, const std::vector<DeferredRegister>& state);

    const void* terminal_handler_pop_rsb_hint = nullptr;
    const void* terminal_handler_fast_dispatch_hint = nullptr;
    FastDispatchEntry& (*fast_dispatch_table_lookup)(u64) = nullptr;
    void GenTerminalHandlers();
    void GenBaselineReturn();
    const void* baseline_return = nullptr;
    const void* baseline_hot_return = nullptr;
    const void* shared_indirect_lookup = nullptr;

    // Only the owning execution thread reads or changes this arena metadata.
    // Shared optimized emitters never allocate from it.
    A64BaselineCodeCache baseline_cache;

    // Microinstruction emitters
    void EmitPushRSB(EmitContext& ctx, IR::Inst* inst);
#define OPCODE(...)
#define A32OPC(...)
#define A64OPC(name, type, ...) void EmitA64##name(A64EmitContext& ctx, IR::Inst* inst);
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

    // Helpers
    std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const override;

    // Fastmem information
    using DoNotFastmemMarker = std::tuple<IR::LocationDescriptor, unsigned>;
    struct FastmemPatchInfo {
        u64 resume_rip;
        u64 callback;
        DoNotFastmemMarker marker;
        bool recompile;
    };
    tsl::robin_map<u64, FastmemPatchInfo> fastmem_patch_info;
    std::set<DoNotFastmemMarker> do_not_fastmem;
    std::optional<DoNotFastmemMarker> ShouldFastmem(A64EmitContext& ctx, IR::Inst* inst) const;
    FakeCall FastmemCallback(u64 rip);

    // The compiler is the only writer. Signal/exception handlers traverse only
    // immutable nodes, published with release/acquire and retained until the
    // complete shared region is destroyed after its reader grace period.
    struct SharedFastmemRecord {
        u64 rip;
        FastmemPatchInfo info;
        const SharedFastmemRecord* next;
    };
    static constexpr size_t shared_fastmem_bucket_count = 4096;
    std::unique_ptr<std::atomic<const SharedFastmemRecord*>[]> shared_fastmem_buckets;
    std::vector<std::unique_ptr<SharedFastmemRecord>> shared_fastmem_records;
    void RegisterFastmemPatch(u64 rip, FastmemPatchInfo info);
    FakeCall SharedFastmemCallback(u64 rip) const;

    template<auto callback>
    ArgCallback GetUserCallback() const {
        return MakeA64Callback<callback>(conf.callbacks, shared_optimized);
    }
    void EmitSharedDispatch();
    std::function<void(u64, std::atomic<const void*>*, const void*)> shared_link_resolver;
    void EmitSharedPCValue(Xbyak::Reg64 target);
    void EmitSharedPCFlush();
    void PrepareSharedDispatch(IR::Block& block);
    void EmitSharedLinkFallback(IR::LocationDescriptor target);
    void EmitSharedReturn(IR::LocationDescriptor initial_location, bool is_single_step);
    void EmitSharedIndirect(IR::LocationDescriptor initial_location, bool is_single_step);
    void EmitSharedDispatchCacheIndex(Xbyak::Reg64 index, Xbyak::Reg64 location, Xbyak::Reg64 scratch);
    void EmitSharedDispatchCacheFill(Xbyak::Reg64 entrypoint, Xbyak::Reg64 location);
    void EmitSharedLink(IR::LocationDescriptor next, IR::LocationDescriptor initial_location, bool is_single_step);
    std::optional<IR::LocationDescriptor> shared_static_dispatch;
    const IR::Inst* shared_static_pc_inst = nullptr;
    const IR::Inst* shared_dynamic_pc_inst = nullptr;
    bool shared_pc_in_r11 = false;
    bool shared_static_fpcr = false;
    CodePtr shared_block_entrypoint = nullptr;
    // A pure shared loop can retain selected architectural GPRs across its
    // self edge. External entries initialize them; all exits materialize them.
    std::array<std::optional<HostLoc>, 31> shared_loop_registers{};
    bool shared_loop_flags = false;
    void PrepareSharedLoop(IR::Block& block, std::vector<HostLoc>& gpr_order);
    void EmitSharedLoopEntry();
    void EmitSharedLoopFlush();

    void EmitRuntimeConfigPointer(Xbyak::Reg64 target);
    void EmitRuntimeReservationPointer(Xbyak::Reg64 target);

    // Memory access helpers
    void EmitCheckMemoryAbort(A64EmitContext& ctx, IR::Inst* inst, Xbyak::Label* end = nullptr);
    void EmitExclusiveAlignmentCheck(A64EmitContext& ctx, IR::Inst* inst, Xbyak::Reg64 vaddr, std::size_t bitsize);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryRead(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryWrite(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemoryInline(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemoryInline(A64EmitContext& ctx, IR::Inst* inst);

    // Terminal instruction emitters
    void EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::ReturnToDispatch terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::PopRSBHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::FastDispatchHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;
    void EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) override;

    // Patching
    void Unpatch(const IR::LocationDescriptor& target_desc) override;
    void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) override;
};

}  // namespace Dynarmic::Backend::X64
