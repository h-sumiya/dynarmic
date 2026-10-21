#if A2E_DYNARMIC_SHARDED_MONITOR
// A2E_SHARDED_EXCLUSIVE_HELPERS_V1
template<typename UserConfig>
void EmitExclusiveLock(BlockOfCode& code, const UserConfig& conf, Xbyak::Reg64 vaddr, Xbyak::Reg64 pointer, Xbyak::Reg32 tmp) {
    // Keep the hash identical to ExclusiveMonitor::BucketIndex. The pointer is
    // live until unlock, including across the existing fault callback stubs.
    code.mov(pointer, vaddr);
    code.shr(pointer, ExclusiveMonitor::GRANULE_SHIFT);
    code.mov(tmp.cvt64(), pointer);
    code.shr(tmp.cvt64(), ExclusiveMonitor::BUCKET_SHIFT);
    code.xor_(pointer, tmp.cvt64());
    code.and_(pointer, ExclusiveMonitor::BUCKET_MASK);
    code.shl(pointer, 6);
    code.mov(tmp.cvt64(), mcl::bit_cast<u64>(GetExclusiveMonitorBuckets(conf.global_monitor)));
    code.add(pointer, tmp.cvt64());
    EmitSpinLockLock(code, pointer, tmp);
}

inline void EmitExclusiveUnlock(BlockOfCode& code, Xbyak::Reg64 pointer, Xbyak::Reg32 tmp) {
    EmitSpinLockUnlock(code, pointer, tmp);
}

template<typename UserConfig>
void EmitExclusiveBumpEpoch(BlockOfCode& code, const UserConfig& conf, Xbyak::Reg64 bucket) {
    code.inc(qword[bucket + offsetof(ExclusiveMonitor::Bucket, epoch)]);
    Xbyak::Label done;
    code.jnz(done, code.T_NEAR);
    // A wrap is practically unreachable, but must not turn an ancient saved
    // epoch into a valid reservation. The cold path clears atomic epoch fields
    // without acquiring another bucket while this bucket's lock is held.
    code.sub(rsp, 8);  // The ABI save helper expects entry after CALL.
    ABI_PushCallerSaveRegistersAndAdjustStack(code);
    if (bucket.getIdx() != code.ABI_PARAM2.getIdx()) {
        code.mov(code.ABI_PARAM2, bucket);
    }
    code.mov(code.ABI_PARAM1, mcl::bit_cast<u64>(conf.global_monitor));
    code.CallLambda([](ExclusiveMonitor& monitor, ExclusiveMonitor::Bucket& wrapped) {
        monitor.ResetEpochAfterWrap(wrapped);
    });
    ABI_PopCallerSaveRegistersAndAdjustStack(code);
    code.add(rsp, 8);
    code.L(done);
}


#else
#include "dynarmic/backend/x64/emit_exclusive_helpers_original.inc"
#endif
