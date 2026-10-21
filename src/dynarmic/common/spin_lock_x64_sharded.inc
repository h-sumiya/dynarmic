/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mutex>

#include <xbyak/xbyak.h>

#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/hostloc.h"
#include "dynarmic/common/spin_lock.h"

namespace Dynarmic {

void EmitSpinLockLock(Xbyak::CodeGenerator& code, Xbyak::Reg64 ptr, Xbyak::Reg32 tmp) {
    // Test-and-test-and-set: a contended waiter polls with shared reads instead
    // of repeatedly bouncing the lock cache line with locked XCHG operations.
    Xbyak::Label attempt, wait, done;
    code.L(attempt);
    code.mov(tmp, 1);
    code.xchg(code.dword[ptr], tmp);  // XCHG memory is implicitly locked.
    code.test(tmp, tmp);
    code.jz(done);
    code.L(wait);
    code.pause();
    code.cmp(code.dword[ptr], 0);
    code.jnz(wait);
    code.jmp(attempt);
    code.L(done);
}

void EmitSpinLockUnlock(Xbyak::CodeGenerator& code, Xbyak::Reg64 ptr, [[maybe_unused]] Xbyak::Reg32 tmp) {
    // Aligned MOV is a release store on x64. LOCK CMPXCHG at the guest store
    // supplies its stronger ordering; the monitor unlock needs no extra fence.
    code.mov(code.dword[ptr], 0);
}

namespace {

struct SpinLockImpl {
    void Initialize();

    Xbyak::CodeGenerator code;

    void (*lock)(volatile int*);
    void (*unlock)(volatile int*);
};

std::once_flag flag;
SpinLockImpl impl;

void SpinLockImpl::Initialize() {
    const Xbyak::Reg64 ABI_PARAM1 = Backend::X64::HostLocToReg64(Backend::X64::ABI_PARAM1);

    code.align();
    lock = code.getCurr<void (*)(volatile int*)>();
    EmitSpinLockLock(code, ABI_PARAM1, code.eax);
    code.ret();

    code.align();
    unlock = code.getCurr<void (*)(volatile int*)>();
    EmitSpinLockUnlock(code, ABI_PARAM1, code.eax);
    code.ret();
}

}  // namespace

void SpinLock::Lock() {
    std::call_once(flag, &SpinLockImpl::Initialize, impl);
    impl.lock(&storage);
}

void SpinLock::Unlock() {
    std::call_once(flag, &SpinLockImpl::Initialize, impl);
    impl.unlock(&storage);
}

}  // namespace Dynarmic
