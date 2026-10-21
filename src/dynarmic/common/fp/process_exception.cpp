/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/common/fp/process_exception.h"

#include <mcl/assert.hpp>

#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"

namespace Dynarmic::FP {

// apk2exe precise baseline FP trap hook.
thread_local BaselineTrapHandler baseline_trap_handler = nullptr;

void FPProcessException(FPExc exception, FPCR fpcr, FPSR& fpsr) {
    switch (exception) {
    case FPExc::InvalidOp:
        if (fpcr.IOE()) {
            if (baseline_trap_handler) baseline_trap_handler(exception);
            ASSERT_FALSE("Raising floating point exceptions unimplemented");
        }
        fpsr.IOC(true);
        break;
    case FPExc::DivideByZero:
        if (fpcr.DZE()) {
            if (baseline_trap_handler) baseline_trap_handler(exception);
            ASSERT_FALSE("Raising floating point exceptions unimplemented");
        }
        fpsr.DZC(true);
        break;
    case FPExc::Overflow:
        if (fpcr.OFE()) {
            if (baseline_trap_handler) baseline_trap_handler(exception);
            ASSERT_FALSE("Raising floating point exceptions unimplemented");
        }
        fpsr.OFC(true);
        break;
    case FPExc::Underflow:
        if (fpcr.UFE()) {
            if (baseline_trap_handler) baseline_trap_handler(exception);
            ASSERT_FALSE("Raising floating point exceptions unimplemented");
        }
        fpsr.UFC(true);
        break;
    case FPExc::Inexact:
        if (fpcr.IXE()) {
            if (baseline_trap_handler) baseline_trap_handler(exception);
            ASSERT_FALSE("Raising floating point exceptions unimplemented");
        }
        fpsr.IXC(true);
        break;
    case FPExc::InputDenorm:
        if (fpcr.IDE()) {
            if (baseline_trap_handler) baseline_trap_handler(exception);
            ASSERT_FALSE("Raising floating point exceptions unimplemented");
        }
        fpsr.IDC(true);
        break;
    default:
        UNREACHABLE();
        break;
    }
}

}  // namespace Dynarmic::FP
