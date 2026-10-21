/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/interface/A64/config.h"

namespace Dynarmic::Backend::X64 {

/// Shared native code never embeds the producing thread's callback object or
/// vtable. The ordinary virtual thunk also permits different callback classes
/// to execute the same published code without retaining a prototype object.
template<auto member>
ArgCallback MakeA64Callback(A64::UserCallbacks* callbacks, bool shared_optimized) {
    if (shared_optimized) {
        return ArgCallback::FromRuntimeState(
            &impl::ThunkBuilder<decltype(member), member>::Thunk,
            offsetof(A64JitState, runtime_callbacks));
    }
    return Devirtualize<member>(callbacks);
}

}  // namespace Dynarmic::Backend::X64
