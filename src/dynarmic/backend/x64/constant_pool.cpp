/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/constant_pool.h"

#include <cstring>

#include <mcl/assert.hpp>

#include "dynarmic/backend/x64/block_of_code.h"

namespace Dynarmic::Backend::X64 {

ConstantPool::ConstantPool(BlockOfCode& code, size_t size)
        : code(code), insertion_point(0) {
    code.EnsureMemoryCommitted(align_size + size);
    code.int3();
    code.align(align_size);
    pool = std::span<ConstantT>(
        reinterpret_cast<ConstantT*>(code.AllocateFromCodeSpace(size)), size / align_size);
}

Xbyak::Address ConstantPool::GetConstant(const Xbyak::AddressFrame& frame, u64 lower, u64 upper) {
    const auto constant = ConstantT(lower, upper);
    auto iter = constant_info.find(constant);
    if (iter == constant_info.end()) {
        // A shared region can fill its immutable constant pool before its
        // instruction pages. Let the worker retire it and retry in a fresh
        // region instead of terminating the process on normal capacity use.
        if (insertion_point >= pool.size())
            throw Xbyak::Error(Xbyak::ERR_CODE_IS_TOO_BIG);
        ConstantT& target_constant = pool[insertion_point];
        target_constant = constant;
        iter = constant_info.insert({constant, &target_constant}).first;
        ++insertion_point;
    }
    return frame[code.rip + iter->second];
}

void* ConstantPool::AllocateMutableData(size_t bytes) {
    const size_t slots = bytes / align_size + ((bytes % align_size) != 0);
    if (slots > pool.size() - insertion_point)
        throw Xbyak::Error(Xbyak::ERR_CODE_IS_TOO_BIG);
    auto* result = pool.data() + insertion_point;
    insertion_point += slots;
    return result;
}

}  // namespace Dynarmic::Backend::X64
