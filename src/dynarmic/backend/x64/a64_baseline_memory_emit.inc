// Copyright (c) 2026 apk2exe contributors
// SPDX-License-Identifier: MIT
// Included after the Emitter class definition, inside its namespace.

Emitter::Result Emitter::MemoryInstruction(u32 instruction) {
    BaselineMemoryOp op;
    if (!DecodeBaselineMemory(instruction, op))
        return Result::Unsupported;
    using Kind = BaselineMemoryOp::Kind;
    using Address = BaselineMemoryOp::Address;

    const bool contiguous_structure = op.kind == Kind::StructureMultiple && op.structures == 1;
    const bool native = FastMemory() &&
        (op.kind == Kind::Simd || op.kind == Kind::SimdPair ||
         op.kind == Kind::SimdLiteral || contiguous_structure);
    if (!native) {
        EmitHelper(offsetof(A64JitState, baseline_memory_instruction), instruction);
        return Result::Continue;
    }

    const unsigned count = contiguous_structure ? op.repetitions : op.pair ? 2 : 1;
    const unsigned bytes = contiguous_structure ? (op.quad ? 16 : 8) : op.bytes;
    std::array<unsigned, 4> destinations{};
    for (unsigned n = 0; n < count; ++n)
        destinations[n] = contiguous_structure ? (op.rt + n) % 32 : n == 0 ? op.rt : op.rt2;

    // Preserve hot GPR/vector cache mappings across memory instructions. Cold
    // paths are emitted before the common destination commit, so their saved
    // mappings describe the precise state before this guest instruction.
    const auto incoming_slots = slots;
    const auto incoming_vectors = vector_slots;
    if (op.address == Address::Literal) {
        lea(rbx, ptr[r12 + offset]);
        AddOffset(rbx, op.displacement);
    } else {
        Read(op.rn, rbx, true, true);
        if (op.Writeback()) {
            mov(rbp, rbx);
            if (contiguous_structure && op.rm != 31) {
                Read(op.rm, rcx);
                add(rbp, rcx);
            } else {
                AddOffset(rbp, contiguous_structure ? count * bytes : op.displacement);
            }
        }
        if (op.address == Address::RegisterOffset) {
            Read(op.rm, rcx);
            Extend(rcx, op.option, true);
            if (op.shift)
                shl(rcx, op.shift);
            add(rbx, rcx);
        } else if (op.address != Address::PostIndex) {
            AddOffset(rbx, op.displacement);
        }
    }

    std::array<u32, 4> access_offsets{};
    for (unsigned n = 0; n < count; ++n) {
        const Xbyak::Xmm temporary{static_cast<int>(n)};
        if (!op.load) {
            ReadVector(destinations[n], temporary);
            if (bytes < 4)
                movd(eax, temporary);
        }
        access_offsets[n] = static_cast<u32>(getSize());
        if (op.load) {
            switch (bytes) {
            case 1: movzx(eax, byte[r13 + rbx + n * bytes]); movd(temporary, eax); break;
            case 2: movzx(eax, word[r13 + rbx + n * bytes]); movd(temporary, eax); break;
            case 4: movd(temporary, dword[r13 + rbx + n * bytes]); break;
            case 8: movq(temporary, qword[r13 + rbx + n * bytes]); break;
            case 16: movdqu(temporary, xword[r13 + rbx + n * bytes]); break;
            }
        } else {
            switch (bytes) {
            case 1: mov(byte[r13 + rbx + n * bytes], al); break;
            case 2: mov(word[r13 + rbx + n * bytes], ax); break;
            case 4: movd(dword[r13 + rbx + n * bytes], temporary); break;
            case 8: movq(qword[r13 + rbx + n * bytes], temporary); break;
            case 16: movdqu(xword[r13 + rbx + n * bytes], temporary); break;
            }
        }
    }
    Xbyak::Label commit;
    jmp(commit, T_NEAR);

    // Each access has its own recovery location. Earlier successful stores
    // are never repeated. Earlier loaded values are staged before a C++ call
    // can overwrite the volatile XMM registers, and all destinations commit
    // together only after the remaining accesses succeed.
    for (unsigned n = 0; n < count; ++n) {
        faults[fault_count++] = {access_offsets[n], static_cast<u32>(getSize())};
        if (n != 0) {
            if (op.load) {
                for (unsigned previous = 0; previous < n; ++previous)
                    movdqu(xword[r15 + offsetof(A64JitState, baseline_memory_saved) + previous * 16],
                           Xbyak::Xmm{static_cast<int>(previous)});
            }
            mov(byte[r15 + offsetof(A64JitState, baseline_memory_phase)], n);
        }
        EmitHelper(offsetof(A64JitState, baseline_memory_instruction), instruction, false, false);
        // EmitHelper flushes/invalidates the compile-time cache. Restore the
        // incoming mapping for this edge, reloading every slot from the helper's
        // now-committed architectural state before joining the native edge.
        slots = incoming_slots;
        vector_slots = incoming_vectors;
        ReloadAfterMemoryCallback();
        if (op.load) {
            for (unsigned value = 0; value < count; ++value)
                movdqu(Xbyak::Xmm{static_cast<int>(value)},
                       xword[r15 + offsetof(A64JitState, vec) + destinations[value] * 16]);
        }
        if (op.Writeback())
            Read(op.rn, rbp, true, true);
        if (n + 1 < count)
            jmp(commit, T_NEAR);
    }
    L(commit);
    if (op.load) {
        for (unsigned n = 0; n < count; ++n)
            WriteVector(destinations[n], Xbyak::Xmm{static_cast<int>(n)});
    }
    if (op.Writeback())
        Write(op.rn, rbp, true, true);
    return Result::Continue;
}
