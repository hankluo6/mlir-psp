# MLIR CPU Backend for PPSSPP

Build an experimental, correctness-first MLIR CPU backend for PPSSPP.

## Goal

The preferred pipeline is:

```text
PSP MIPS instructions
→ PPSSPP MIPS frontend / IR
→ MLIR
→ LLVM IR
→ native host code
→ execution inside PPSSPP
```

Reusing PPSSPP's existing IR is preferred over implementing a second MIPS
lifter because it should preserve more of PPSSPP's decoded instruction
semantics, block formation, and delay-slot handling. A direct decoded-MIPS to
MLIR path remains an option if PPSSPP IR proves unsuitable.

PPSSPP remains responsible for PSP memory, HLE/syscalls, GPU, audio, input,
filesystem, timing, game loading, debugging, and save states.

## Scope and expectations

This is initially a research and integration backend, not a replacement for
PPSSPP's highly optimized native JIT. Correctness and a complete end-to-end
path take priority over compile time, runtime performance, optimization, and
ISA coverage. The first supported host is x86-64 desktop. LLVM and MLIR are
pinned to a known-compatible version.

## Initial execution contract

Use a small C-compatible entry ABI, conceptually:

```cpp
struct MLIRBlockResult {
    uint32_t nextPC;
    uint32_t exitReason;
};

MLIRBlockResult ExecuteMLIRBlock(MIPSState *state, uint8_t *memoryBase);
```

The exact ABI may use an out-parameter or packed integer if that is easier for
LLVM's JIT calling convention. Generated code initially accesses guest
registers through `MIPSState`; it does not attempt whole-block SSA register
reconstruction.

Every block must preserve `$zero`, produce the correct next guest PC, implement
delay slots, update `downcount`, return when PPSSPP must process timing or an
exception, and synchronize observable state before returning or calling a
helper.

## Memory model

The first implementation uses PPSSPP memory helpers for guest loads and stores.
This keeps address translation, MMIO, invalid addresses, endianness, alignment,
and fault handling within PPSSPP. Direct mapped-memory access is a later
optimization and must retain helper slow paths.

## Compilation and cache policy

- Compile one PPSSPP basic block at a time.
- Cache by guest start address and any execution-mode state affecting codegen.
- Integrate with `InvalidateICache()`, `ClearJitCache()`, and pending cache
  clears so self-modifying code cannot execute stale blocks.
- Initially use a side table rather than patching guest memory with private
  dispatch opcodes.
- Compile synchronously; background compilation is out of scope.

## Unsupported operations and exits

Fallback is block-granular for the MVP. If any instruction is unsupported, do
not install the compiled block; execute the original block with PPSSPP's IR
interpreter (preferred) or interpreter. Mid-block fallback is deferred because
it requires precise synchronization of PC, cycles, delay state, exceptions,
and registers.

Exit reasons should distinguish normal control flow, timing expiration,
fallback, exceptions or memory-helper exits, HLE/replacement operations, and
cache invalidation/backend shutdown.

## ISA rollout

1. Integer constants, moves, arithmetic, logic, and shifts.
2. Comparisons and straight-line blocks.
3. Loads and stores through PPSSPP helpers.
4. Conditional branches, jumps, link operations, and delay slots.
5. HI/LO multiply and divide operations.
6. Unaligned memory operations and LL/SC.
7. Scalar FPU and control state.
8. VFPU instructions and prefix/control state.

Syscalls, breakpoints, replacement instructions, and other dispatcher-visible
operations exit through PPSSPP rather than being reimplemented in MLIR.

## Verification strategy

Differential testing against PPSSPP is the correctness oracle. Each test starts
from identical `MIPSState` and memory, executes the same block with the
reference core and MLIR backend, then compares registers, PC, HI/LO, modified
memory, delay/exception state, cycles, and exit reason. FPU/VFPU state is added
as those instructions become supported.

Use hand-written edge cases and randomized instruction sequences. Failures must
log enough initial state and instruction words to reproduce them. Run MLIR's
verifier before lowering and LLVM's verifier before JIT execution.

## Milestones

### M0: Repository and toolchain

- Add pinned PPSSPP source (submodule or documented checkout).
- Record the compatible LLVM/MLIR revision.
- Add a desktop-only `USE_MLIR_CPU_BACKEND` build option.
- Leave ordinary PPSSPP builds unaffected when disabled.

### M1: Frontend feasibility probe

- Feed a short MIPS sequence through PPSSPP's existing frontend.
- Capture PPSSPP IR and document its block boundaries and exits.
- Decide with an executable probe between PPSSPP-IR-to-MLIR and direct
  decoded-MIPS-to-MLIR.

### M2: Standalone vertical slice

- Translate a straight-line integer block to MLIR.
- Lower it through LLVM and JIT it on x86-64.
- Execute it against `MIPSState` through the defined ABI.
- Compare the state against PPSSPP's reference execution.

### M3: Memory and control flow

- Add helper-based memory operations.
- Add branch/jump exits and correct delay slots.
- Add cycle accounting, exit reasons, and whole-block fallback.

### M4: PPSSPP integration

- Implement backend lifecycle, dispatcher, cache, and invalidation hooks.
- Integrate behind a compile-time flag and internal/debug CPU-core value.
- Preserve save-state, debugger, stepping, and core-switch behavior.

### M5: First program

- Expose `CPU Core: MLIR JIT` after the internal core is stable.
- Run a small homebrew program with measured MLIR block execution.
- Report compiled/fallback blocks, compile failures, and exit reasons.

## MVP success criteria

A small PSP homebrew program runs correctly inside PPSSPP, some blocks execute
as MLIR-generated native code, unsupported blocks safely use the reference
core, self-modifying-code invalidation works, and differential tests cover the
implemented instruction subset.

## Explicitly deferred

- Competitive performance with PPSSPP's native JIT.
- Cross-block linking, traces, and superblocks.
- Full SSA reconstruction and guest register allocation.
- Direct fast-memory accesses.
- Background or tiered compilation.
- Non-x86-64 hosts.
- Complete FPU/VFPU coverage.

## TODO / cleanup backlog

- Split the current `adapter/` directory by responsibility:
  move the production PPSSPP/MLIR target definitions into `integration/`,
  move `PPSSPPReferenceAdapter.cpp` and its CTest declarations into `tests/`,
  and keep PPSSPP injection-only logic under `cmake/`.
- Update `InjectPPSSPPAdapter.cmake`, `README.md`, the root CMake test
  adapter path, and `.vscode/launch.json` together when that split is made.
- Rename `MLIR_PSP_ADAPTER_SOURCE_DIR` to a less ambiguous project/integration
  source variable after the directory split.
- Revisit Windows asset deployment: avoid copying the complete `assets`
  directory after every build when a reliable runtime lookup, working
  directory, link, or incremental copy is available.
- Add configure-time status messages showing whether MLIR injection and the
  PPSSPP JIT replacement are active.
- Keep C++/CTest as the canonical cross-platform test interface. Use
  `tests/headless/` directly; the optional PowerShell wrapper was removed.
- Simplify the block-cache API: `PPSSPPMLIRRuntime::PrepareBlock()` currently
  calls `HasCachedBlock()` and then `MLIRJitDriver::PrepareBlock()` performs a
  second lookup.  Let the driver own the cache decision while still reporting
  cache-hit status to the runtime.

## Build integration follow-ups

- Revisit the Windows multi-config asset handling. The injected build currently
  copies the complete PPSSPP `assets` directory beside each GUI and headless
  executable after every build. Prefer a reliable runtime asset lookup or a
  configured debugger/launcher working directory; alternatively use a link or
  incremental copy so unchanged assets are not repeatedly duplicated.
