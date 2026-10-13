# Validation notes

Layout cleanup (2026-10-13): the optional PowerShell wrapper was removed.
Use `tests/headless/` directly. References to the wrapper below are historical.

## 2026-09-30: Cave Story save/load manual validation

The user reports successful save/load, repeated loads, switching between save
states, gameplay after loading, restart and switching games. These are manual
reports, not automated GUI assertions or an all-games compatibility claim.
The JIT bridge now clears the runtime cache only for `PointerWrap::MODE_READ`,
not when measuring/writing a save. Release GUI and headless builds succeeded;
the headless CPU ALU regression passed. Loading still takes longer but gameplay
recovers normally; cache destruction versus recompilation time is not yet measured.

## 2026-09-22: Codegen-level microbenchmark A/B

Added profiling-build-only explicit TargetMachine level selection and adapter
`--benchmark` (precompile before timing, require cached MLIR execution, retain
interpreter comparison). Five samples each at codegen levels 0 and 2 for
20-million-block increment and 5-million-block memory-update loops all passed.
Level 0 reduced single-block compile medians but did not improve warm execution;
production defaults are unchanged. Measurements/reproduction: [profiling.md](profiling.md).
Both levels passed 3/3 adapter integration tests. Restored profiling OFF and
rebuilt adapter, GUI and headless; 3/3 adapter tests plus benchmark smoke and
fallback-rejection checks passed. No full-suite rerun or game FPS claim.

## 2026-09-22: Initial BlockOut profiling

Follow-up compile-only profiling (hot timers compiled out) measured 28.406 s
total compile: 21.023 s lookup/materialization, 4.858 s passes, 2.416 s engine
creation, 0.011 s initialization. All 5,841 compiled blocks were measured.
This localizes cold cost to materialization (code generation plus linking),
not ordinary symbol lookup. The same ALU/FPU/VFPU-colors smoke tests passed
3/3 with this profiling configuration. See profiling.md for artifact paths.

Added default-OFF `MLIR_PSP_ENABLE_PROFILING` in standalone and integrated
builds. A 120-second BlockOut headless run measured 33.782 seconds in
CompileToExecutable for 5,841 blocks, versus 0.767 seconds total in frontend
and CompileBlockToMLIR. This establishes significant cold compilation cost,
not the cause of steady-state GUI lag. Nested high-frequency execution timers
perturb execution; do not add inclusive buckets or interpret them as FPS.
See [profiling.md](profiling.md) for full measurements, artifact path and caveats.
Profiling-enabled ALU/FPU/VFPU-colors tests passed (3/3); Release GUI/headless
were subsequently rebuilt with profiling OFF for normal use.

## 2026-09-15: VFPU asin precision

Built `PPSSPPHeadless` in `build-ppsspp-mlir-release` (Release) and the
standalone test runner in `build` (Debug), with one build job.

- `cpu/vfpu/vector`: the additional MLIR `vasin_s/p/t/q` differences disappeared
  after using PPSSPP's `vfpu_asin` in the integrated backend. The remaining
  printed differences also occur in the native backend: `vavg`, `vfad`,
  `vcrsp`, `vdot`, `vhdp`, and `vdet`. This is still a failing expected-output
  test, not a pass.
- `cpu/vfpu/vregs`: both backends fail, with matching visible Upgrade/Combine
  differences and missing output. Cause not yet established; it must not be
  classified as an upstream bug solely from this observation.
- MLIR `cpu/fpu/fcr` and `cpu/vfpu/prefixes`: passed.
- Standalone Debug CTest `mlir_block_compiler`: passed.

No expected output or tolerance was changed. These checks do not constitute
a complete rerun of the PRX suite. GUI and Debug PPSSPP executables were not
rebuilt by this validation.

Remaining work: investigate shared native/MLIR failures against the matching
PRX source and PSP semantics, and continue the full suite. Prior reports that
Native passed `cpu/vfpu/vector` were incorrect.

### CPU suite follow-up

All 15 CPU PRXs with matching `.expected` files were run:

- MLIR Release: 13 passed, 2 failed (`vector`, `vregs`).
- Native Release: 11 passed, 4 failed (`fcr`, `prefixes`, `vector`, `vregs`).
- Logs: `build-ppsspp-mlir-release/cpu-validation.log` and
  `build-ppsspp-native/cpu-validation.log`.
- Added `tests/cases/mips_vfpu_asin.mipstest`, checking the fallback's
  `asin(0.1f)` rounding via `mtv`/`vasin.s`/`mfv`. Debug CTest passed again.

`vregs` also fails in the native pure interpreter (`-i`). Its source and
expected output include `Reuse` and `TestSwizzle2`, but those literal strings
are absent from the supplied PRX (while `Upgrade` and `Combine` are present).
This indicates a stale test binary; rebuild it with PSPSDK before treating
its missing output as an emulator regression. No `psp-g++` is on PATH and
the standard `C:/pspsdk` and `D:/pspsdk` locations were absent.

The shared vector precision differences need separate investigation:
`InterpreterVFPU.cpp` sets `USE_VFPU_DOT = false`, so merely switching to
the interpreter does not enable the existing accurate dot-product paths.

### Wider suite / test infrastructure

Native Release completed all 430 supplied PRXs with matching expected files:
316 passed, 114 failed, 0 missing in that selected list. Full log:
`build-ppsspp-native/full-validation.log`. This includes GPU, audio, HLE,
filesystem and video failures; it is not solely a CPU backend test suite.

The initial MLIR batch was interrupted and is **not** a complete result.
Its first 34 completed test sections matched Native output exactly. It was
replaced with per-PRX CTest execution using `tests/headless/CMakeLists.txt`,
which gives each case a process timeout and separate memory stick. Because
execution isolation differs, any newly differing case needs a matching
isolated Native rerun before assigning blame to MLIR.

The CTest entry point was smoke-tested with Native `cpu/cpu_alu/cpu_alu`
and MLIR `cpu/fpu/fcr` (both passed). Configuration also reports three
`.expected` files without matching PRXs: `kirk/kirk`,
`threads/semaphores/semaphore_greater_than_zero/semaphore_greater_than_zero`,
and `umd/raw_access/raw_acess`. These are not counted as passes.

## 2026-09-16: isolated continuation

Resumed MLIR CTest at test 30 (tests 1-29 completed previously: 27 passed,
2 failed). A separate Native CTest build avoids simultaneous writers to the
same CTest report directory. Native isolated result: 317 passed, 113 failed
out of 430; report: `build-headless-native-validation/native-results.xml`.
Unlike the earlier batch run, Native `utility/savedata/filelist` passed.

Code investigation of the shared VFPU precision failures:
`IRFrontend::Comp_VDot` emits ordinary `FMul`/`FAdd` for smaller vectors,
while four-lane inputs can use `Vec4Dot`. Therefore replacing just our
`Vec4Dot` lowering cannot fix the pair-vector failures; changing every
floating-point add/multiply would also affect scalar FPU instructions.
A correct precision fix needs to preserve the guest dot-product operation
through the frontend, not merely change generic floating-point rounding.

### Fixed host0 test contamination

`io/file/file` produced different Native/MLIR output despite matching CPU
behavior: the PRX creates `workdir/testfile` relative to `host0`, opens it
without truncation, and later appends. Earlier runs left an ignored file in
the source checkout, so its size increased on subsequent runs. Separate
memory sticks alone did not prevent this.

`tests/headless/RunCase.cmake` now gives every invocation fresh host0 and
memstick directories, copying only tracked fixture files from the working
tree. It saves output, bounds the child process lifetime, and removes only
the disposable attempt directory it created. Existing source debris was
preserved, not deleted.

Isolated `io/file/file` now has identical Native and MLIR output. It still
fails expected comparison due to shared file metadata/rename behavior; this
is not reported as a pass. The ongoing broad run predates this isolation fix,
so its host0-sensitive results require clean reruns before drawing conclusions.

### Completed broad comparison

The resumed MLIR run finished: 401 tests, 292 passed, 109 failed. Together
with the verified first segment (29 tests, 27 passed, 2 failed), the result
is **319 passed, 111 failed out of 430**. Native: **317 passed, 113 failed**.
There were no tests that passed Native but failed MLIR in this run.
The resumed report is `build-headless-validation/mlir-resumed-results.xml`;
the earlier segment was recorded in `Testing/Temporary/LastTest.log.tmp06703`.

Comparison of the 401 resumed outputs found only four differing cases:

- `cpu/fpu/fcr`, `cpu/vfpu/prefixes`: MLIR passed; Native failed.
- `io/file/file`: shared host0 contamination, fixed in the runner. After the
  fix, two Native and two MLIR runs produced byte-identical output logs
  (SHA256 `06E6073918B5C1E2F2FA32EF215F4EE33D6664ABD7A89B5FAE048B81983F2F32`).
  The expected-output test remains failed for shared metadata/rename behavior.
- `threads/threads/profiler`: differences are checkpoint elapsed-time fields.
  The test enables `CHECKPOINT_ENABLE_TIME`; `common.c` prints differences of
  `sceKernelGetSystemTimeWide()`. A Native `--ir` rerun produced the same
  checkpoint values as MLIR, including the initial 413 vs native-JIT 414.
  Both return the same profiler register values. This test remains failed.

These broad totals predate the host0 isolation change; they are **not** a
claim that all 430 were rerun through the final runner. The final runner was
checked with both backends on ALU (passed), file I/O and profiler (failed as
described), repeated file I/O, and a forced 1 ms process timeout (correctly
failed and cleaned up its disposable directory). No attempt directories or
headless processes remained after these checks.

`RunHeadlessValidation.ps1` now delegates to this CMake/CTest runner rather
than directly executing against the source fixtures. Its Native ALU smoke
test passed. Standalone Debug `mlir_block_compiler` CTest also passed again.
No expected output or tolerance was relaxed, and no PPSSPP submodule source
was changed. The remaining 111 failures are still failures, not an all-pass
result; shared VFPU precision, stale/missing PRXs and other PPSSPP subsystem
behavior require separate work.

### Post-COFF full regression (started 2026-09-21; pending)

Running 430 isolated cases per backend in `build-headless-post-coff-mlir`
and `build-headless-post-coff-native`, with `--output-junit results.xml`.
Do not interpret earlier full-suite totals as results of this run.
Compare matching names for Native-pass / MLIR-fail regressions; shared failures
remain deferred. Three fixtures still lack PRXs and are not counted as passes.
No MLIR_PSP environment variables were set in the test environment.

Executable SHA256 fingerprints at start:
- MLIR: `9B8CD443DBB5CE64F3CF9CC4123BDFBD9B21577BCA0EC2C9FB32484DC9AE88D4`
- Native: `851F07CED2489D608553ED680C2673D4B9D1916CEA8A987F9420F40003D63891`

Existing standalone Debug CTest also passed (1/1, 11.08 seconds); it was not
rebuilt and is not evidence of coverage of the patched Release LLVM library.

### Windows COFF JIT allocation fix (2026-09-21)

LLDB loaded `PPSSPPWindows.exe.39140.dmp` using the matching executable and
Windows system DLLs. The abort caller's error string is
`IMAGE_REL_AMD64_ADDR32NB relocation requires an ordered section layout`.
Export-only symbol names with large offsets are not reliable function names.
This is a RuntimeDyld relocation failure, not evidence of a graphics-driver bug.

With user approval, changed the external LLVM checkout at
`D:/llvm-project/mlir/lib/ExecutionEngine/ExecutionEngine.cpp`: enable
SectionMemoryManager's existing reserve-allocation mode for x86_64 COFF as
well as AArch64. It allocates code/RO/RW sections within one allocation instead
of relying on best-effort nearby allocations. Other targets are unchanged.
This is a local LLVM source change, not an upstream fix or a PPSSPP modification.
Preserve/reapply it when replacing the LLVM checkout/build.

Rebuilt `MLIRExecutionEngine` in `llvm-build-release`, then relinked both
Release `PPSSPPWindows` and `PPSSPPHeadless` successfully with one build worker.
GUI BlockOut crash reproduction still requires validation with these binaries;
successful builds alone do not establish that the crash is fixed.
Post-relink headless regressions passed: `cpu/cpu_alu/cpu_alu`, `cpu/fpu/fpu`,
and `cpu/vfpu/colors` (3/3, 40.54 seconds total). The full suite was not rerun.

#### Independent relocation reproduction and review (2026-09-21)

Added `tools/reproducers/coff-addr32nb/coff-addr32nb-repro.c` and `tools/reproducers/coff-addr32nb/` as a small
reproducer independent of PPSSPP. MSVC x64 compiled the C file to COFF.
`tools/reproducers/coff-addr32nb/run.cmd` runs the full reproduction from a fresh object;
it passed on 2026-09-21. It requires no PPSSPP executable or game files.
`llvm-readobj --relocations` confirmed **three**
`IMAGE_REL_AMD64_ADDR32NB` entries in `.pdata` (offsets 0, 4, 8), plus one
`REL32` in `.text$mn`. This specifically tests the relocation named by the
dump, not only JIT startup.

Built `llvm-readobj` and `llvm-rtdyld` with `--parallel 1`. Using
`llvm-rtdyld -verify -triple=x86_64-pc-windows-msvc` with target address
`0x10000000`, section separation 0 succeeded; section separation
`0x100000000` aborted with the exact ordered-section-layout message. This
independently confirms the layout sensitivity.

Built the standalone MSVC C++ harness with `--parallel 1`. Its mapper
ignores near hints and allocates each independent mapped region at fixed
addresses separated by more than 4 GiB. With `SectionMemoryManager`'s
`ReserveAlloc=false`, `RuntimeDyld::resolveRelocations()` aborted with the
same error (Windows exit `-1073740791`). With `ReserveAlloc=true`, the same
object and mapper succeeded (`reserved layout passed`, exit 0) and
`finalizeMemory()` succeeded. This is a deterministic before/after test of
the precise memory-manager switch used by the MLIR patch; the harness does
not itself create an MLIR `ExecutionEngine`.

Source review: RuntimeDyld invokes `reserveAllocationSpace()` before loading
sections when `needsToReserveAllocationSpace()` is true. The manager reserves
one block, partitions page-rounded code, RO and RW areas, and services their
section allocations from it. The pre-existing `finalizeMemory()` changes
code to RX and RO to R while RW remains writable; its destructor owns and
releases the combined block once via `CodeMem`. The harness exercised
finalization and destructor without error. `.pdata` is collected by the
x86-64 COFF RuntimeDyld unwind path; the patch does not bypass this path.
Actual Windows stack unwinding through generated code was **not** tested.

Limits: this only guarantees nearby sections when the summed reservation
fits the `ADDR32NB` 32-bit span and `reserveAllocationSpace()` succeeds.
That function silently returns on allocation failure, after which ordinary
section allocations may again be distant. Its page-boundary assertions also
assume section alignment no greater than a page; the MSVC test object uses
ordinary alignments. A custom `sectionMemoryMapper` must honor a single
contiguous reservation and support per-page permissions. These are existing
limitations of `SectionMemoryManager`, not changes introduced by this patch.

Upstream check on 2026-09-21: LLVM `main` still has AArch64-only
`reserveAlloc` in MLIR `ExecutionEngine.cpp`, so this specific change is not
upstream. Issue https://github.com/llvm/llvm-project/issues/55386 describes
the same fatal message and `.pdata` layout sensitivity; it is closed with no
linked PR shown on the issue page. It is related evidence, not proof of the
same application root cause. An upstream patch should keep the condition
limited to x86-64 COFF, retain AArch64 behavior, and include an MLIR/RuntimeDyld
regression test with an adversarial memory mapper and verified `ADDR32NB`
object. Do not claim this fixes GUI stutter or all RuntimeDyld relocations.

Conclusion: the fix is technically well supported for x86-64 COFF and is a
reasonable upstream candidate after adding an in-tree regression test and
checking reviewer preference for the scope/location of the fix. The reported
GUI improvement is field evidence, not a controlled before/after GUI test;
the old dump must not be symbolized against the newly linked executable.

### BlockOut II extended smoke test (2026-09-18)

Ran the existing MLIR headless Release executable with software graphics,
logging, a copied game directory and separate memstick for 180 seconds.
Artifacts: `build-headless-blockout/080df7cbf62c4c4f8513898d2df6d98f/long.log`
and `long.trace`. The earlier 30-second run had reached `sounds/wozz.wav`;
the extended run reached menu textures and sustained GE display-list submission
and synchronization. No `compile_error` or `execute_error` appeared.

The hot block at `0881eb84` had exactly 6,119,715 executions in both runs:
it finished rather than spinning throughout the extended run. The extended
summary reports 5,841 compiled blocks, 52 fallback blocks and 100,406,678
MLIR executions. This supports slow initialization, not a permanent stall
in that loop. The timeout is intentional for this interactive game and is
not an automated pass. GUI pixels, input and gameplay remain unverified;
headless progress does not prove that the reported GUI black screen is fixed.

### Fresh-host0 full rerun (completed; reviewed 2026-09-17)

Both full suites completed with the final isolation runner. Reports are
`build-headless-clean-mlir/results.xml` and
`build-headless-clean-native/results.xml` (completed 2026-09-16).
Generated `build-headless-*` directories are ignored by Git.

Compared all 430 matching test names and their JUnit failure elements:

- Both passed: 317.
- Native passed, MLIR failed: **0**.
- MLIR passed, Native failed: 2 (`cpu/fpu/fcr`, `cpu/vfpu/prefixes`).
- Both failed: 111.

Totals: MLIR 319 passed / 111 failed; Native 317 passed / 113 failed.
These are now verified fresh-host0 results, not the earlier split-run totals.
They apply to the tested executables, not a newly rebuilt version: MLIR
headless executable last modified 2026-09-15, Native 2026-09-05.
Standalone Debug CTest was rerun on 2026-09-17: `mlir_block_compiler` passed
(1/1, 10.69 seconds total).

Per the user's scope, investigate only Native-pass / MLIR-fail cases.
There are none in this suite. Shared failures remain recorded but are deferred;
this is not an all-tests-pass or all-games-compatible claim. The frontend
extension idea below is deferred, not an approved or established required fix.

Checked for PSPSDK in Windows PATH and the usual installation directories,
then in the installed Ubuntu-22.04 WSL distribution (PATH, `/usr/local/pspdev`,
`/opt/pspdev`); no `psp-gcc`/`psp-g++` was found. No SDK was installed.

Frontend extension constraint: operation methods such as `Comp_VDot` are
virtual, but `IRFrontend`'s IR writer, prefix helpers and JIT state are private.
An external subclass cannot emit a prefix-correct accurate dot operation with
the existing API. A small upstream extension interface, maintained as a patch
outside the submodule, needs agreement before implementation. No frontend
source or generic multiply/add semantics have been changed.
