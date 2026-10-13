# JIT profiling

Configure either the standalone project or injected PPSSPP build with
`-DMLIR_PSP_ENABLE_PROFILING=ON`, then rebuild the executable being measured.
Default is OFF: scopes and counters are compiled out. Disable again and rebuild
after diagnosis. No PPSSPP submodule or LLVM changes are required.

`MLIR_PSP_PROFILE_EXECUTION` is independently default OFF. Enable it as well
to measure hot dispatch/execute/entry/read/write buckets. With only the main
profiling option ON, those scopes are compiled out; zero rows mean unmeasured,
not zero execution cost. The first instrumented run below predates this split
and included hot-path timers.

For LLVM legacy codegen pass detail, set `MLIR_PSP_TIME_LLVM_PASSES=1` in a
profiling-enabled process. This enables LLVM's process-wide pass timing;
the driver destructor reports and resets these timers. Use a single JIT driver
at a time for attribution. This diagnostic adds overhead and does not measure
all linker work; uncovered lookup time is not automatically linker time.
The switch has no effect when profiling is compiled out.

Each executing thread emits `MLIR_PROFILE` rows to stderr at normal thread
exit. Abnormal termination may lose the report. Counts and steady-clock elapsed
milliseconds are thread-local; multiple threads produce separate sets of rows.
No per-call file writes or shared counter locks are used.

Buckets:

- `lift`: PPSSPP frontend DoJit.
- `lower`: CompileBlockToMLIR on a cache miss (including rejected blocks).
- `compile`: CompileToExecutable, including lowering, engine creation and lookup.
- `passes`: passManager.run within compile.
- `engine_create`: ExecutionEngine::create within compile.
- `initialize`: engine initialization within compile.
- `lookup`: lookupPacked within compile; includes lazy materialization/codegen
  and linking when triggered, not just symbol-table search.
- `dispatch`: ExecuteCachedBlock, including lookup and Execute.
- `execute`: CompiledMLIRBlock::Execute, including ABI setup and entry call.
- `entry`: packed machine-code entry, including all helpers it calls.
- `read`, `write`: memory helper bodies, including callbacks when used. Failed
  accesses count too; conditional writes count when they call the write helper.

Times are **inclusive**: dispatch contains execute, execute contains entry,
and entry contains memory helpers. Do not sum these as independent costs.
Timer overhead (particularly per memory access) perturbs these measurements.
Parent-minus-child values still include instrumentation overhead and are not
precise estimates of uninstrumented costs. Float helpers are not separately
timed; entry minus memory is not purely machine-code arithmetic.

Use a copied game and fresh memstick, no `-l` or MLIR_PSP_TRACE variables.
Fixed wall-time runs can reach different guest work depending on instrumentation;
they are not equal-work benchmarks or GUI FPS measurements. Headless `--bench`
restarts the guest and must not be described as a warm-block benchmark.

## BlockOut II first instrumented run (2026-09-22)

120-second headless software-renderer run, copied game, fresh memstick, no
trace or detailed log. Report:
`build-headless-perf/blockout-7690a1798fbb4c10b1ead9d744308ac6/profile.log`.
TIMEOUT is expected for an interactive game, not a correctness pass.

| Bucket | Calls | Inclusive ms |
| --- | ---: | ---: |
| lift | 5893 | 40.094 |
| lower | 5893 | 726.793 |
| compile | 5841 | 33781.605 |
| dispatch | 239440289 | 69468.744 |
| execute | 239440289 | 57294.740 |
| entry | 239440289 | 43389.009 |
| read | 416385465 | 10234.377 |
| write | 281921903 | 6856.957 |

Compilation is a substantial cost (about 33.8 seconds); frontend/lowering is
under one second in this run. This does not establish the cause of steady-state
GUI lag. Execution buckets include hundreds of millions of timer calls, nested
helper costs, and scheduling. No native equal-work or instrumentation-overhead
comparison has yet been made. Unmeasured runtime/fallback/rendering time is not
attributed to dispatch. The initial binary printed an unused zero-counter thread;
the final header suppresses empty reports without changing measured scopes.

Profiling-enabled headless ALU, FPU and VFPU/colors regression tests passed
(3/3). After measurement the Release build was reconfigured with profiling OFF
and both GUI and headless executables were relinked. No LLVM or PPSSPP source
changes were made for this instrumentation.
The same three headless tests passed again after disabling profiling (3/3,
34.25 seconds). This is a smoke regression, not a full-suite rerun.

## Compile-only breakdown (2026-09-22)

Same 120-second BlockOut setup, hot timers compiled out, fresh copied game and
memstick. Artifact:
`build-headless-perf/compile-824fa78c49534ee89735c96cbb28cf5f/profile.log`.

5,841 successful compile attempts, 5,893 frontend/lowering attempts:

| Compile stage | Inclusive ms |
| --- | ---: |
| Compile total | 28406.211 |
| Lowering passes | 4857.741 |
| Engine creation | 2415.509 |
| Initialization | 10.748 |
| Lookup/materialization | 21023.498 |

Lookup-triggered materialization accounts for about 74% of compile time.
This includes native code generation and linking; it is not merely a symbol
lookup. Engine construction is about 8.5%, so sharing engines alone should not
be assumed to eliminate most of compilation cost. Runs are not a controlled
speedup comparison. Steady-state GUI lag remains unmeasured.

Next experiment should separate codegen from linking or compare explicit target
machine optimization settings on equal guest work. In this LLVM checkout,
`ExecutionEngine::create` creates a TargetMachine before the compile-function
callback; that callback sets the builder's opt level but uses the existing TM.
Do not assume setting only `jitCodeGenOptLevel` actually changes code generation;
verify the effective TargetMachine configuration before an A/B experiment.

## LLVM pass detail (2026-09-22)

Artifact: `build-headless-perf/passes-c807d8693b6c415b9514bac4db306dfa/profile.log`.
120-second BlockOut run, same 5,841 compiled blocks, execution scopes disabled,
`MLIR_PSP_TIME_LLVM_PASSES=1`. Compile total 44.917 s; lookup 36.417 s;
LLVM legacy pass report totals 25.8394 s wall time. Detailed timing noticeably
perturbs compilation (prior compile-only run was 28.406 s), so do not interpret
the difference as a performance regression or linker cost.

Summing report rows after removing instance suffixes (` #N`) gives approximate
wall times (individual printed rows are rounded):

- X86 DAG instruction selection: 5.760 s.
- Basic alias analysis: 4.198 s.
- Post-register-allocation scheduling: 1.723 s.
- Machine instruction scheduling: 1.574 s.
- X86 assembly printer: 0.589 s.
- Greedy register allocation: 0.497 s.

This confirms substantial backend pass work, but does not fully separate linker
time: untimed setup and timing overhead are also in lookup. Next useful controlled
experiment is explicit TargetMachine codegen-level A/B on equal guest work,
checking both compilation latency and warm execution/correctness before adopting
any lower optimization level. No optimization setting was changed in this run.

## Explicit codegen-level A/B (2026-09-22)

Profiling builds now accept `MLIR_PSP_CODEGEN_LEVEL=0|1|2|3`. The experiment
constructs a host TargetMachine with the requested CodeGenOptLevel and passes
it directly to ExecutionEngine::create. Invalid values return a compile error.
Normal profiling-OFF builds ignore this environment variable and retain the
original default. These are backend codegen levels, not LLVM IR O0/O2 pipelines.

Adapter `--benchmark --execute-mlir` prepares one MLIR block before timing
RunBlocks, then performs its normal interpreter comparison outside the timed
region. It requires the repeated block to remain at its entry PC; cache-hit
counts enforce reuse. Supply sufficient downcount via `--reg 248 2000000000`.
The warm timing includes runtime dispatch and helpers, not only generated code.

Five interleaved samples per level, each in a fresh process, no trace, hot-path
timers or LLVM pass timers. Artifacts: `build-headless-perf/codegen-ab/*.log`.
Every sample returned zero and MLIR_OK (register/PC/downcount/memory comparison).

| Workload | Level | Median compile ms | Median warm ms |
| --- | ---: | ---: | ---: |
| Increment + jump, 20 million blocks | 2 | 10.189 | 833.705 |
| Increment + jump, 20 million blocks | 0 | 6.854 | 846.381 |
| Load/increment/store + jump, 5 million blocks | 2 | 10.663 | 239.460 |
| Load/increment/store + jump, 5 million blocks | 0 | 6.719 | 247.996 |

Commands after setting the environment variable (use the Release adapter):

```text
ppsspp_reference_adapter.exe 0x08800000 0x24420001 0x0A200000 0 --reg 248 2000000000 --repeat-block 20000000 --execute-mlir --benchmark
ppsspp_reference_adapter.exe 0x08800000 0x8C820000 0x24420001 0xAC820000 0x0A200000 0 --reg 4 0x08801000 --reg 248 2000000000 --memory 0x08801000 0 --observe 0x08801000 --repeat-block 5000000 --execute-mlir --benchmark
```

Level 0 compiled these tiny blocks faster, but did not improve warm execution.
The warm differences are small and no statistical significance is claimed.
Single-block cold results include first-use initialization and do not predict
BlockOut startup savings. Both levels passed all three adapter integration tests.
Do not change the production default based on these microbenchmarks; broader
arithmetic/FP blocks and game-level measurements are still needed.
