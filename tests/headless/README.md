# Headless PRX validation

This CMake/CTest entry point runs one PRX per process, on any platform where
PPSSPPHeadless is built. It does not build PPSSPP or require PowerShell.

Configure with absolute executable paths (either backend can be omitted):

```text
cmake -S tests/headless -B build-headless-validation -DMLIR_HEADLESS=/absolute/path/to/mlir/PPSSPPHeadless -DNATIVE_HEADLESS=/absolute/path/to/native/PPSSPPHeadless
ctest --test-dir build-headless-validation -C Release --output-on-failure
```

On Windows include `.exe` in the executable paths. To run a subset:

```text
ctest --test-dir build-headless-validation -C Release -R "^MLIR/cpu/" --output-on-failure
ctest --test-dir build-headless-validation -C Release --rerun-failed --output-on-failure
```

Tests run serially. Each invocation copies the tracked fixture files into a
fresh writable `host0` and creates a fresh memory stick under the validation
build directory. This requires Git; modifications to tracked source files are
included, but untracked debris from previous test runs is not. Add any new
fixture to Git's index before configuring the suite.

`HEADLESS_PROCESS_TIMEOUT` (default 120 seconds) bounds each emulator process
even if it does not return to its internal timeout check. CTest adds 60 seconds
for fixture setup and cleanup. `HEADLESS_TIMEOUT` defaults to 60 seconds.
Both can be changed with CMake `-D` arguments. A timeout is a failure, not a pass.

Only PRXs with `.expected` files are registered. Missing PRXs produce explicit
configuration warnings; they have not been tested. A native failure is still
a failure and does not automatically excuse an MLIR failure. CTest retains
output in `Testing/Temporary/LastTest.log`; `--output-junit results.xml` can
also produce a machine-readable report. Per-attempt emulator output is retained
under `runs/<backend>/<test>/output-*.log`. Disposable fixture/memory-stick
copies are removed after the emulator exits, including ordinary failures and
process timeouts; an externally interrupted wrapper can leave an attempt
directory behind. Existing source fixtures and prior logs are never removed.

The supplied `cpu/vfpu/vregs.prx` appears older than its source/expected output;
see `../../docs/validation-notes.md`. Do not regenerate expected output from the
emulator to hide these failures.
