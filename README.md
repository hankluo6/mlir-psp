# mlir-psp

Experimental MLIR/LLVM CPU backend for PPSSPP. PSP MIPS instructions are
translated by PPSSPP's IR frontend, lowered through MLIR to native code, and
executed inside the PPSSPP runtime. Unsupported blocks can fall back to
PPSSPP's IR interpreter.

This is a side project, not a replacement for upstream PPSSPP or a claim of
compatibility with all PSP games. Windows x64 is the primary tested platform.

## Get the sources

Clone this repository with `git clone --recurse-submodules <repository-url>`.
For an existing checkout, run:

```sh
git submodule update --init --recursive
```

PPSSPP is pinned as a submodule. Backend code lives outside it; CMake injects
the integration without editing the submodule sources.

## Build and test

Building requires LLVM/MLIR development libraries and CMake. Configure
`MLIR_DIR` and `LLVM_DIR` for your local LLVM installation.
Match the compiler, architecture and Debug/Release ABI of your LLVM build.

Windows x64 builds used a local LLVM COFF allocation fix: see the
[validation notes](docs/validation-notes.md) and
[standalone reproducer](tools/reproducers/coff-addr32nb/README.md). This external LLVM
modification is not included automatically by cloning this repository.

- [Headless validation](tests/headless/README.md): cross-platform CMake/CTest
  runner comparing native and MLIR backend results.
- `tests/unit/cases/`: MIPS instruction regression cases.
- `tests/integration/`: reference execution and integration-test executable, not a
  requirement to launch the GUI.
- [Validation history](docs/validation-notes.md): results and limitations.
- [Profiling](docs/profiling.md): optional instrumentation and measurements.

## Source layout

| Directory | Purpose |
| --- | --- |
| `src/` | IR lowering, native compilation and compiled-block cache |
| `include/mlir_psp/` | Backend interfaces and state/memory ABI |
| `integration/` | PPSSPP JIT bridge, runtime dispatch and memory callbacks |
| `cmake/` | Build integration and Windows dependency configuration |
| `adapter/` | Compatibility CMake entry point |
| `tests/integration/` | PPSSPP-backed test harness |
| `tests/` | Regression cases, runners and diagnostic reproducers |
| `ppsspp/` | Upstream submodule |

See [plan.md](plan.md) for the development plan and cleanup backlog.
Builds, dumps, editor settings and local agent configuration are excluded
from Git. Game downloads, memory sticks and save states should not be committed.

## Licensing

PPSSPP and its dependencies retain their respective licenses; see the
submodule's license files. A license for this project's original code still
needs to be selected by its author before redistribution terms are finalized.
