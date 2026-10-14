# mlir-psp

An experimental MLIR/LLVM CPU backend for PPSSPP.

The backend translates PPSSPP's MIPS intermediate representation into MLIR,
lowers it to LLVM IR, and JIT-compiles native code for execution inside PPSSPP.
Compiled blocks are cached, while unsupported blocks can fall back to
PPSSPP's IR interpreter.

PPSSPP continues to handle graphics, audio, input, memory and PSP system
services. It is included as a submodule; the MLIR backend and integration code
live separately from the upstream sources.

The project includes instruction-level differential tests, runtime integration
tests and a headless validation runner. Windows x64 is the primary tested
platform; compatibility with all PSP games is not guaranteed.
