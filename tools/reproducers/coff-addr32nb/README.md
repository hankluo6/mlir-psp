# COFF ADDR32NB RuntimeDyld reproducer

Run `tools\reproducers\coff-addr32nb\run.cmd` from a Windows command prompt. It loads
the MSVC x64 developer environment, builds the COFF object and harness, checks
that `ADDR32NB` occurs, and asserts the failure/pass pair. It needs the LLVM
Release build at `D:/llvm-project/llvm-build-release` with `llvm-readobj`.
The path can be changed at the top of the script.

For manual inspection, run from `D:/mlir-psp` in an MSVC x64 developer
environment. Build LLVM's `llvm-readobj` and `llvm-rtdyld` with `--parallel 1`
if absent.

```powershell
cl /nologo /c /Od /GS- /Fo:build-coff-addr32nb-repro.obj tools/reproducers/coff-addr32nb/coff-addr32nb-repro.c
D:/llvm-project/llvm-build-release/bin/llvm-readobj.exe --relocations build-coff-addr32nb-repro.obj
cmake -S tools/reproducers/coff-addr32nb -B build-coff-addr32nb-msvc -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=D:/llvm-project/llvm-build-release/lib/cmake/llvm
cmake --build build-coff-addr32nb-msvc --parallel 1
./build-coff-addr32nb-msvc/coff-addr32nb-harness.exe 0 build-coff-addr32nb-repro.obj
./build-coff-addr32nb-msvc/coff-addr32nb-harness.exe 1 build-coff-addr32nb-repro.obj
```

The object contains three `IMAGE_REL_AMD64_ADDR32NB` relocations in `.pdata`.
The harness mapper asks `VirtualAlloc` for fixed addresses more than 4 GiB
apart and ignores near-allocation hints. Mode `0` should abort with the
ordered-section-layout error; mode `1` should print `reserved layout passed`.
The harness uses the checkout's `SectionMemoryManager` and `RuntimeDyld`, but
does not instantiate MLIR `ExecutionEngine`; it tests the exact boolean's
effect on the memory manager. A fixed address may be unavailable if occupied
by another mapping, in which case the test fails with an allocation error.
