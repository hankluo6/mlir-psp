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

## Build (Windows x64)

Requirements: Git, CMake, Visual Studio 2022 with Desktop development with C++,
a Windows SDK, and an MSVC x64 Release build of LLVM with MLIR enabled.
The tested LLVM revision is `e4e8ea4c913bbf253cbab2dbcb29d3500a5689ac`
(`LLVM_ENABLE_PROJECTS=mlir`, `LLVM_TARGETS_TO_BUILD=Native`). LLVM/MLIR C++ APIs
change; other revisions are not guaranteed to build. Do not mix MSYS2 LLVM
libraries with this MSVC configuration, or Debug and Release runtime libraries.

### LLVM prerequisite

The tested Windows build uses a local change in
`mlir/lib/ExecutionEngine/ExecutionEngine.cpp`: enable reserved section
allocation for x86-64 COFF as well as AArch64:

```cpp
const llvm::Triple &allocationTriple = llvmModule->getTargetTriple();
bool reserveAlloc = allocationTriple.isAArch64() ||
    (allocationTriple.getArch() == llvm::Triple::x86_64 &&
     allocationTriple.isOSBinFormatCOFF());
```

Replace the existing `reserveAlloc` initialization at the tested revision,
then rebuild LLVM/MLIR before linking this project. This change is **not**
applied automatically. Without it, the tested LLVM build can fail on Windows
with an `IMAGE_REL_AMD64_ADDR32NB` relocation error.

### Configure and build PPSSPP

Run these commands in PowerShell. Set `$llvmBuild` to your LLVM/MLIR Release
build directory containing `lib/cmake/mlir` and `lib/cmake/llvm`.

```powershell
git clone --recurse-submodules https://github.com/hankluo6/mlir-psp.git
cd mlir-psp
# For an existing checkout:
git submodule update --init --recursive

$projectRoot = (Get-Location).Path
$llvmBuild = "D:/llvm-project/llvm-build-release"

cmake -S ppsspp -B build-ppsspp-mlir-release -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_PROJECT_INCLUDE=$projectRoot/cmake/InjectPPSSPPAdapter.cmake" `
  "-DMLIR_PSP_ADAPTER_SOURCE_DIR=$projectRoot/adapter" `
  -DMLIR_PSP_ENABLE_JIT=ON `
  "-DMLIR_DIR=$llvmBuild/lib/cmake/mlir" `
  "-DLLVM_DIR=$llvmBuild/lib/cmake/llvm" `
  "-DSDL3_DIR=$projectRoot/cmake/PPSSPPWindowsDependencyStubs" `
  "-DSDL3_ttf_DIR=$projectRoot/cmake/PPSSPPWindowsDependencyStubs"

cmake --build build-ppsspp-mlir-release --config Release --target PPSSPPWindows --parallel 1
```

The SDL entries above are configure-only stubs for this native Windows build,
not a general replacement for SDL. Assets are copied beside the executable by
the build; keep that assets directory when running PPSSPP.

## Run

```powershell
.\build-ppsspp-mlir-release\Release\PPSSPPWindows.exe
# Or open a homebrew/game directly:
.\build-ppsspp-mlir-release\Release\PPSSPPWindows.exe "C:/path/to/EBOOT.PBP"
```

Select a JIT CPU core in PPSSPP's settings. This build redirects both native
JIT selections to the MLIR backend; the interpreter selection does not use it.
Game files are not included.

## Test

Build and run the runtime integration tests:

```powershell
cmake --build build-ppsspp-mlir-release --config Release --target ppsspp_reference_adapter --parallel 1
ctest --test-dir build-ppsspp-mlir-release -C Release -R mlir_psp_ppsspp --output-on-failure
```

For instruction-level tests, configure the root project using the same LLVM
build and the adapter just built (in the same PowerShell session):

```powershell
cmake -S . -B build-tests-release -G "Visual Studio 17 2022" -A x64 `
  "-DMLIR_DIR=$llvmBuild/lib/cmake/mlir" `
  "-DLLVM_DIR=$llvmBuild/lib/cmake/llvm" `
  "-DMLIR_PSP_REFERENCE_ADAPTER=$projectRoot/build-ppsspp-mlir-release/mlir-psp-adapter/Release/ppsspp_reference_adapter.exe"
cmake --build build-tests-release --config Release --target mlir_psp_test --parallel 1
ctest --test-dir build-tests-release -C Release --output-on-failure
```

For full PSP program tests, build the `PPSSPPHeadless` target and follow the
[headless validation guide](tests/headless/README.md). See
[tests/README.md](tests/README.md) for the test layout.
