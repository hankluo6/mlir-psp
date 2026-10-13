@echo off
setlocal
set "ROOT=%~dp0..\..\.."
set "LLVM_BUILD=D:\llvm-project\llvm-build-release"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" (
  echo Missing MSVC x64 developer environment: "%VCVARS%"
  exit /b 1
)
call "%VCVARS%" >nul || exit /b 1

cl /nologo /c /Od /GS- /Fo:"%ROOT%\build-coff-addr32nb-repro.obj" "%~dp0coff-addr32nb-repro.c" || exit /b 1
"%LLVM_BUILD%\bin\llvm-readobj.exe" --relocations "%ROOT%\build-coff-addr32nb-repro.obj" > "%ROOT%\build-coff-addr32nb-relocations.txt" || exit /b 1
findstr /c:"IMAGE_REL_AMD64_ADDR32NB" "%ROOT%\build-coff-addr32nb-relocations.txt" >nul || (
  echo The object has no IMAGE_REL_AMD64_ADDR32NB relocation.
  exit /b 1
)

cmake -S "%~dp0." -B "%ROOT%\build-coff-addr32nb-msvc" -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="%LLVM_BUILD%\lib\cmake\llvm" || exit /b 1
cmake --build "%ROOT%\build-coff-addr32nb-msvc" --parallel 1 || exit /b 1

"%ROOT%\build-coff-addr32nb-msvc\coff-addr32nb-harness.exe" 0 "%ROOT%\build-coff-addr32nb-repro.obj" > "%ROOT%\build-coff-addr32nb-before.txt" 2>&1
findstr /c:"IMAGE_REL_AMD64_ADDR32NB relocation requires an ordered section layout" "%ROOT%\build-coff-addr32nb-before.txt" >nul || (
  echo FAIL: separate layout failed for a different reason.
  type "%ROOT%\build-coff-addr32nb-before.txt"
  exit /b 1
)

"%ROOT%\build-coff-addr32nb-msvc\coff-addr32nb-harness.exe" 1 "%ROOT%\build-coff-addr32nb-repro.obj" || exit /b 1
echo PASS: ADDR32NB reproduced with separate layout and resolved with reserved layout.
