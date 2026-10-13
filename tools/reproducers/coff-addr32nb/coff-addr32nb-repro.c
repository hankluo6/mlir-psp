// Compile with MSVC x64: cl /nologo /c /Od /GS- /Fo:coff-addr32nb-repro.obj coff-addr32nb-repro.c
// A non-leaf function emits .pdata with IMAGE_REL_AMD64_ADDR32NB relocations.
__declspec(noinline) int repro_callee(int x) { return x + 1; }
__declspec(dllexport) int repro_entry(int x) { return repro_callee(x) + 2; }
