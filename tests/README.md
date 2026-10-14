# Tests

- `unit/`: small MIPS block regression cases, loader and comparison runner.
  These are differential tests and require the reference adapter, not isolated
  dependency-free unit tests. Build `mlir_psp_test` from the repository root.
- `integration/`: PPSSPP reference adapter and runtime integration tests.
  Build `ppsspp_reference_adapter` in the injected PPSSPP build, then run
  `ctest --test-dir <ppsspp-build> -C Release -R mlir_psp_ppsspp --output-on-failure`.
  `cases/*.args` contains adapter command-line arguments; `#` starts a comment.
- `headless/`: full PSP program validation; see [the runner guide](headless/README.md).

Production backend targets are defined in `integration/CMakeLists.txt` at the
repository root. The root `adapter/CMakeLists.txt` is only a compatibility entry
point for existing build configurations and executable paths.
