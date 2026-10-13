cmake_minimum_required(VERSION 3.20)
project(mlir_psp LANGUAGES CXX)
option(MLIR_PSP_ENABLE_PROFILING "Instrument inclusive JIT timings (perturbs execution)" OFF)
option(MLIR_PSP_PROFILE_EXECUTION "Also instrument hot execution paths" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR CONFIG REQUIRED)

add_library(mlir_psp
	src/MLIRBlockCompiler.cpp
	src/MLIRJitEngine.cpp
	src/MLIRJitDriver.cpp
)
if(MLIR_PSP_ENABLE_PROFILING)
	target_compile_definitions(mlir_psp PRIVATE MLIR_PSP_ENABLE_PROFILING=1)
	if(MLIR_PSP_PROFILE_EXECUTION)
		target_compile_definitions(mlir_psp PRIVATE MLIR_PSP_PROFILE_EXECUTION=1)
	endif()
endif()
target_include_directories(mlir_psp
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/ppsspp
		${MLIR_INCLUDE_DIRS}
		${LLVM_INCLUDE_DIRS}
)
target_link_libraries(mlir_psp
	PUBLIC
		MLIRIR
		MLIRFuncDialect
		MLIRArithDialect
		MLIRControlFlowDialect
		MLIRMemRefDialect
		MLIRArithToLLVM
		MLIRControlFlowToLLVM
		MLIRFuncToLLVM
		MLIRMemRefToLLVM
		MLIRReconcileUnrealizedCasts
		MLIRExecutionEngine
		MLIRBuiltinToLLVMIRTranslation
		MLIRLLVMToLLVMIRTranslation
)

enable_testing()
add_executable(mlir_psp_test
	tests/unit/MIPSTestCaseLoader.cpp
	tests/unit/MLIRBlockCompilerTest.cpp
)
target_include_directories(mlir_psp_test PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ppsspp
	${MLIR_INCLUDE_DIRS}
	${LLVM_INCLUDE_DIRS}
)
target_link_libraries(mlir_psp_test PRIVATE mlir_psp)
set(MLIR_PSP_REFERENCE_ADAPTER
	"${CMAKE_CURRENT_SOURCE_DIR}/build-ppsspp-mlir/mlir-psp-adapter/$<CONFIG>/ppsspp_reference_adapter.exe"
	CACHE STRING "Path to the PPSSPP reference adapter executable")
add_test(NAME mlir_block_compiler COMMAND mlir_psp_test
	${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/cases --no-dump
	${MLIR_PSP_REFERENCE_ADAPTER})
set_tests_properties(mlir_block_compiler PROPERTIES
	ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:C:/msys64/clang64/bin")
