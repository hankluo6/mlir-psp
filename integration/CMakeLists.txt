option(MLIR_PSP_ENABLE_JIT "Build the PPSSPP adapter with the MLIR JIT" OFF)
option(MLIR_PSP_ENABLE_PROFILING "Instrument inclusive JIT timings (perturbs execution)" OFF)
option(MLIR_PSP_PROFILE_EXECUTION "Also instrument hot execution paths" OFF)
if(MLIR_PSP_ENABLE_JIT)
	find_package(MLIR CONFIG REQUIRED)
	add_library(mlir_psp_ppsspp_backend STATIC
		${CMAKE_CURRENT_LIST_DIR}/../src/MLIRBlockCompiler.cpp
		${CMAKE_CURRENT_LIST_DIR}/../src/MLIRJitEngine.cpp
		${CMAKE_CURRENT_LIST_DIR}/../src/MLIRJitDriver.cpp
		${CMAKE_CURRENT_LIST_DIR}/../integration/PPSSPPMemoryAccess.cpp
		${CMAKE_CURRENT_LIST_DIR}/../integration/PPSSPPMLIRJit.cpp
		${CMAKE_CURRENT_LIST_DIR}/../integration/PPSSPPMLIRSelector.cpp
		${CMAKE_CURRENT_LIST_DIR}/../integration/PPSSPPMLIRRuntime.cpp
	)
	set_target_properties(mlir_psp_ppsspp_backend PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
	)
	target_include_directories(mlir_psp_ppsspp_backend PUBLIC
		${CMAKE_CURRENT_LIST_DIR}/../include
		${CMAKE_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/Common
		${MLIR_INCLUDE_DIRS}
		${LLVM_INCLUDE_DIRS}
	)
	target_compile_definitions(mlir_psp_ppsspp_backend PUBLIC
		MLIR_PSP_ADAPTER_ENABLE_JIT=1
	)
	if(MLIR_PSP_ENABLE_PROFILING)
		target_compile_definitions(mlir_psp_ppsspp_backend PRIVATE MLIR_PSP_ENABLE_PROFILING=1)
		if(MLIR_PSP_PROFILE_EXECUTION)
			target_compile_definitions(mlir_psp_ppsspp_backend PRIVATE MLIR_PSP_PROFILE_EXECUTION=1)
		endif()
	endif()
	target_link_libraries(mlir_psp_ppsspp_backend PUBLIC
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
endif()
