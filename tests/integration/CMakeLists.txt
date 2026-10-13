add_executable(ppsspp_reference_adapter
	PPSSPPReferenceAdapter.cpp
	${CMAKE_SOURCE_DIR}/Windows/CaptureDevice.cpp
)
target_include_directories(ppsspp_reference_adapter PRIVATE
	${CMAKE_CURRENT_LIST_DIR}/../../include
	${CMAKE_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/Common
)
target_compile_definitions(ppsspp_reference_adapter PRIVATE UNICODE _UNICODE)
target_link_libraries(ppsspp_reference_adapter PRIVATE
	Core Common d2d1 dwrite d3d11 mfplat mfuuid
)
if(MSVC)
	# PPSSPP applies this globally later in its top-level CMakeLists for its
	# prebuilt FFmpeg libraries.  This injected target is created before that
	# call, so mirror the option here to keep the Debug DLL runtime authoritative.
	target_link_options(ppsspp_reference_adapter PRIVATE /NODEFAULTLIB:libcmt.lib)
endif()

if(MLIR_PSP_ENABLE_JIT)
	target_link_libraries(ppsspp_reference_adapter PRIVATE
		mlir_psp_ppsspp_backend
	)

	file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/cases/mlir_psp_ppsspp_memory_integration.args" case_lines)
	set(case_args)
	foreach(line IN LISTS case_lines)
		string(REGEX REPLACE "#.*$" "" line "${line}")
		separate_arguments(line_args UNIX_COMMAND "${line}")
		list(APPEND case_args ${line_args})
	endforeach()
	add_test(NAME mlir_psp_ppsspp_memory_integration
		COMMAND $<TARGET_FILE:ppsspp_reference_adapter> ${case_args})
	set_tests_properties(mlir_psp_ppsspp_memory_integration PROPERTIES
		PASS_REGULAR_EXPRESSION "MLIR_OK"
	)

	file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/cases/mlir_psp_ppsspp_fallback_integration.args" case_lines)
	set(case_args)
	foreach(line IN LISTS case_lines)
		string(REGEX REPLACE "#.*$" "" line "${line}")
		separate_arguments(line_args UNIX_COMMAND "${line}")
		list(APPEND case_args ${line_args})
	endforeach()
	add_test(NAME mlir_psp_ppsspp_fallback_integration
		COMMAND $<TARGET_FILE:ppsspp_reference_adapter> ${case_args})
	set_tests_properties(mlir_psp_ppsspp_fallback_integration PROPERTIES
		PASS_REGULAR_EXPRESSION "MLIR_FALLBACK_OK"
	)

	file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/cases/mlir_psp_ppsspp_multiblock_dispatch.args" case_lines)
	set(case_args)
	foreach(line IN LISTS case_lines)
		string(REGEX REPLACE "#.*$" "" line "${line}")
		separate_arguments(line_args UNIX_COMMAND "${line}")
		list(APPEND case_args ${line_args})
	endforeach()
	add_test(NAME mlir_psp_ppsspp_multiblock_dispatch
		COMMAND $<TARGET_FILE:ppsspp_reference_adapter> ${case_args})
	set_tests_properties(mlir_psp_ppsspp_multiblock_dispatch PROPERTIES
		PASS_REGULAR_EXPRESSION "MLIR_OK"
	)
endif()
