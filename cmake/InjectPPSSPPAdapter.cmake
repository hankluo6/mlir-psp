# Loaded after each project() through CMAKE_PROJECT_INCLUDE. Add the adapter
# exactly once from PPSSPP's top-level project. CMake resolves its Core/Common
# link items after the corresponding targets are declared later in configure.
if(PROJECT_NAME STREQUAL "PPSSPP" AND DEFINED MLIR_PSP_ADAPTER_SOURCE_DIR AND
		NOT TARGET ppsspp_reference_adapter)
	# The adapter is injected immediately after project(), before PPSSPP's own
	# enable_testing() call.  Enable it here so tests declared by the adapter
	# are emitted into the generated CTest files.
	enable_testing()
	# PPSSPP's native Windows build still compiles draw_text.cpp with
	# USE_SDL3_TTF enabled, but does not propagate SDL's include directory to
	# Common.  Supply our configure-only declarations without importing the
	# incompatible MSYS2 libraries into this MSVC build.
	include_directories("${CMAKE_CURRENT_LIST_DIR}/PPSSPPWindowsDependencyStubs/include")
	add_subdirectory("${MLIR_PSP_ADAPTER_SOURCE_DIR}"
		"${CMAKE_BINARY_DIR}/mlir-psp-adapter")

	# project() runs before PPSSPP creates Core, so defer the actual hook until
	# the end of the top-level directory. Only MIPS.cpp sees the redirect;
	# PPSSPP's built-in CreateNativeJit implementation remains unchanged.
	function(mlir_psp_attach_backend)
		# PPSSPP's CMake headless target adds both its SDL and native Windows
		# window hosts whenever an SDL package is visible.  Our MSVC builds use a
		# configure-only SDL target, and software headless testing needs neither
		# SDL source file.  Keep the native Windows host selected by Headless.cpp.
		if(WIN32 AND TARGET PPSSPPHeadless)
			get_target_property(headless_sources PPSSPPHeadless SOURCES)
			list(FILTER headless_sources EXCLUDE REGEX
				"(^|/)(SDLHeadlessGLGraphicsContext|SDLUtil)\\.(cpp|h)$")
			set_property(TARGET PPSSPPHeadless PROPERTY SOURCES
				"${headless_sources}")
			# CaptureDevice.cpp and Misc.cpp are shared with the GUI target but the
			# upstream CMake headless target omits their Windows system libraries.
			target_link_libraries(PPSSPPHeadless
				avrt comctl32 dinput8 dwmapi hid ksuser mfplat mfuuid setupapi uxtheme version)
		endif()

		if(MLIR_PSP_ENABLE_JIT AND TARGET Core AND
				TARGET mlir_psp_ppsspp_backend)
			set_property(SOURCE "${CMAKE_SOURCE_DIR}/Core/MIPS/MIPS.cpp"
				TARGET_DIRECTORY Core APPEND PROPERTY COMPILE_DEFINITIONS
				"CreateNativeJit=CreateMLIRJit")
			target_link_libraries(Core mlir_psp_ppsspp_backend)
			if(TARGET PPSSPPWindows)
				# Multi-config generators place the executable in Debug/Release while
				# PPSSPP generates assets at the build root. Keep a runnable asset
				# tree beside the executable so double-click and debugger launches work.
				add_custom_command(TARGET PPSSPPWindows POST_BUILD
					COMMAND "${CMAKE_COMMAND}" -E copy_directory
						"${CMAKE_BINARY_DIR}/assets"
						"$<TARGET_FILE_DIR:PPSSPPWindows>/assets"
					COMMENT "Copying PPSSPP assets beside the MLIR-enabled executable")
			endif()
		endif()
		if(TARGET PPSSPPHeadless)
			add_custom_command(TARGET PPSSPPHeadless POST_BUILD
				COMMAND "${CMAKE_COMMAND}" -E copy_directory
					"${CMAKE_BINARY_DIR}/assets"
					"$<TARGET_FILE_DIR:PPSSPPHeadless>/assets"
				COMMENT "Copying PPSSPP assets beside the headless executable")
		endif()
	endfunction()
	cmake_language(DEFER CALL mlir_psp_attach_backend)
endif()
