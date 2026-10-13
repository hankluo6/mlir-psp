# PPSSPP's top-level configure currently requires SDL3 even for its native
# Windows target, which does not link SDL.  The MLIR integration runner only
# builds that Windows target's Core/Common libraries, so an interface target is
# sufficient and avoids mixing MinGW SDL binaries into an MSVC build.
if(NOT TARGET SDL3::SDL3)
	add_library(SDL3::SDL3 INTERFACE IMPORTED)
	set_target_properties(SDL3::SDL3 PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
endif()
set(SDL3_FOUND TRUE)
