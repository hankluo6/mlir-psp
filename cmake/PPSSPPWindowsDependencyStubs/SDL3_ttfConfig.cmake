# See SDL3Config.cmake in this directory.  Native Windows PPSSPP does not use
# SDL_ttf, but the top-level configure requires a target to exist.
if(NOT TARGET SDL3_ttf::SDL3_ttf)
	add_library(SDL3_ttf::SDL3_ttf INTERFACE IMPORTED)
	set_target_properties(SDL3_ttf::SDL3_ttf PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
endif()
set(SDL3_ttf_FOUND TRUE)
