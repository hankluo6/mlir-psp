# Compatibility entry point for existing MLIR_PSP_ADAPTER_SOURCE_DIR caches.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../integration" "${CMAKE_CURRENT_BINARY_DIR}/backend")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../tests/integration" "${CMAKE_CURRENT_BINARY_DIR}/tests")
# Keep existing debugger and test executable paths working.
set_target_properties(ppsspp_reference_adapter PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
