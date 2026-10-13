cmake_minimum_required(VERSION 3.20)
project(MLIRPSPHeadlessValidation NONE)
enable_testing()
find_package(Git REQUIRED)

# Configure independently: no LLVM rebuild is needed to register PRX tests.
get_filename_component(default_root "${CMAKE_CURRENT_LIST_DIR}/../../ppsspp/pspautotests" ABSOLUTE)
set(PSP_AUTOTEST_ROOT "${default_root}" CACHE PATH "pspautotests checkout")
set(MLIR_HEADLESS "" CACHE FILEPATH "MLIR-enabled PPSSPPHeadless executable")
set(NATIVE_HEADLESS "" CACHE FILEPATH "Unmodified PPSSPPHeadless executable")
set(HEADLESS_TIMEOUT 60 CACHE STRING "Emulator timeout in seconds")
set(HEADLESS_PROCESS_TIMEOUT 120 CACHE STRING "CTest wall-clock timeout in seconds")

if(NOT MLIR_HEADLESS AND NOT NATIVE_HEADLESS)
  message(FATAL_ERROR "Set MLIR_HEADLESS and/or NATIVE_HEADLESS")
endif()
file(GLOB_RECURSE expected_files CONFIGURE_DEPENDS "${PSP_AUTOTEST_ROOT}/tests/*.expected")
if(NOT expected_files)
  message(FATAL_ERROR "No expected-output fixtures under ${PSP_AUTOTEST_ROOT}/tests")
endif()
set(runnable_expected)
foreach(expected IN LISTS expected_files)
  string(REGEX REPLACE "\\.expected$" ".prx" prx "${expected}")
  if(EXISTS "${prx}")
    list(APPEND runnable_expected "${expected}")
  else()
    message(WARNING "Not runnable (PRX must be built): ${expected}")
  endif()
endforeach()
list(LENGTH runnable_expected runnable_count)
message(STATUS "Registering ${runnable_count} runnable PRXs per selected backend")
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PSP_AUTOTEST_ROOT}" ls-files
  OUTPUT_VARIABLE fixture_files RESULT_VARIABLE manifest_result)
if(NOT manifest_result EQUAL 0)
  message(FATAL_ERROR "Cannot list tracked fixtures in ${PSP_AUTOTEST_ROOT}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fixture-manifest.txt" "${fixture_files}")

foreach(backend IN ITEMS MLIR NATIVE)
  if(NOT ${backend}_HEADLESS)
    continue()
  endif()
  if(NOT EXISTS "${${backend}_HEADLESS}")
    message(FATAL_ERROR "Missing ${backend} executable: ${${backend}_HEADLESS}")
  endif()
  foreach(expected IN LISTS runnable_expected)
    string(REGEX REPLACE "\\.expected$" ".prx" prx "${expected}")
    file(RELATIVE_PATH test_name "${PSP_AUTOTEST_ROOT}/tests" "${prx}")
    string(REGEX REPLACE "\\.prx$" "" test_name "${test_name}")
    set(sandbox "${CMAKE_CURRENT_BINARY_DIR}/runs/${backend}/${test_name}")
    file(MAKE_DIRECTORY "${sandbox}")
    add_test(NAME "${backend}/${test_name}"
      COMMAND "${CMAKE_COMMAND}"
        "-DHEADLESS_EXECUTABLE=${${backend}_HEADLESS}"
        "-DFIXTURE_SOURCE=${PSP_AUTOTEST_ROOT}"
        "-DFIXTURE_MANIFEST=${CMAKE_CURRENT_BINARY_DIR}/fixture-manifest.txt"
        "-DTEST_NAME=${test_name}"
        "-DCASE_DIRECTORY=${sandbox}"
        "-DHEADLESS_TIMEOUT=${HEADLESS_TIMEOUT}"
        "-DHEADLESS_PROCESS_TIMEOUT=${HEADLESS_PROCESS_TIMEOUT}"
        -P "${CMAKE_CURRENT_LIST_DIR}/RunCase.cmake")
    math(EXPR outer_timeout "${HEADLESS_PROCESS_TIMEOUT} + 60")
    set_tests_properties("${backend}/${test_name}" PROPERTIES
      WORKING_DIRECTORY "${sandbox}"
      TIMEOUT "${outer_timeout}"
      RUN_SERIAL TRUE
      LABELS "headless;${backend}")
  endforeach()
endforeach()
