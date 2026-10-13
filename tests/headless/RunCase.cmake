cmake_minimum_required(VERSION 3.20)

foreach(required HEADLESS_EXECUTABLE FIXTURE_SOURCE FIXTURE_MANIFEST TEST_NAME
    CASE_DIRECTORY HEADLESS_TIMEOUT HEADLESS_PROCESS_TIMEOUT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing ${required}")
  endif()
endforeach()
if(IS_ABSOLUTE "${TEST_NAME}" OR TEST_NAME MATCHES "(^|/)\\.\\.(/|$)")
  message(FATAL_ERROR "Test name must stay within the fixture root")
endif()

# Both host0 and ms0 are writable from guest code. A separate memstick alone
# does not isolate tests such as io/file/file, which appends to host0/workdir.
# Only tracked files are copied, excluding debris left by earlier test runs.
get_filename_component(case_root "${CASE_DIRECTORY}" REALPATH)
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef attempt_id)
set(attempt "${case_root}/attempt-${attempt_id}")
if(EXISTS "${attempt}")
  message(FATAL_ERROR "Refusing to reuse an existing test attempt: ${attempt}")
endif()
file(MAKE_DIRECTORY "${attempt}/host0" "${attempt}/memstick")
file(STRINGS "${FIXTURE_MANIFEST}" fixture_files)
foreach(relative IN LISTS fixture_files)
  if(IS_ABSOLUTE "${relative}" OR relative MATCHES "(^|/)\\.\\.(/|$)")
    message(FATAL_ERROR "Invalid fixture manifest entry: ${relative}")
  endif()
  get_filename_component(parent "${relative}" DIRECTORY)
  file(COPY "${FIXTURE_SOURCE}/${relative}"
    DESTINATION "${attempt}/host0/${parent}")
endforeach()

execute_process(COMMAND "${HEADLESS_EXECUTABLE}"
  --root "${attempt}/host0" --memstick "${attempt}/memstick"
  --compare "--timeout=${HEADLESS_TIMEOUT}" --graphics=software
  "${attempt}/host0/tests/${TEST_NAME}.prx"
  WORKING_DIRECTORY "${attempt}"
  TIMEOUT "${HEADLESS_PROCESS_TIMEOUT}"
  RESULT_VARIABLE exit_status OUTPUT_VARIABLE output ERROR_VARIABLE errors)
# Normalize only the generated root path in reports, never guest results.
string(REPLACE "${attempt}/host0/" "" output "${output}")
string(REPLACE "${attempt}/host0/" "" errors "${errors}")
file(WRITE "${case_root}/output-${attempt_id}.log" "${output}${errors}")
message("${output}${errors}")

# Remove only the disposable directory created above, not the source checkout
# or case directory. Validate its resolved parent before recursive cleanup.
get_filename_component(resolved_attempt "${attempt}" REALPATH)
get_filename_component(resolved_parent "${resolved_attempt}" DIRECTORY)
if(NOT resolved_parent STREQUAL case_root OR
   NOT resolved_attempt STREQUAL attempt)
  message(FATAL_ERROR "Unexpected test attempt path; refusing cleanup: ${attempt}")
endif()
file(REMOVE_RECURSE "${attempt}")
if(NOT exit_status STREQUAL "0")
  message(FATAL_ERROR "Headless test ${TEST_NAME} failed: ${exit_status}")
endif()
