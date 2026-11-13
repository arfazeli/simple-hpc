cmake_minimum_required(VERSION 3.20)
project(simple_hpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SIMPLE_OPENMP "Enable OpenMP for the independent kernels" ON)
option(SIMPLE_HIP "Build the HIP GPU backend (T23; needs ROCm)" OFF)
option(SIMPLE_MPI "Build with MPI (row-slab domain decomposition)" OFF)

execute_process(COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE SIMPLE_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET RESULT_VARIABLE git_rc)
if(NOT git_rc EQUAL 0 OR SIMPLE_GIT_HASH STREQUAL "")
  set(SIMPLE_GIT_HASH "unknown")
endif()

file(GLOB CORE_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(FILTER CORE_SRC EXCLUDE REGEX "/main\\.cpp$")
add_library(simple_core STATIC ${CORE_SRC})
target_include_directories(simple_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(simple_core PUBLIC SIMPLE_GIT_HASH="${SIMPLE_GIT_HASH}"
  SIMPLE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
# CXX only: the HIP sources (SIMPLE_HIP) get their own warning flags below.
target_compile_options(simple_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>)
# No FMA contraction, so results are reproducible across compilers and the snapshot tests can
# compare bit for bit.
target_compile_options(simple_core PUBLIC -ffp-contract=off)

if(SIMPLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(simple_core PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(STATUS "OpenMP not found: building serial")
  endif()
endif()

add_executable(simple src/main.cpp)
target_link_libraries(simple PRIVATE simple_core)

if(SIMPLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(simple_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(simple_core PUBLIC SIMPLE_MPI)
endif()

if(SIMPLE_HIP)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx942) # MI300A
  endif()
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  add_library(simple_hip STATIC ${CMAKE_SOURCE_DIR}/src/hip/solver_hip.hip)
  target_link_libraries(simple_hip PUBLIC simple_core hip::host)
  # ponytail: no -Werror on HIP yet, so a new compiler warning costs no cluster round trip.
  target_compile_options(simple_hip PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Wall -Wextra>)
  target_link_libraries(simple PRIVATE simple_hip)
  target_compile_definitions(simple PRIVATE SIMPLE_HIP)
  set_target_properties(simple PROPERTIES LINKER_LANGUAGE CXX)
endif()

include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4)
FetchContent_MakeAvailable(Catch2)

enable_testing()
file(GLOB TEST_SRC CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/tests/unit/*.cpp ${CMAKE_SOURCE_DIR}/tests/regression/*.cpp)
add_executable(simple_tests ${TEST_SRC})
target_link_libraries(simple_tests PRIVATE simple_core Catch2::Catch2WithMain)
target_include_directories(simple_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_definitions(simple_tests PRIVATE SIMPLE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
include(Catch)
# Momentum and p' residuals are fixed-order row sums (same bits for any thread count), but the cg
# dot products are OpenMP reductions, so tests run with one thread by default (test_threads.cpp
# sets its own counts).
catch_discover_tests(simple_tests PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1 SKIP_RETURN_CODE 4)

# Ghia validation runs: slow, labelled "ghia" (ctest -LE ghia skips them). Each test runs the
# solver on a copy of the config whose output_dir points into the build tree, then checks the
# centrelines against the published data.
# Pass -DPython3_EXECUTABLE=... to pick an interpreter; one without numpy skips the tests.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                  RESULT_VARIABLE numpy_missing OUTPUT_QUIET ERROR_QUIET)
  if(numpy_missing)
    message(STATUS "Ghia tests skipped: ${Python3_EXECUTABLE} has no numpy")
  endif()
endif()
if(Python3_FOUND AND NOT numpy_missing)
  # Extra arguments KEY=VALUE override keys of the config copy (like run_bench.py --set).
  function(add_ghia_test name re max_u max_v)
    set(out ${CMAKE_BINARY_DIR}/ghia/${name})
    file(READ ${CMAKE_SOURCE_DIR}/configs/validation/re${re}_80.cfg cfg_text)
    string(REGEX REPLACE "output_dir *= *[^\n]*" "output_dir = ${out}" cfg_text "${cfg_text}")
    foreach(kv IN LISTS ARGN)
      string(REGEX MATCH "^([^=]+)=(.*)$" _kv "${kv}")
      string(REGEX REPLACE "\n${CMAKE_MATCH_1} *= *[^\n]*" "\n${CMAKE_MATCH_1} = ${CMAKE_MATCH_2}"
             cfg_text "${cfg_text}")
    endforeach()
    file(WRITE ${CMAKE_BINARY_DIR}/ghia/${name}.cfg "${cfg_text}")
    add_test(NAME ${name}
      COMMAND sh -c
        "\"$<TARGET_FILE:simple>\" \"${CMAKE_BINARY_DIR}/ghia/${name}.cfg\" > /dev/null && \"${Python3_EXECUTABLE}\" \"${CMAKE_SOURCE_DIR}/postprocess/validate_ghia.py\" \"${out}\" --check --max-rms-u ${max_u} --max-rms-v ${max_v}")
    set_tests_properties(${name} PROPERTIES LABELS ghia TIMEOUT 3600)
  endfunction()
  add_ghia_test(ghia_re400_80 400 0.010 0.010)
  add_ghia_test(ghia_re1000_80 1000 0.025 0.015)
  add_ghia_test(ghia_re400_80_multicolour 400 0.010 0.010
                momentum_solver=gs_multicolour pressure_solver=gs_redblack)
  add_ghia_test(ghia_re1000_80_multicolour 1000 0.025 0.015
                momentum_solver=gs_multicolour pressure_solver=gs_redblack)
  add_ghia_test(ghia_re400_80_cg 400 0.010 0.010 pressure_solver=cg)
  add_ghia_test(ghia_re1000_80_cg 1000 0.025 0.015 pressure_solver=cg)
  add_ghia_test(ghia_re400_80_mg 400 0.010 0.010
                momentum_solver=gs_multicolour pressure_solver=mg)
  add_ghia_test(ghia_re1000_80_mg 1000 0.025 0.015
                momentum_solver=gs_multicolour pressure_solver=mg)
endif()

if(SIMPLE_MPI)
  # u, v, p on N ranks must be byte-identical to one rank (tests/mpi/test_mpi_slab.cpp).
  add_executable(simple_mpi_tests ${CMAKE_SOURCE_DIR}/tests/mpi/test_mpi_slab.cpp)
  target_link_libraries(simple_mpi_tests PRIVATE simple_core)
  foreach(n 1 2 3 4)
    add_test(NAME mpi_slab_n${n}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:simple_mpi_tests> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(mpi_slab_n${n} PROPERTIES LABELS mpi ENVIRONMENT OMP_NUM_THREADS=1
                         PROCESSORS ${n})
  endforeach()
endif()

if(SIMPLE_HIP)
  add_executable(simple_hip_tests ${CMAKE_SOURCE_DIR}/tests/hip/test_hip.cpp)
  target_link_libraries(simple_hip_tests PRIVATE simple_hip Catch2::Catch2WithMain)
  target_include_directories(simple_hip_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_compile_definitions(simple_hip_tests PRIVATE SIMPLE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(simple_hip_tests PROPERTIES LINKER_LANGUAGE CXX)
  catch_discover_tests(simple_hip_tests PROPERTIES LABELS hip ENVIRONMENT OMP_NUM_THREADS=1)
endif()
