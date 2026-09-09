cmake_minimum_required(VERSION 3.25...4.2)
project(acpd_filterreg_cpp VERSION 0.3.0 LANGUAGES CXX)
option(ACPD_BUILD_PYTHON "Build the nanobind extension" ON)
option(ACPD_INSTALL_CORE "Install the reusable C++ core and its headers" OFF)
option(ACPD_BUILD_TESTS "Build native tests and the test driver" OFF)
option(ACPD_BUILD_BENCH "Build the ablation/profiling benchmark binary" OFF)
option(ACPD_ENABLE_CUDA "Build the CUDA Gaussian-sum E-step operator" OFF)
set(ACPD_CUDA_ARCHITECTURES "61" CACHE STRING "CUDA architectures for the E-step kernel")
set(ACPD_CUDA_HOST_COMPILER "" CACHE STRING "Host compiler for nvcc; needed when the C++ compiler is newer than the toolkit supports")
option(ACPD_USE_SYSTEM_EIGEN "Use Eigen3 from the build environment" OFF)
add_library(acpd_core STATIC src/types.cpp src/lattice.cpp src/fgt.cpp src/cuda.cpp src/gaussian.cpp src/rigid.cpp src/analytic.cpp src/registration.cpp)
target_compile_features(acpd_core PUBLIC cxx_std_17)
target_include_directories(acpd_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
if(ACPD_USE_SYSTEM_EIGEN)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  target_link_libraries(acpd_core PUBLIC Eigen3::Eigen)
else()
  target_include_directories(acpd_core SYSTEM PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen> $<INSTALL_INTERFACE:include/acpd/third_party/eigen>)
  # The vendored Eigen is the unmodified 3.3.90 snapshot taken from the attached FilterReg tree.
  # Its Transpositions.h calls trt.derived() on a type without that member; upstream Eigen fixed
  # this in 3.4.0. GCC >= 15 diagnoses uninstantiated template bodies at parse time, turning the
  # latent defect into a hard error. Keep the third-party headers byte-identical and restore the
  # pre-GCC-15 diagnostic behaviour instead. Set ACPD_USE_SYSTEM_EIGEN=ON to use Eigen >= 3.4.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    target_compile_options(acpd_core PUBLIC -Wno-template-body)
  endif()
  # Apple Clang 21 eagerly diagnoses an unused dependent template body in the
  # byte-identical Eigen 3.3.90 snapshot.  Delayed parsing preserves the
  # vendored source and matches the behavior of older Clang/GCC toolchains;
  # users needing a fully current Eigen can still set ACPD_USE_SYSTEM_EIGEN=ON.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(acpd_core PUBLIC -fdelayed-template-parsing)
  endif()
endif()
target_compile_definitions(acpd_core PUBLIC EIGEN_MPL2_ONLY)
set_target_properties(acpd_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(acpd_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(acpd_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(ACPD_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_native src/bindings.cpp)
  target_link_libraries(_native PRIVATE acpd_core)
  install(TARGETS _native LIBRARY DESTINATION acpd_filterreg_cpp RUNTIME DESTINATION acpd_filterreg_cpp)
endif()
if(ACPD_BUILD_TESTS)
  enable_testing()
  add_executable(acpd_core_tests tests/core_tests.cpp)
  target_link_libraries(acpd_core_tests PRIVATE acpd_core)
  add_test(NAME acpd_core_tests COMMAND acpd_core_tests)
  add_executable(acpd_test_driver tests/driver.cpp)
  target_link_libraries(acpd_test_driver PRIVATE acpd_core)
endif()

if(ACPD_ENABLE_CUDA)
  # The kernel is a separate translation unit with a plain C ABI, so nvcc may use a
  # different (older) host compiler than the rest of the core without ABI risk.
  if(ACPD_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${ACPD_CUDA_HOST_COMPILER}")
  endif()
  enable_language(CUDA)
  # nvcc reports its host compiler's library directories. The C++ objects are built
  # and linked by the project's own compiler, so pulling in the host compiler's
  # libstdc++ mixes two glibc versions. The .cu unit exposes a plain C ABI and needs
  # only cudart, so those directories and the static CUDA runtime are dropped.
  set(_acpd_cuda_link_dirs "")
  foreach(_dir IN LISTS CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES)
    if(NOT _dir MATCHES "/lib/gcc/")
      list(APPEND _acpd_cuda_link_dirs "${_dir}")
    endif()
  endforeach()
  set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${_acpd_cuda_link_dirs}")
  set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "")
  add_library(acpd_cuda STATIC cuda/gaussian_transform.cu)
  set_target_properties(acpd_cuda PROPERTIES CUDA_ARCHITECTURES "${ACPD_CUDA_ARCHITECTURES}"
                                             POSITION_INDEPENDENT_CODE ON
                                             CUDA_RUNTIME_LIBRARY Shared)
  target_compile_options(acpd_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
  target_link_libraries(acpd_core PUBLIC acpd_cuda)
  target_compile_definitions(acpd_core PUBLIC ACPD_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(acpd_cuda PUBLIC CUDA::cudart)
endif()

if(ACPD_BUILD_BENCH)
  add_executable(acpd_bench bench/bench.cpp)
  target_link_libraries(acpd_bench PRIVATE acpd_core)
endif()

if(ACPD_INSTALL_CORE)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(TARGETS acpd_core EXPORT acpdTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/acpd DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if(NOT ACPD_USE_SYSTEM_EIGEN)
    install(DIRECTORY third_party/eigen/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/acpd/third_party/eigen)
  endif()
  install(EXPORT acpdTargets NAMESPACE acpd:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/acpd)
  configure_package_config_file(cmake/acpdConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/acpdConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/acpd)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/acpdConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/acpdConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/acpdConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/acpd)
endif()
