cmake_minimum_required(VERSION 3.20)

project(parallel_conv2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ADDRESS_SANITIZER "Build with AddressSanitizer" OFF)
option(ENABLE_UNDEFINED_SANITIZER "Build with UndefinedBehaviorSanitizer" OFF)
option(ENABLE_SANITIZER_DRIVER "Register the nested sanitizer-build test" ON)
option(ENABLE_BENCHMARK_DRIVER "Register the benchmark-pipeline smoke test" ON)

if(ENABLE_ADDRESS_SANITIZER AND ENABLE_UNDEFINED_SANITIZER)
    message(FATAL_ERROR "Enable AddressSanitizer and UBSan in separate builds")
endif()
if(ENABLE_ADDRESS_SANITIZER OR ENABLE_UNDEFINED_SANITIZER)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
        message(FATAL_ERROR "The configured compiler does not support these sanitizer flags")
    endif()
    add_compile_options(-fno-omit-frame-pointer)
endif()
if(ENABLE_ADDRESS_SANITIZER)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()
if(ENABLE_UNDEFINED_SANITIZER)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

# Apple Clang ships without an OpenMP runtime. Homebrew's keg-only libomp is
# the conventional local installation, while Linux/GCC is handled directly by
# FindOpenMP.
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    foreach(LIBOMP_PREFIX "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
        if(EXISTS "${LIBOMP_PREFIX}/include/omp.h" AND
           EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_CXX_INCLUDE_DIR "${LIBOMP_PREFIX}/include")
            break()
        endif()
    endforeach()
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(WARNING_FLAGS $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)

# --- Convolution implementations (src/) --------------------------------------

add_library(convolution_serial src/convolution_serial.cpp)
target_include_directories(convolution_serial PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(convolution_serial PRIVATE ${WARNING_FLAGS})

add_library(convolution_omp_basic src/convolution_omp_basic.cpp)
target_link_libraries(convolution_omp_basic PUBLIC convolution_serial OpenMP::OpenMP_CXX)
target_compile_options(convolution_omp_basic PRIVATE ${WARNING_FLAGS})

add_library(convolution_omp_optimized src/convolution_omp_optimized.cpp)
target_link_libraries(convolution_omp_optimized PUBLIC convolution_serial OpenMP::OpenMP_CXX)
target_compile_options(convolution_omp_optimized PRIVATE ${WARNING_FLAGS})

# Command-line tools: convolve text matrices (used by the Python tests).
add_executable(conv_serial_cli src/conv_serial_cli.cpp)
target_link_libraries(conv_serial_cli PRIVATE convolution_serial)

add_executable(conv_omp_cli src/conv_omp_cli.cpp)
target_link_libraries(conv_omp_cli PRIVATE convolution_omp_basic)

add_executable(conv_optimized_cli src/conv_optimized_cli.cpp)
target_link_libraries(conv_optimized_cli PRIVATE convolution_omp_optimized)

# --- Timing harness (benchmark/) ---------------------------------------------

add_executable(convolution_benchmark benchmark/convolution_benchmark.cpp)
target_link_libraries(convolution_benchmark PRIVATE
    convolution_omp_basic convolution_omp_optimized OpenMP::OpenMP_CXX)
target_compile_definitions(convolution_benchmark PRIVATE
    CONV_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    CONV_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    CONV_BUILD_TYPE="$<CONFIG>")
target_compile_options(convolution_benchmark PRIVATE ${WARNING_FLAGS})

# --- Tests (tests/) ----------------------------------------------------------

add_executable(test_serial_properties tests/test_serial_properties.cpp)
target_link_libraries(test_serial_properties PRIVATE convolution_serial)

add_executable(test_omp_basic tests/test_omp_basic.cpp)
target_link_libraries(test_omp_basic PRIVATE convolution_omp_basic)

add_executable(test_omp_optimized tests/test_omp_optimized.cpp)
target_link_libraries(test_omp_optimized PRIVATE
    convolution_omp_basic convolution_omp_optimized)

add_executable(test_npy_image tests/test_npy_image.cpp)
target_include_directories(test_npy_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmark)
target_link_libraries(test_npy_image PRIVATE convolution_serial)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

    # The external-reference tests need a Python with NumPy, PyTorch and
    # Pillow. Prefer the interpreter FindPython3 selected (an activated virtual
    # environment comes first); on macOS fall back to the system Python.
    set(REFERENCE_PYTHON "${Python3_EXECUTABLE}")
    execute_process(
        COMMAND "${REFERENCE_PYTHON}" -c "import numpy, torch, PIL"
        RESULT_VARIABLE REFERENCE_HAS_TORCH
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT REFERENCE_HAS_TORCH EQUAL 0 AND EXISTS "/usr/bin/python3")
        execute_process(
            COMMAND /usr/bin/python3 -c "import numpy, torch, PIL"
            RESULT_VARIABLE SYSTEM_PYTHON_HAS_TORCH
            OUTPUT_QUIET ERROR_QUIET
        )
        if(SYSTEM_PYTHON_HAS_TORCH EQUAL 0)
            set(REFERENCE_PYTHON "/usr/bin/python3")
        endif()
    endif()
    message(STATUS "Python for the reference tests: ${REFERENCE_PYTHON}")

    # Datasets are downloaded (and checksum-verified) on first use into ./data
    # unless CONV_DATA_ROOT points to an existing copy when CMake configures.
    if(DEFINED ENV{CONV_DATA_ROOT} AND NOT "$ENV{CONV_DATA_ROOT}" STREQUAL "")
        set(CONV_TEST_DATA_ROOT "$ENV{CONV_DATA_ROOT}")
    else()
        set(CONV_TEST_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/data")
    endif()
    message(STATUS "Dataset directory: ${CONV_TEST_DATA_ROOT}")

    # S1: serial reference implementation.
    add_test(NAME S1_T1_manual_known_case
        COMMAND test_serial_properties manual)
    add_test(NAME S1_T2_identity_property
        COMMAND test_serial_properties identity)
    add_test(NAME S1_T3_zero_kernel
        COMMAND test_serial_properties zero)
    add_test(NAME S1_T4_all_one_kernel
        COMMAND test_serial_properties all_one)
    add_test(NAME S1_T5_random_vs_pytorch
        COMMAND "${REFERENCE_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_external_reference.py"
            --mode random
            --conv-cli $<TARGET_FILE:conv_serial_cli>)
    add_test(NAME S1_T6_cifar10_vs_pytorch
        COMMAND "${REFERENCE_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_external_reference.py"
            --mode cifar
            --conv-cli $<TARGET_FILE:conv_serial_cli>
            --data-root "${CONV_TEST_DATA_ROOT}")

    # S2: basic OpenMP implementation.
    add_test(NAME S2_T1_serial_vs_omp_1
        COMMAND test_omp_basic cross_validation 1)
    add_test(NAME S2_T2_serial_vs_omp_2
        COMMAND test_omp_basic cross_validation 2)
    add_test(NAME S2_T3_serial_vs_omp_4
        COMMAND test_omp_basic cross_validation 4)
    add_test(NAME S2_T4_serial_vs_omp_8
        COMMAND test_omp_basic cross_validation 8)
    add_test(NAME S2_T5_cifar_cross_validation
        COMMAND "${REFERENCE_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_external_reference.py"
            --mode cifar-omp
            --conv-cli $<TARGET_FILE:conv_serial_cli>
            --omp-cli $<TARGET_FILE:conv_omp_cli>
            --data-root "${CONV_TEST_DATA_ROOT}")
    add_test(NAME S2_T6_random_dimensions
        COMMAND test_omp_basic random_dimensions)
    add_test(NAME S2_T7_repeatability
        COMMAND test_omp_basic repeatability)

    # S3: optimized OpenMP implementation and the image datasets.
    add_test(NAME S3_T1_optimized_regression
        COMMAND test_omp_optimized regression)
    add_test(NAME S3_T2_boundary_cases
        COMMAND test_omp_optimized boundary)

    function(add_dataset_test test_name dataset kernel_size)
        add_test(NAME ${test_name}
            COMMAND "${REFERENCE_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_datasets.py"
                --dataset ${dataset}
                --kernel-size ${kernel_size}
                --serial-cli $<TARGET_FILE:conv_serial_cli>
                --basic-cli $<TARGET_FILE:conv_omp_cli>
                --optimized-cli $<TARGET_FILE:conv_optimized_cli>
                --data-root "${CONV_TEST_DATA_ROOT}")
    endfunction()

    add_dataset_test(S3_T3_cifar10_100_3x3 cifar10 3)
    add_dataset_test(S3_T4_cifar100_100_3x3 cifar100 3)
    add_dataset_test(S3_T5_cifar100_100_5x5 cifar100 5)
    add_dataset_test(S3_T6_imagenet_20_3x3 imagenet 3)
    add_dataset_test(S3_T7_imagenet_20_7x7 imagenet 7)

    if(ENABLE_SANITIZER_DRIVER)
        add_test(NAME S3_T8_sanitizer_builds
            COMMAND "${REFERENCE_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_sanitizer_checks.py"
                --cmake "${CMAKE_COMMAND}"
                --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --build-root "${CMAKE_BINARY_DIR}/sanitizers"
                --generator "${CMAKE_GENERATOR}"
                --compiler "${CMAKE_CXX_COMPILER}")
    endif()

    set_tests_properties(
        S1_T5_random_vs_pytorch
        S1_T6_cifar10_vs_pytorch
        S2_T5_cifar_cross_validation
        PROPERTIES TIMEOUT 180
    )
    set_tests_properties(
        S1_T6_cifar10_vs_pytorch
        S2_T5_cifar_cross_validation
        S3_T3_cifar10_100_3x3
        PROPERTIES RESOURCE_LOCK dataset-cifar10
    )
    set_tests_properties(
        S3_T4_cifar100_100_3x3
        S3_T5_cifar100_100_5x5
        PROPERTIES RESOURCE_LOCK dataset-cifar100 TIMEOUT 600
    )
    set_tests_properties(
        S3_T6_imagenet_20_3x3
        S3_T7_imagenet_20_7x7
        PROPERTIES RESOURCE_LOCK dataset-imagenet TIMEOUT 300
    )
    if(ENABLE_SANITIZER_DRIVER)
        set_tests_properties(S3_T8_sanitizer_builds PROPERTIES TIMEOUT 600)
    endif()

    # Every S1-S3 test gates the benchmark: nothing is timed unless all pass.
    get_property(CONVOLUTION_TESTS DIRECTORY PROPERTY TESTS)
    foreach(CONVOLUTION_TEST IN LISTS CONVOLUTION_TESTS)
        if(CONVOLUTION_TEST MATCHES "^S[123]_T")
            set_property(TEST ${CONVOLUTION_TEST}
                APPEND PROPERTY LABELS correctness)
        endif()
    endforeach()

    # S4: benchmark tooling. T1 runs the whole pipeline on a tiny matrix
    # (correctness gate, dataset conversion of the ImageNet test photos, timing,
    # end-to-end pass, CSV output) inside the build tree; T2 checks the .npy
    # image reader and the Gaussian kernels used by the timing harness.
    if(ENABLE_BENCHMARK_DRIVER)
        add_test(NAME S4_T1_benchmark_pipeline_smoke
            COMMAND "${REFERENCE_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/run_experiments.py"
                --build-dir "${CMAKE_BINARY_DIR}"
                --benchmark-executable $<TARGET_FILE:convolution_benchmark>
                --ctest "${CMAKE_CTEST_COMMAND}"
                --output-dir "${CMAKE_BINARY_DIR}/benchmark-smoke"
                --data-root "${CONV_TEST_DATA_ROOT}"
                --preset smoke
                --allow-non-release)
        set_tests_properties(S4_T1_benchmark_pipeline_smoke PROPERTIES
            LABELS benchmark
            RESOURCE_LOCK dataset-imagenet
            RUN_SERIAL TRUE
            TIMEOUT 900)
    endif()
    add_test(NAME S4_T2_npy_image_reader COMMAND test_npy_image)
    set_tests_properties(S4_T2_npy_image_reader PROPERTIES LABELS benchmark)
endif()
