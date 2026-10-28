// Timing harness for the convolution implementations.
//
//   convolution_benchmark IMPLEMENTATION KERNEL_SIZE THREADS WARMUPS RUNS IMAGE.npy [IMAGE.npy ...]
//   convolution_benchmark --metadata
//
// IMPLEMENTATION is serial, basic or optimized. Each IMAGE.npy is an 8-bit
// grayscale dataset image written by benchmark/prepare_dataset.py; its pixels
// are scaled to [0, 1] doubles. All images must have the same shape. The kernel
// is a normalized KERNEL_SIZE x KERNEL_SIZE Gaussian blur (see npy_image.hpp).
// Every image is loaded before timing starts. The WARMUPS untimed calls use the
// first image; timed call r (counting from 0) uses image r mod (number of
// images). One runtime in milliseconds is printed per timed call, covering the
// convolution call only (including allocation of its output).

#include "convolution.hpp"
#include "npy_image.hpp"

#include <omp.h>

#include <cctype>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CONV_COMPILER_ID
#define CONV_COMPILER_ID "unknown"
#endif

#ifndef CONV_COMPILER_VERSION
#define CONV_COMPILER_VERSION "unknown"
#endif

#ifndef CONV_BUILD_TYPE
#define CONV_BUILD_TYPE "unknown"
#endif

namespace {

volatile double benchmark_sink = 0.0;

enum class Implementation { serial, basic, optimized };

struct Configuration {
    Implementation implementation;
    std::size_t kernel_size;
    int thread_count;
    int warmup_count;
    int measured_runs;
    std::vector<std::string> image_paths;
};

std::size_t parse_size(const char* text, const std::string& label) {
    if (*text == '\0') {
        throw std::invalid_argument(label + " must be a positive integer");
    }
    for (const char* character = text; *character != '\0'; ++character) {
        if (!std::isdigit(static_cast<unsigned char>(*character))) {
            throw std::invalid_argument(label + " must be a positive integer");
        }
    }
    std::size_t parsed_characters = 0;
    const unsigned long long value = std::stoull(text, &parsed_characters);
    if (text[parsed_characters] != '\0' || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(label + " must be a positive integer");
    }
    return static_cast<std::size_t>(value);
}

int parse_nonnegative_int(const char* text, const std::string& label) {
    std::size_t parsed_characters = 0;
    const long value = std::stol(text, &parsed_characters);
    if (text[parsed_characters] != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(label + " must be a non-negative integer");
    }
    return static_cast<int>(value);
}

Implementation parse_implementation(const std::string& text) {
    if (text == "serial") {
        return Implementation::serial;
    }
    if (text == "basic") {
        return Implementation::basic;
    }
    if (text == "optimized") {
        return Implementation::optimized;
    }
    throw std::invalid_argument(
        "implementation must be serial, basic, or optimized");
}

Configuration parse_configuration(int argc, char** argv) {
    if (argc < 7) {
        throw std::invalid_argument(
            "usage: convolution_benchmark IMPLEMENTATION KERNEL_SIZE THREADS "
            "WARMUPS RUNS IMAGE.npy [IMAGE.npy ...]");
    }

    Configuration configuration{
        parse_implementation(argv[1]),
        parse_size(argv[2], "kernel size"),
        parse_nonnegative_int(argv[3], "thread count"),
        parse_nonnegative_int(argv[4], "warmup count"),
        parse_nonnegative_int(argv[5], "measured runs"),
        std::vector<std::string>(argv + 6, argv + argc),
    };
    if (configuration.thread_count <= 0) {
        throw std::invalid_argument("thread count must be positive");
    }
    if (configuration.measured_runs <= 0) {
        throw std::invalid_argument("measured runs must be positive");
    }
    if (configuration.implementation == Implementation::serial &&
        configuration.thread_count != 1) {
        throw std::invalid_argument("serial benchmark requires one thread");
    }
    return configuration;
}

conv::Matrix execute(const Configuration& configuration,
                     const conv::Matrix& input,
                     const conv::Matrix& kernel) {
    switch (configuration.implementation) {
        case Implementation::serial:
            return conv::convolution_serial(input, kernel);
        case Implementation::basic:
            return conv::convolution_omp_basic(
                input, kernel, configuration.thread_count);
        case Implementation::optimized:
            return conv::convolution_omp_optimized(
                input, kernel, configuration.thread_count);
    }
    throw std::logic_error("unreachable implementation value");
}

void consume(const conv::Matrix& matrix) {
    if (matrix.values.empty()) {
        throw std::logic_error("benchmark produced an empty output");
    }
    benchmark_sink += matrix.values.front();
    benchmark_sink += matrix.values[matrix.values.size() / 2];
    benchmark_sink += matrix.values.back();
}

void print_metadata() {
    std::cout << "compiler_id=" << CONV_COMPILER_ID << '\n';
    std::cout << "compiler_version=" << CONV_COMPILER_VERSION << '\n';
    std::cout << "build_type=" << CONV_BUILD_TYPE << '\n';
#ifdef _OPENMP
    std::cout << "openmp_version=" << _OPENMP << '\n';
#else
    std::cout << "openmp_version=unavailable\n";
#endif
    std::cout << "openmp_processors=" << omp_get_num_procs() << '\n';
    std::cout << "openmp_max_threads=" << omp_get_max_threads() << '\n';
    std::cout << "input=uint8 grayscale .npy images scaled to [0, 1]\n";
    std::cout << "kernel=normalized Gaussian, sigma = 0.3 * ((K - 1) / 2 - 1) + 0.8\n";
}

void run_benchmark(const Configuration& configuration) {
    omp_set_dynamic(0);
    std::vector<conv::Matrix> images;
    images.reserve(configuration.image_paths.size());
    for (const std::string& path : configuration.image_paths) {
        images.push_back(bench::load_grayscale_npy(path));
        if (images.back().rows != images.front().rows ||
            images.back().cols != images.front().cols) {
            throw std::invalid_argument("all images must have the same shape: " + path);
        }
    }
    if (configuration.kernel_size > images.front().rows ||
        configuration.kernel_size > images.front().cols) {
        throw std::invalid_argument("kernel must not be larger than the images");
    }
    const conv::Matrix kernel = bench::gaussian_kernel(configuration.kernel_size);

    for (int run = 0; run < configuration.warmup_count; ++run) {
        consume(execute(configuration, images.front(), kernel));
    }

    std::cout << std::setprecision(17);
    for (int run = 0; run < configuration.measured_runs; ++run) {
        const conv::Matrix& input =
            images[static_cast<std::size_t>(run) % images.size()];
        const auto start = std::chrono::steady_clock::now();
        conv::Matrix output = execute(configuration, input, kernel);
        const auto stop = std::chrono::steady_clock::now();
        consume(output);
        const std::chrono::duration<double, std::milli> duration = stop - start;
        std::cout << duration.count() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--metadata") {
            print_metadata();
            return 0;
        }
        run_benchmark(parse_configuration(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "convolution_benchmark: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
