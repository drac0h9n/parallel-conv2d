#include "convolution.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void assert_matrix_close(const conv::Matrix& actual,
                         const conv::Matrix& expected,
                         double tolerance = 1e-12) {
    if (actual.rows != expected.rows || actual.cols != expected.cols) {
        throw std::runtime_error("matrix dimensions differ");
    }
    for (std::size_t index = 0; index < actual.values.size(); ++index) {
        if (std::abs(actual.values[index] - expected.values[index]) > tolerance) {
            throw std::runtime_error("matrix values differ at flat index " +
                                     std::to_string(index));
        }
    }
}

int effective_thread_count(int requested_threads) {
    return std::max(1, std::min(requested_threads, omp_get_num_procs()));
}

std::vector<int> thread_matrix() {
    std::vector<int> threads;
    for (const int requested : {1, 2, 4, 8}) {
        const int effective = effective_thread_count(requested);
        if (threads.empty() || threads.back() != effective) {
            threads.push_back(effective);
        }
    }
    return threads;
}

conv::Matrix random_matrix(std::size_t rows, std::size_t cols,
                           std::mt19937& generator) {
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    std::vector<double> values(rows * cols);
    for (double& value : values) {
        value = distribution(generator);
    }
    return conv::Matrix(rows, cols, std::move(values));
}

void test_cross_validation(int requested_threads) {
    std::mt19937 generator(750 + requested_threads);
    const conv::Matrix input = random_matrix(37, 29, generator);
    const conv::Matrix kernel = random_matrix(5, 3, generator);
    const conv::Matrix serial = conv::convolution_serial(input, kernel);
    const int threads = effective_thread_count(requested_threads);
    assert_matrix_close(
        conv::convolution_omp_basic(input, kernel, threads), serial);
    std::cout << "requested " << requested_threads << ", used " << threads
              << " OpenMP thread(s)\n";
}

void test_random_dimensions() {
    std::mt19937 generator(750);
    const std::vector<std::pair<std::size_t, std::size_t>> input_dimensions = {
        {8, 8}, {17, 23}, {32, 32}, {64, 79}, {128, 128},
    };
    const std::vector<std::size_t> kernel_sizes = {1, 2, 3, 5, 7};

    for (const auto& dimensions : input_dimensions) {
        const conv::Matrix input =
            random_matrix(dimensions.first, dimensions.second, generator);
        for (const std::size_t kernel_size : kernel_sizes) {
            const conv::Matrix kernel =
                random_matrix(kernel_size, kernel_size, generator);
            const conv::Matrix serial = conv::convolution_serial(input, kernel);
            for (const int threads : thread_matrix()) {
                assert_matrix_close(
                    conv::convolution_omp_basic(input, kernel, threads), serial);
            }
        }
    }
}

void test_repeatability() {
    std::mt19937 generator(750);
    const conv::Matrix input = random_matrix(64, 64, generator);
    const conv::Matrix kernel = random_matrix(3, 3, generator);
    const conv::Matrix expected = conv::convolution_serial(input, kernel);
    const int threads = effective_thread_count(8);

    for (int repetition = 0; repetition < 100; ++repetition) {
        assert_matrix_close(
            conv::convolution_omp_basic(input, kernel, threads), expected);
    }
    std::cout << "100 identical executions with " << threads
              << " OpenMP thread(s)\n";
}

int parse_positive_integer(const char* argument) {
    std::size_t parsed_characters = 0;
    const int value = std::stoi(argument, &parsed_characters);
    if (argument[parsed_characters] != '\0' || value <= 0) {
        throw std::invalid_argument("expected a positive integer");
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        omp_set_dynamic(0);
        if (argc == 3 && std::string(argv[1]) == "cross_validation") {
            test_cross_validation(parse_positive_integer(argv[2]));
        } else if (argc == 2 && std::string(argv[1]) == "random_dimensions") {
            test_random_dimensions();
        } else if (argc == 2 && std::string(argv[1]) == "repeatability") {
            test_repeatability();
        } else {
            std::cerr << "usage: test_omp_basic "
                         "cross_validation THREADS|random_dimensions|repeatability\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASS: " << argv[1] << '\n';
    return 0;
}
