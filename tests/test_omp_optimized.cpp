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

void test_regression() {
    std::mt19937 generator(750);
    const std::vector<std::pair<std::size_t, std::size_t>> input_dimensions = {
        {8, 8}, {17, 23}, {37, 29}, {64, 79}, {128, 128},
    };
    const std::vector<std::pair<std::size_t, std::size_t>> kernel_dimensions = {
        {1, 1}, {2, 3}, {3, 3}, {5, 3}, {7, 7},
    };

    for (const auto& input_size : input_dimensions) {
        const conv::Matrix input =
            random_matrix(input_size.first, input_size.second, generator);
        for (const auto& kernel_size : kernel_dimensions) {
            const conv::Matrix kernel =
                random_matrix(kernel_size.first, kernel_size.second, generator);
            const conv::Matrix serial =
                conv::convolution_serial(input, kernel);
            for (const int threads : thread_matrix()) {
                const conv::Matrix basic =
                    conv::convolution_omp_basic(input, kernel, threads);
                const conv::Matrix optimized =
                    conv::convolution_omp_optimized(input, kernel, threads);
                assert_matrix_close(basic, serial);
                assert_matrix_close(optimized, serial);
                assert_matrix_close(optimized, basic);
            }
        }
    }
}

void expect_invalid_argument(void (*operation)(), const std::string& label) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(label + " did not throw std::invalid_argument");
}

void serial_larger_kernel() {
    const conv::Matrix input(3, 3);
    const conv::Matrix kernel(4, 3);
    static_cast<void>(conv::convolution_serial(input, kernel));
}

void basic_larger_kernel() {
    const conv::Matrix input(3, 3);
    const conv::Matrix kernel(3, 4);
    static_cast<void>(conv::convolution_omp_basic(input, kernel, 2));
}

void optimized_larger_kernel() {
    const conv::Matrix input(3, 3);
    const conv::Matrix kernel(4, 4);
    static_cast<void>(conv::convolution_omp_optimized(input, kernel, 2));
}

void optimized_invalid_threads() {
    const conv::Matrix input(3, 3);
    const conv::Matrix kernel(1, 1);
    static_cast<void>(conv::convolution_omp_optimized(input, kernel, 0));
}

void test_boundary_cases() {
    std::mt19937 generator(751);
    for (const auto& dimensions :
         std::vector<std::pair<std::size_t, std::size_t>>{{3, 3}, {5, 7}}) {
        const conv::Matrix input =
            random_matrix(dimensions.first, dimensions.second, generator);
        const conv::Matrix kernel =
            random_matrix(dimensions.first, dimensions.second, generator);
        const conv::Matrix serial = conv::convolution_serial(input, kernel);
        if (serial.rows != 1 || serial.cols != 1) {
            throw std::runtime_error("kernel=input did not produce a 1x1 output");
        }
        for (const int threads : thread_matrix()) {
            assert_matrix_close(
                conv::convolution_omp_basic(input, kernel, threads), serial);
            assert_matrix_close(
                conv::convolution_omp_optimized(input, kernel, threads),
                serial);
        }
    }

    expect_invalid_argument(serial_larger_kernel, "serial larger kernel");
    expect_invalid_argument(basic_larger_kernel, "basic larger kernel");
    expect_invalid_argument(optimized_larger_kernel,
                            "optimized larger kernel");
    expect_invalid_argument(optimized_invalid_threads,
                            "optimized invalid thread count");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        omp_set_dynamic(0);
        if (argc == 2 && std::string(argv[1]) == "regression") {
            test_regression();
        } else if (argc == 2 && std::string(argv[1]) == "boundary") {
            test_boundary_cases();
        } else {
            std::cerr << "usage: test_omp_optimized regression|boundary\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASS: " << argv[1] << '\n';
    return 0;
}
