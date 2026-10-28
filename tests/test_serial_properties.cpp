#include "convolution.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
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

void test_manual() {
    const conv::Matrix input(4, 4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    });
    const conv::Matrix kernel(2, 2, {1, 0, 0, -1});
    const conv::Matrix expected(3, 3, {
        -5, -5, -5,
        -5, -5, -5,
        -5, -5, -5,
    });
    assert_matrix_close(conv::convolution_serial(input, kernel), expected);
}

void test_identity() {
    const conv::Matrix input(3, 5, {
        -2.5, 0, 1, 2, 3,
        4, 5.25, 6, 7, 8,
        9, 10, 11, 12, 13,
    });
    const conv::Matrix identity(1, 1, {1});
    assert_matrix_close(conv::convolution_serial(input, identity), input);
}

void test_zero() {
    const conv::Matrix input(3, 4, {
        1, -2, 3, 4,
        5, 6, -7, 8,
        9, 10, 11, -12,
    });
    const conv::Matrix zero_kernel(2, 3, {0, 0, 0, 0, 0, 0});
    const conv::Matrix expected(2, 2, {0, 0, 0, 0});
    assert_matrix_close(conv::convolution_serial(input, zero_kernel), expected);
}

void test_all_one() {
    const conv::Matrix input(4, 4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    });
    const conv::Matrix all_one_kernel(3, 3, {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    });
    const conv::Matrix expected(2, 2, {54, 63, 90, 99});
    assert_matrix_close(conv::convolution_serial(input, all_one_kernel), expected);

    const conv::Matrix all_one_input(5, 5, std::vector<double>(25, 1.0));
    const conv::Matrix all_nines(3, 3, std::vector<double>(9, 9.0));
    assert_matrix_close(
        conv::convolution_serial(all_one_input, all_one_kernel), all_nines);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_serial_properties CASE\n";
        return 2;
    }

    try {
        const std::string test_case = argv[1];
        if (test_case == "manual") {
            test_manual();
        } else if (test_case == "identity") {
            test_identity();
        } else if (test_case == "zero") {
            test_zero();
        } else if (test_case == "all_one") {
            test_all_one();
        } else {
            throw std::invalid_argument("unknown test case: " + test_case);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASS: " << argv[1] << '\n';
    return 0;
}
