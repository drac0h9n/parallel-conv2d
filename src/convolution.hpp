#pragma once

#include <cstddef>
#include <vector>

namespace conv {

struct Matrix {
    std::size_t rows{};
    std::size_t cols{};
    std::vector<double> values;

    Matrix() = default;
    Matrix(std::size_t row_count, std::size_t column_count);
    Matrix(std::size_t row_count, std::size_t column_count,
           std::vector<double> elements);

    // Defined inline in the header so that every implementation translation
    // unit can inline element access. Keeping the body in a single .cpp file
    // forces an out-of-line call plus a bounds check for each element access,
    // which measurably penalises implementations compiled in other files.
    double& operator()(std::size_t row, std::size_t col) {
        return values.at(row * cols + col);
    }
    const double& operator()(std::size_t row, std::size_t col) const {
        return values.at(row * cols + col);
    }
};

// Computes a two-dimensional, unit-stride, valid cross-correlation. This is
// the operation commonly called "convolution" by neural-network libraries.
Matrix convolution_serial(const Matrix& input, const Matrix& kernel);

// Basic OpenMP implementation. Output rows are divided among thread_count
// threads; each output element retains the serial accumulation order.
Matrix convolution_omp_basic(const Matrix& input, const Matrix& kernel,
                             int thread_count);

// Optimized OpenMP implementation. The two-dimensional output space is
// distributed with a static schedule, while contiguous storage is accessed
// directly inside each output window.
Matrix convolution_omp_optimized(const Matrix& input, const Matrix& kernel,
                                 int thread_count);

}  // namespace conv
