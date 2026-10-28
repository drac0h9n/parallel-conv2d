#include "convolution.hpp"

#include <stdexcept>
#include <utility>

namespace conv {

namespace {

std::size_t checked_size(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("matrix dimensions must be non-zero");
    }
    if (cols > static_cast<std::size_t>(-1) / rows) {
        throw std::length_error("matrix dimensions are too large");
    }
    return rows * cols;
}

}  // namespace

Matrix::Matrix(std::size_t row_count, std::size_t column_count)
    : rows(row_count),
      cols(column_count),
      values(checked_size(row_count, column_count), 0.0) {}

Matrix::Matrix(std::size_t row_count, std::size_t column_count,
               std::vector<double> elements)
    : rows(row_count), cols(column_count), values(std::move(elements)) {
    if (values.size() != checked_size(rows, cols)) {
        throw std::invalid_argument("matrix element count does not match dimensions");
    }
}

Matrix convolution_serial(const Matrix& input, const Matrix& kernel) {
    if (input.rows == 0 || input.cols == 0 || kernel.rows == 0 ||
        kernel.cols == 0) {
        throw std::invalid_argument("input and kernel must be non-empty");
    }
    if (kernel.rows > input.rows || kernel.cols > input.cols) {
        throw std::invalid_argument("kernel must not be larger than input");
    }

    Matrix output(input.rows - kernel.rows + 1,
                  input.cols - kernel.cols + 1);

    for (std::size_t output_row = 0; output_row < output.rows; ++output_row) {
        for (std::size_t output_col = 0; output_col < output.cols; ++output_col) {
            double sum = 0.0;
            for (std::size_t kernel_row = 0; kernel_row < kernel.rows;
                 ++kernel_row) {
                for (std::size_t kernel_col = 0; kernel_col < kernel.cols;
                     ++kernel_col) {
                    sum += input(output_row + kernel_row,
                                 output_col + kernel_col) *
                           kernel(kernel_row, kernel_col);
                }
            }
            output(output_row, output_col) = sum;
        }
    }

    return output;
}

}  // namespace conv
