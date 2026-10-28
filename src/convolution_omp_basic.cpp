#include "convolution.hpp"

#include <cstddef>
#include <stdexcept>

namespace conv {

Matrix convolution_omp_basic(const Matrix& input, const Matrix& kernel,
                             int thread_count) {
    if (input.rows == 0 || input.cols == 0 || kernel.rows == 0 ||
        kernel.cols == 0) {
        throw std::invalid_argument("input and kernel must be non-empty");
    }
    if (kernel.rows > input.rows || kernel.cols > input.cols) {
        throw std::invalid_argument("kernel must not be larger than input");
    }
    if (thread_count <= 0) {
        throw std::invalid_argument("thread count must be positive");
    }

    Matrix output(input.rows - kernel.rows + 1,
                  input.cols - kernel.cols + 1);

#pragma omp parallel for default(none) shared(input, kernel, output) \
    num_threads(thread_count)
    for (std::ptrdiff_t output_row = 0;
         output_row < static_cast<std::ptrdiff_t>(output.rows); ++output_row) {
        for (std::size_t output_col = 0; output_col < output.cols; ++output_col) {
            double sum = 0.0;
            for (std::size_t kernel_row = 0; kernel_row < kernel.rows;
                 ++kernel_row) {
                for (std::size_t kernel_col = 0; kernel_col < kernel.cols;
                     ++kernel_col) {
                    sum += input(static_cast<std::size_t>(output_row) + kernel_row,
                                 output_col + kernel_col) *
                           kernel(kernel_row, kernel_col);
                }
            }
            output(static_cast<std::size_t>(output_row), output_col) = sum;
        }
    }

    return output;
}

}  // namespace conv
