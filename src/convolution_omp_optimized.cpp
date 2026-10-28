#include "convolution.hpp"

#include <cstddef>
#include <stdexcept>

namespace conv {

Matrix convolution_omp_optimized(const Matrix& input, const Matrix& kernel,
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

    const double* const input_data = input.values.data();
    const double* const kernel_data = kernel.values.data();
    double* const output_data = output.values.data();
    const std::size_t input_cols = input.cols;
    const std::size_t kernel_rows = kernel.rows;
    const std::size_t kernel_cols = kernel.cols;
    const std::size_t output_cols = output.cols;
    const std::ptrdiff_t output_rows =
        static_cast<std::ptrdiff_t>(output.rows);
    const std::ptrdiff_t output_columns =
        static_cast<std::ptrdiff_t>(output.cols);

#pragma omp parallel for collapse(2) schedule(static) default(none)           \
    shared(input_data, kernel_data, output_data, input_cols, kernel_rows,     \
               kernel_cols, output_cols, output_rows, output_columns)         \
    num_threads(thread_count)
    for (std::ptrdiff_t output_row = 0; output_row < output_rows;
         ++output_row) {
        for (std::ptrdiff_t output_col = 0; output_col < output_columns;
             ++output_col) {
            const std::size_t row =
                static_cast<std::size_t>(output_row);
            const std::size_t col =
                static_cast<std::size_t>(output_col);
            const double* const input_window =
                input_data + row * input_cols + col;

            double sum = 0.0;
            for (std::size_t kernel_row = 0; kernel_row < kernel_rows;
                 ++kernel_row) {
                const double* const input_line =
                    input_window + kernel_row * input_cols;
                const double* const kernel_line =
                    kernel_data + kernel_row * kernel_cols;
                for (std::size_t kernel_col = 0; kernel_col < kernel_cols;
                     ++kernel_col) {
                    sum += input_line[kernel_col] * kernel_line[kernel_col];
                }
            }
            output_data[row * output_cols + col] = sum;
        }
    }

    return output;
}

}  // namespace conv
