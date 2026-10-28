#include "convolution.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

conv::Matrix read_matrix(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open matrix file: " + path);
    }

    std::size_t rows = 0;
    std::size_t cols = 0;
    if (!(stream >> rows >> cols) || rows == 0 || cols == 0) {
        throw std::runtime_error("invalid matrix header in: " + path);
    }

    std::vector<double> values;
    values.reserve(rows * cols);
    double value = 0.0;
    while (stream >> value) {
        values.push_back(value);
    }
    if (!stream.eof()) {
        throw std::runtime_error("invalid numeric value in: " + path);
    }
    return conv::Matrix(rows, cols, std::move(values));
}

void write_matrix(const conv::Matrix& matrix, const std::string& path) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot write matrix file: " + path);
    }

    stream << matrix.rows << ' ' << matrix.cols << '\n';
    stream << std::setprecision(17);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            if (col != 0) {
                stream << ' ';
            }
            stream << matrix(row, col);
        }
        stream << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: conv_serial_cli INPUT KERNEL OUTPUT\n";
        return 2;
    }

    try {
        const conv::Matrix input = read_matrix(argv[1]);
        const conv::Matrix kernel = read_matrix(argv[2]);
        write_matrix(conv::convolution_serial(input, kernel), argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "conv_serial_cli: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
