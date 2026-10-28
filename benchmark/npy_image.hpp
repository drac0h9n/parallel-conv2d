#pragma once

// Input helpers for the timing harness: dataset images stored as NumPy .npy
// files (written by benchmark/prepare_dataset.py) and the Gaussian kernels.

#include "convolution.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bench {

namespace detail {

inline std::size_t parse_dimension(const std::string& text, const std::string& path) {
    std::size_t first = text.find_first_not_of(" ");
    std::size_t last = text.find_last_not_of(" ");
    if (first == std::string::npos) {
        throw std::runtime_error(path + ": empty dimension in .npy shape");
    }
    const std::string digits = text.substr(first, last - first + 1);
    if (digits.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error(path + ": invalid dimension '" + digits + "' in .npy shape");
    }
    const unsigned long long value = std::stoull(digits);
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(path + ": unsupported dimension in .npy shape");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace detail

// Reads a two-dimensional uint8 NumPy array (format 1.0 to 3.0, C order) and
// returns it as a matrix whose values are the pixels divided by 255.
inline conv::Matrix load_grayscale_npy(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[6] = {};
    stream.read(magic, 6);
    if (!stream || std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
        throw std::runtime_error(path + ": not a .npy file");
    }
    unsigned char version[2] = {};
    stream.read(reinterpret_cast<char*>(version), 2);
    std::uint32_t header_length = 0;
    if (stream && version[0] == 1) {
        unsigned char bytes[2] = {};
        stream.read(reinterpret_cast<char*>(bytes), 2);
        header_length = static_cast<std::uint32_t>(bytes[0]) |
                        (static_cast<std::uint32_t>(bytes[1]) << 8);
    } else if (stream && (version[0] == 2 || version[0] == 3)) {
        unsigned char bytes[4] = {};
        stream.read(reinterpret_cast<char*>(bytes), 4);
        header_length = static_cast<std::uint32_t>(bytes[0]) |
                        (static_cast<std::uint32_t>(bytes[1]) << 8) |
                        (static_cast<std::uint32_t>(bytes[2]) << 16) |
                        (static_cast<std::uint32_t>(bytes[3]) << 24);
    } else {
        throw std::runtime_error(path + ": unsupported .npy format version");
    }
    if (!stream || header_length == 0 || header_length > (1u << 20)) {
        throw std::runtime_error(path + ": invalid .npy header length");
    }
    std::string header(header_length, '\0');
    stream.read(&header[0], static_cast<std::streamsize>(header_length));
    if (!stream) {
        throw std::runtime_error(path + ": truncated .npy header");
    }
    if (header.find("'descr': '|u1'") == std::string::npos &&
        header.find("'descr': '<u1'") == std::string::npos &&
        header.find("'descr': 'u1'") == std::string::npos) {
        throw std::runtime_error(path + ": expected a uint8 array (descr '|u1')");
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error(path + ": expected a C-order (row-major) array");
    }
    const std::size_t shape_key = header.find("'shape':");
    const std::size_t open = header.find('(', shape_key);
    const std::size_t close = header.find(')', open);
    if (shape_key == std::string::npos || open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error(path + ": .npy header has no shape");
    }
    std::string shape = header.substr(open + 1, close - open - 1);
    if (!shape.empty() && shape.back() == ',') {
        shape.pop_back();
    }
    const std::size_t comma = shape.find(',');
    if (comma == std::string::npos || shape.find(',', comma + 1) != std::string::npos) {
        throw std::runtime_error(path + ": expected a two-dimensional array");
    }
    const std::size_t rows = detail::parse_dimension(shape.substr(0, comma), path);
    const std::size_t cols = detail::parse_dimension(shape.substr(comma + 1), path);
    if (cols > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::runtime_error(path + ": image is too large");
    }

    std::vector<unsigned char> pixels(rows * cols);
    stream.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (static_cast<std::size_t>(stream.gcount()) != pixels.size()) {
        throw std::runtime_error(path + ": truncated .npy data");
    }
    std::vector<double> values(pixels.size());
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        values[index] = static_cast<double>(pixels[index]) / 255.0;
    }
    return conv::Matrix(rows, cols, std::move(values));
}

// Standard deviation OpenCV's getGaussianKernel uses for a K-tap kernel when no
// sigma is given: 0.3 * ((K - 1) / 2 - 1) + 0.8.
inline double default_gaussian_sigma(std::size_t size) {
    return 0.3 * ((static_cast<double>(size) - 1.0) * 0.5 - 1.0) + 0.8;
}

// Normalized K x K Gaussian blur kernel: the outer product of a 1-D Gaussian
// with the default sigma, so the weights sum to one.
inline conv::Matrix gaussian_kernel(std::size_t size) {
    if (size == 0) {
        throw std::invalid_argument("kernel size must be positive");
    }
    const double sigma = default_gaussian_sigma(size);
    const double centre = (static_cast<double>(size) - 1.0) / 2.0;
    std::vector<double> line(size);
    double sum = 0.0;
    for (std::size_t index = 0; index < size; ++index) {
        const double offset = static_cast<double>(index) - centre;
        line[index] = std::exp(-(offset * offset) / (2.0 * sigma * sigma));
        sum += line[index];
    }
    conv::Matrix kernel(size, size);
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t col = 0; col < size; ++col) {
            kernel(row, col) = (line[row] / sum) * (line[col] / sum);
        }
    }
    return kernel;
}

}  // namespace bench
