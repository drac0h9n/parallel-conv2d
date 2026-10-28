// Checks the timing harness's dataset reader (uint8 .npy images) and its
// Gaussian kernels.

#include "npy_image.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Writes a .npy file with an arbitrary header dictionary and payload.
void write_npy(const fs::path& path, const std::string& dictionary,
               const std::vector<unsigned char>& payload, int major_version = 1) {
    const std::size_t prefix = major_version == 1 ? 10 : 12;
    std::string header = dictionary;
    while ((prefix + header.size() + 1) % 64 != 0) {
        header += ' ';
    }
    header += '\n';
    std::ofstream stream(path, std::ios::binary);
    stream.write("\x93NUMPY", 6);
    const char version[2] = {static_cast<char>(major_version), 0};
    stream.write(version, 2);
    const std::uint32_t length = static_cast<std::uint32_t>(header.size());
    if (major_version == 1) {
        const char bytes[2] = {static_cast<char>(length & 0xff), static_cast<char>((length >> 8) & 0xff)};
        stream.write(bytes, 2);
    } else {
        const char bytes[4] = {static_cast<char>(length & 0xff), static_cast<char>((length >> 8) & 0xff),
                               static_cast<char>((length >> 16) & 0xff), static_cast<char>((length >> 24) & 0xff)};
        stream.write(bytes, 4);
    }
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

std::vector<unsigned char> ramp(std::size_t count) {
    std::vector<unsigned char> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<unsigned char>((index * 23) % 256);
    }
    return values;
}

void expect_failure(const fs::path& path, const std::string& label) {
    try {
        static_cast<void>(bench::load_grayscale_npy(path.string()));
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(label + " was accepted");
}

void test_reader(const fs::path& directory) {
    const std::string uint8_3x4 = "{'descr': '|u1', 'fortran_order': False, 'shape': (3, 4), }";
    const std::vector<unsigned char> pixels = ramp(12);

    for (const int version : {1, 2}) {
        const fs::path path = directory / ("v" + std::to_string(version) + ".npy");
        write_npy(path, uint8_3x4, pixels, version);
        const conv::Matrix image = bench::load_grayscale_npy(path.string());
        require(image.rows == 3 && image.cols == 4, "wrong shape");
        for (std::size_t index = 0; index < pixels.size(); ++index) {
            require(image.values[index] == static_cast<double>(pixels[index]) / 255.0,
                    "pixel " + std::to_string(index) + " differs");
        }
    }

    const fs::path bad_magic = directory / "bad_magic.npy";
    std::ofstream(bad_magic, std::ios::binary) << "not a numpy file";
    expect_failure(bad_magic, "bad magic");

    const struct {
        const char* name;
        std::string dictionary;
        std::size_t bytes;
    } invalid[] = {
        {"float64", "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }", 96},
        {"fortran", "{'descr': '|u1', 'fortran_order': True, 'shape': (3, 4), }", 12},
        {"one_dimensional", "{'descr': '|u1', 'fortran_order': False, 'shape': (12,), }", 12},
        {"three_dimensional", "{'descr': '|u1', 'fortran_order': False, 'shape': (1, 3, 4), }", 12},
        {"truncated", "{'descr': '|u1', 'fortran_order': False, 'shape': (3, 4), }", 11},
    };
    for (const auto& item : invalid) {
        const fs::path path = directory / (std::string(item.name) + ".npy");
        write_npy(path, item.dictionary, ramp(item.bytes));
        expect_failure(path, item.name);
    }
    expect_failure(directory / "missing.npy", "missing file");
}

void test_gaussian() {
    const conv::Matrix single = bench::gaussian_kernel(1);
    require(single.rows == 1 && single.cols == 1 && std::abs(single.values[0] - 1.0) < 1e-15,
            "1x1 Gaussian must be [1]");
    const double sigmas[][2] = {{3, 0.8}, {5, 1.1}, {7, 1.4}, {11, 2.0}};
    for (const auto& entry : sigmas) {
        const std::size_t size = static_cast<std::size_t>(entry[0]);
        require(std::abs(bench::default_gaussian_sigma(size) - entry[1]) < 1e-12, "unexpected sigma");
        const conv::Matrix kernel = bench::gaussian_kernel(size);
        double sum = 0.0;
        const std::size_t centre = size / 2;
        for (std::size_t row = 0; row < size; ++row) {
            for (std::size_t col = 0; col < size; ++col) {
                const double value = kernel(row, col);
                sum += value;
                require(value > 0.0 && value <= kernel(centre, centre), "Gaussian peak must be central");
                require(std::abs(value - kernel(col, row)) < 1e-15, "Gaussian must be symmetric");
                require(std::abs(value - kernel(size - 1 - row, col)) < 1e-15, "Gaussian must be mirror-symmetric");
                // Separable (rank one): k(r, c) * k(m, m) == k(r, m) * k(m, c).
                require(std::abs(value * kernel(centre, centre) -
                                 kernel(row, centre) * kernel(centre, col)) < 1e-15,
                        "Gaussian must be separable");
            }
        }
        require(std::abs(sum - 1.0) < 1e-12, "Gaussian weights must sum to one");
    }
}

}  // namespace

int main() {
    const fs::path directory = fs::temp_directory_path() / "conv_npy_image_test";
    try {
        fs::create_directories(directory);
        test_reader(directory);
        test_gaussian();
        fs::remove_all(directory);
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS: npy image reader and Gaussian kernels\n";
    return 0;
}
