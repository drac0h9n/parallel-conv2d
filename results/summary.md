# Benchmark summary

## Test environment

- CPU: Apple M4 Pro, 12 physical cores (8 performance + 4 efficiency), one hardware thread per core
- Machine: MacBook Pro (Mac16,8)
- Memory: 24.0 GiB unified memory
- OS: macOS 15.6.1 (build 24G90)
- Power: Now drawing from 'AC Power'
- OpenMP runtime: LLVM libomp 23.1.1 (Homebrew)
- Compiler: Apple clang version 16.0.0 (clang-1600.0.26.6); flags `-O3 -DNDEBUG` plus OpenMP; CMake build type Release
- OpenMP: `_OPENMP` = 202011; omp_get_num_procs() = 12
- cmake version 4.2.0; Python CPython 3.14.4
- Provenance: Measured before this repository's first commit; the per-file source SHA-256 values in environment.json match the committed sources
- Load average at start (1/5/15 min): 3.98 / 3.60 / 4.10
- Measured: 2026-10-28T14:21:43.794937-04:00

## Parameters

- Image sizes N: [256, 512, 1024, 2048, 4096]; kernel sizes K: [3, 5, 7, 11]; OpenMP threads p: [1, 2, 4, 6, 8, 10, 12]
- 3 untimed warm-up + 10 timed runs per configuration (300 configurations); medians reported
- Dataset: DIV2K validation set (high-resolution photos) (https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip), 10 images (0801 to 0810); each timed call uses the next image in turn
- Preprocessing: 8-bit grayscale (ITU-R BT.601 luma, Pillow mode L), central square crop, Lanczos resize to N x N (upscaling when N exceeds the square); stored as uint8 .npy and scaled to [0, 1] by the harness
- Kernels: normalized K x K Gaussian blur, outer product of a 1-D Gaussian with sigma = 0.3 * ((K - 1) / 2 - 1) + 0.8 (OpenCV's default sigma rule); see benchmark/npy_image.hpp

Throughput is images processed per second, 1000 / median time per image in ms. S(p) = T_impl(1) / T_impl(p) (self-relative, the same as the throughput ratio); "vs serial" = T_serial / T_impl(p); E(p) = S(p) / p. The end-to-end section stays in milliseconds because it splits one process run into parts.

## Strong scaling, 4096×4096, 3×3 kernel (sequential baseline 13.8 images/s)

| p | Basic img/s | Basic S(p) | Basic E(p) | Basic vs serial | Opt. img/s | Opt. S(p) | Opt. E(p) | Opt. vs serial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13.9 | 1.00 | 100% | 1.01 | 29.2 | 1.00 | 100% | 2.12 |
| 2 | 26.5 | 1.91 | 95% | 1.92 | 52.6 | 1.80 | 90% | 3.82 |
| 4 | 43.7 | 3.15 | 79% | 3.17 | 81.1 | 2.78 | 69% | 5.89 |
| 6 | 59.4 | 4.28 | 71% | 4.31 | 104 | 3.57 | 60% | 7.57 |
| 8 | 77.1 | 5.55 | 69% | 5.59 | 85.8 | 2.94 | 37% | 6.23 |
| 10 | 65.1 | 4.69 | 47% | 4.73 | 87.3 | 2.99 | 30% | 6.34 |
| 12 | 73.7 | 5.30 | 44% | 5.35 | 101 | 3.44 | 29% | 7.30 |

## Strong scaling, 4096×4096, 5×5 kernel (sequential baseline 6.24 images/s)

| p | Basic img/s | Basic S(p) | Basic E(p) | Basic vs serial | Opt. img/s | Opt. S(p) | Opt. E(p) | Opt. vs serial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.09 | 1.00 | 100% | 0.98 | 11.7 | 1.00 | 100% | 1.87 |
| 2 | 12.0 | 1.97 | 99% | 1.92 | 22.2 | 1.90 | 95% | 3.56 |
| 4 | 22.6 | 3.71 | 93% | 3.62 | 39.9 | 3.41 | 85% | 6.40 |
| 6 | 32.2 | 5.29 | 88% | 5.16 | 54.0 | 4.62 | 77% | 8.66 |
| 8 | 40.5 | 6.65 | 83% | 6.49 | 53.9 | 4.61 | 58% | 8.64 |
| 10 | 34.3 | 5.63 | 56% | 5.49 | 49.6 | 4.24 | 42% | 7.95 |
| 12 | 40.4 | 6.64 | 55% | 6.48 | 61.6 | 5.27 | 44% | 9.87 |

## Throughput vs image size, 3×3 kernel (OpenMP at p = 12)

| N | Serial img/s | Basic img/s | Opt. img/s | Basic vs serial | Opt. vs serial | Opt. S(12) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3,932 | 10,023 | 15,464 | 2.55 | 3.93 | 1.66 |
| 512 | 754 | 4,670 | 6,252 | 6.19 | 8.29 | 3.80 |
| 1024 | 210 | 1,010 | 1,483 | 4.82 | 7.07 | 2.99 |
| 2048 | 55.1 | 242 | 472 | 4.38 | 8.57 | 3.96 |
| 4096 | 13.8 | 73.7 | 101 | 5.35 | 7.30 | 3.44 |

## Throughput vs image size, 5×5 kernel (OpenMP at p = 12)

| N | Serial img/s | Basic img/s | Opt. img/s | Basic vs serial | Opt. vs serial | Opt. S(12) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1,743 | 7,918 | 9,129 | 4.54 | 5.24 | 2.65 |
| 512 | 393 | 2,223 | 3,140 | 5.66 | 8.00 | 3.95 |
| 1024 | 96.3 | 577 | 856 | 5.99 | 8.88 | 4.47 |
| 2048 | 25.1 | 144 | 244 | 5.72 | 9.74 | 5.45 |
| 4096 | 6.24 | 40.4 | 61.6 | 6.48 | 9.87 | 5.27 |

## Kernel size at 4096×4096 (OpenMP at p = 12)

| K | Serial img/s | Serial GFLOP/s | Basic img/s | Basic S | Basic E | Basic vs serial | Basic GFLOP/s | Opt. img/s | Opt. S | Opt. E | Opt. vs serial | Opt. GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 13.8 | 4.2 | 73.7 | 5.30 | 44% | 5.35 | 22.2 | 101 | 3.44 | 29% | 7.30 | 30.3 |
| 5×5 | 6.24 | 5.2 | 40.4 | 6.64 | 55% | 6.48 | 33.8 | 61.6 | 5.27 | 44% | 9.87 | 51.5 |
| 7×7 | 2.89 | 4.7 | 22.8 | 7.90 | 66% | 7.91 | 37.4 | 33.6 | 7.15 | 60% | 11.62 | 55.0 |
| 11×11 | 0.889 | 3.6 | 7.82 | 8.74 | 73% | 8.80 | 31.6 | 14.4 | 8.54 | 71% | 16.15 | 58.0 |

## Highest throughput per kernel at 4096×4096

| K | Implementation | p | Images/s | Speedup vs serial |
| --- | --- | ---: | ---: | ---: |
| 3×3 | optimized | 6 | 104 | 7.57 |
| 5×5 | optimized | 12 | 61.6 | 9.87 |
| 7×7 | optimized | 12 | 33.6 | 11.62 |
| 11×11 | optimized | 12 | 14.4 | 16.15 |

## End-to-end (whole process, cold) vs kernel time

End-to-end = process launch + loading one dataset image + OpenMP start-up + one cold convolution + exit, timed from the parent process (median of repeats). Warm kernel = median steady-state call from the main matrix.

| Impl. | N | K | p | End-to-end ms | Cold kernel ms | Warm kernel ms | Kernel share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 256 | 3 | 1 | 4.05 | 0.282 | 0.254 | 6% |
| serial | 512 | 3 | 1 | 4.95 | 1.146 | 1.326 | 23% |
| serial | 1024 | 3 | 1 | 9.19 | 4.557 | 4.769 | 50% |
| serial | 2048 | 3 | 1 | 26.75 | 18.671 | 18.142 | 68% |
| serial | 4096 | 3 | 1 | 97.12 | 74.707 | 72.563 | 75% |
| serial | 256 | 5 | 1 | 4.15 | 0.627 | 0.574 | 14% |
| serial | 512 | 5 | 1 | 6.28 | 2.494 | 2.546 | 40% |
| serial | 1024 | 5 | 1 | 14.45 | 10.396 | 10.380 | 72% |
| serial | 2048 | 5 | 1 | 48.39 | 40.526 | 39.845 | 82% |
| serial | 4096 | 5 | 1 | 188.00 | 165.562 | 160.262 | 85% |
| basic | 256 | 3 | 12 | 4.17 | 0.227 | 0.100 | 2% |
| basic | 512 | 3 | 12 | 4.27 | 0.403 | 0.214 | 5% |
| basic | 1024 | 3 | 12 | 6.09 | 1.180 | 0.990 | 16% |
| basic | 2048 | 3 | 12 | 12.10 | 4.122 | 4.139 | 34% |
| basic | 4096 | 3 | 12 | 41.09 | 16.749 | 13.570 | 33% |
| basic | 256 | 5 | 12 | 4.02 | 0.260 | 0.126 | 3% |
| basic | 512 | 5 | 12 | 4.70 | 0.606 | 0.450 | 10% |
| basic | 1024 | 5 | 12 | 6.53 | 1.997 | 1.732 | 27% |
| basic | 2048 | 5 | 12 | 15.40 | 7.442 | 6.968 | 45% |
| basic | 4096 | 5 | 12 | 51.46 | 27.462 | 24.745 | 48% |
| optimized | 256 | 3 | 12 | 3.83 | 0.202 | 0.065 | 2% |
| optimized | 512 | 3 | 12 | 4.31 | 0.330 | 0.160 | 4% |
| optimized | 1024 | 3 | 12 | 5.54 | 0.799 | 0.674 | 12% |
| optimized | 2048 | 3 | 12 | 10.80 | 2.898 | 2.117 | 20% |
| optimized | 4096 | 3 | 12 | 34.93 | 11.039 | 9.947 | 28% |
| optimized | 256 | 5 | 12 | 3.68 | 0.215 | 0.110 | 3% |
| optimized | 512 | 5 | 12 | 4.33 | 0.417 | 0.318 | 7% |
| optimized | 1024 | 5 | 12 | 5.98 | 1.269 | 1.168 | 20% |
| optimized | 2048 | 5 | 12 | 12.78 | 4.804 | 4.091 | 32% |
| optimized | 4096 | 5 | 12 | 40.84 | 18.845 | 16.241 | 40% |

Kernel-only vs end-to-end speedup of Optimized OpenMP (p = 12) over the sequential baseline. Process start-up and image loading stay serial, so Amdahl's law caps the end-to-end gain:

| N | K | Kernel speedup | End-to-end speedup |
| --- | ---: | ---: | ---: |
| 256 | 3 | 3.93 | 1.06 |
| 512 | 3 | 8.29 | 1.15 |
| 1024 | 3 | 7.07 | 1.66 |
| 2048 | 3 | 8.57 | 2.48 |
| 4096 | 3 | 7.30 | 2.78 |
| 256 | 5 | 5.24 | 1.13 |
| 512 | 5 | 8.00 | 1.45 |
| 1024 | 5 | 8.88 | 2.42 |
| 2048 | 5 | 9.74 | 3.79 |
| 4096 | 5 | 9.87 | 4.60 |

