#!/usr/bin/env python3
"""Compare the C++ serial implementation with an independent PyTorch result."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


REFERENCE = load_module(
    "reference_conv", REPOSITORY_ROOT / "tests" / "reference_conv.py"
)
PREPARE_CIFAR = load_module(
    "prepare_cifar10", REPOSITORY_ROOT / "tests" / "datasets" / "prepare_cifar10.py"
)


def write_matrix(path: Path, values: Sequence[Sequence[float]]) -> None:
    rows = len(values)
    cols = len(values[0])
    if rows == 0 or cols == 0 or any(len(row) != cols for row in values):
        raise ValueError("matrix must be non-empty and rectangular")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{rows} {cols}\n")
        for row in values:
            stream.write(" ".join(f"{value:.17g}" for value in row))
            stream.write("\n")


def read_matrix(path: Path) -> list[list[float]]:
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) < 2:
        raise ValueError("matrix output has no header")
    rows, cols = int(tokens[0]), int(tokens[1])
    flat_values = [float(value) for value in tokens[2:]]
    if len(flat_values) != rows * cols:
        raise ValueError("matrix output has the wrong element count")
    return [flat_values[row * cols : (row + 1) * cols] for row in range(rows)]


def assert_close(
    actual: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
    tolerance: float = 1e-10,
) -> None:
    if len(actual) != len(expected) or any(
        len(actual_row) != len(expected_row)
        for actual_row, expected_row in zip(actual, expected)
    ):
        raise AssertionError("matrix dimensions differ")

    largest_error = 0.0
    largest_location = (0, 0)
    for row, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        for col, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row)
        ):
            error = abs(actual_value - expected_value)
            if error > largest_error:
                largest_error = error
                largest_location = (row, col)
    if largest_error > tolerance:
        raise AssertionError(
            f"maximum error {largest_error} at {largest_location} exceeds {tolerance}"
        )


def check_case(
    conv_cli: Path,
    input_values: Sequence[Sequence[float]],
    kernel_values: Sequence[Sequence[float]],
    omp_cli: Path | None = None,
    thread_counts: Sequence[int] = (),
) -> None:
    expected = REFERENCE.convolution_reference(input_values, kernel_values)
    with tempfile.TemporaryDirectory(
        prefix="conv-reference-"
    ) as temporary_directory:
        temp = Path(temporary_directory)
        input_path = temp / "input.txt"
        kernel_path = temp / "kernel.txt"
        output_path = temp / "output.txt"
        write_matrix(input_path, input_values)
        write_matrix(kernel_path, kernel_values)
        subprocess.run(
            [str(conv_cli), str(input_path), str(kernel_path), str(output_path)],
            check=True,
        )
        serial = read_matrix(output_path)
        assert_close(serial, expected)

        if omp_cli is not None:
            for thread_count in thread_counts:
                subprocess.run(
                    [
                        str(omp_cli),
                        str(input_path),
                        str(kernel_path),
                        str(output_path),
                        str(thread_count),
                    ],
                    check=True,
                )
                omp_output = read_matrix(output_path)
                assert_close(omp_output, serial)
                assert_close(omp_output, expected)


def test_random(conv_cli: Path) -> None:
    generator = random.Random(750)
    dimensions = [
        (8, 8, 1, 1),
        (9, 13, 2, 2),
        (17, 23, 3, 5),
        (25, 19, 5, 3),
        (32, 32, 7, 7),
    ]
    for input_rows, input_cols, kernel_rows, kernel_cols in dimensions:
        input_values = [
            [generator.uniform(-2.0, 2.0) for _ in range(input_cols)]
            for _ in range(input_rows)
        ]
        kernel_values = [
            [generator.uniform(-1.0, 1.0) for _ in range(kernel_cols)]
            for _ in range(kernel_rows)
        ]
        check_case(conv_cli, input_values, kernel_values)


def load_cifar_images(test_batch: Path, count: int) -> list[list[list[float]]]:
    with test_batch.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    data = batch[b"data"]
    if len(data) < count:
        raise RuntimeError(f"CIFAR test batch contains fewer than {count} images")

    images: list[list[list[float]]] = []
    for flat_image in data[:count]:
        grayscale = [[0.0 for _ in range(32)] for _ in range(32)]
        for row in range(32):
            for col in range(32):
                offset = row * 32 + col
                red = float(flat_image[offset])
                green = float(flat_image[1024 + offset])
                blue = float(flat_image[2048 + offset])
                grayscale[row][col] = (
                    0.299 * red + 0.587 * green + 0.114 * blue
                ) / 255.0
        images.append(grayscale)
    return images


def supported_thread_counts() -> list[int]:
    available_threads = max(1, os.cpu_count() or 1)
    return list(
        dict.fromkeys(
            min(requested, available_threads) for requested in (1, 2, 4, 8)
        )
    )


def test_cifar(
    conv_cli: Path,
    data_root: Path,
    count: int = 10,
    omp_cli: Path | None = None,
) -> None:
    test_batch = PREPARE_CIFAR.prepare(data_root)
    laplacian = [
        [-1.0, -1.0, -1.0],
        [-1.0, 8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]
    thread_counts = supported_thread_counts() if omp_cli is not None else ()
    for image in load_cifar_images(test_batch, count=count):
        check_case(conv_cli, image, laplacian, omp_cli, thread_counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("random", "cifar", "cifar-omp"), required=True
    )
    parser.add_argument("--conv-cli", type=Path, required=True)
    parser.add_argument("--omp-cli", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    if not args.conv_cli.is_file():
        parser.error(f"C++ convolution executable does not exist: {args.conv_cli}")

    if args.mode == "random":
        test_random(args.conv_cli)
    elif args.mode == "cifar":
        if args.data_root is None:
            parser.error("--data-root is required for CIFAR mode")
        test_cifar(args.conv_cli, args.data_root)
    else:
        if args.data_root is None or args.omp_cli is None:
            parser.error("--data-root and --omp-cli are required for CIFAR-OMP mode")
        if not args.omp_cli.is_file():
            parser.error(f"OpenMP convolution executable does not exist: {args.omp_cli}")
        test_cifar(args.conv_cli, args.data_root, count=100, omp_cli=args.omp_cli)
    print(f"PASS: {args.mode} cross-validation matches PyTorch")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
