#!/usr/bin/env python3
"""Cross-validate all implementations on CIFAR-10, CIFAR-100 and ImageNet images."""

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
    "dataset_reference_conv", REPOSITORY_ROOT / "tests" / "reference_conv.py"
)
PREPARE_CIFAR10 = load_module(
    "dataset_prepare_cifar10",
    REPOSITORY_ROOT / "tests" / "datasets" / "prepare_cifar10.py",
)
PREPARE_CIFAR100 = load_module(
    "dataset_prepare_cifar100",
    REPOSITORY_ROOT / "tests" / "datasets" / "prepare_cifar100.py",
)
PREPARE_IMAGENET = load_module(
    "dataset_prepare_imagenet",
    REPOSITORY_ROOT / "tests" / "datasets" / "prepare_imagenet_subset.py",
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


def run_cli(
    executable: Path,
    input_path: Path,
    kernel_path: Path,
    output_path: Path,
    thread_count: int | None = None,
) -> list[list[float]]:
    command = [str(executable), str(input_path), str(kernel_path), str(output_path)]
    if thread_count is not None:
        command.append(str(thread_count))
    subprocess.run(command, check=True)
    return read_matrix(output_path)


def check_case(
    serial_cli: Path,
    basic_cli: Path,
    optimized_cli: Path,
    input_values: Sequence[Sequence[float]],
    kernel_values: Sequence[Sequence[float]],
    thread_count: int,
) -> None:
    expected = REFERENCE.convolution_reference(input_values, kernel_values)
    with tempfile.TemporaryDirectory(prefix="conv-datasets-") as temp_name:
        temp = Path(temp_name)
        input_path = temp / "input.txt"
        kernel_path = temp / "kernel.txt"
        output_path = temp / "output.txt"
        write_matrix(input_path, input_values)
        write_matrix(kernel_path, kernel_values)

        serial = run_cli(serial_cli, input_path, kernel_path, output_path)
        basic = run_cli(
            basic_cli,
            input_path,
            kernel_path,
            output_path,
            thread_count,
        )
        optimized = run_cli(
            optimized_cli,
            input_path,
            kernel_path,
            output_path,
            thread_count,
        )
        assert_close(serial, expected)
        assert_close(basic, serial)
        assert_close(optimized, serial)
        assert_close(basic, expected)
        assert_close(optimized, expected)


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


def load_imagenet_images(paths: Sequence[Path]) -> list[list[list[float]]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for the ImageNet subset tests") from error

    images: list[list[list[float]]] = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((192, 192))
            width, height = image.size
            pixels = list(image.getdata())
        if width < 7 or height < 7:
            raise RuntimeError(f"ImageNet sample is smaller than 7x7: {path}")
        grayscale = []
        for row in range(height):
            output_row = []
            for col in range(width):
                red, green, blue = pixels[row * width + col]
                output_row.append(
                    (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
                )
            grayscale.append(output_row)
        images.append(grayscale)
    return images


def kernel_for_size(kernel_size: int) -> list[list[float]]:
    if kernel_size == 3:
        return [
            [-1.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, -1.0],
        ]
    generator = random.Random(750 + kernel_size)
    scale = 1.0 / (kernel_size * kernel_size)
    return [
        [generator.uniform(-1.0, 1.0) * scale for _ in range(kernel_size)]
        for _ in range(kernel_size)
    ]


def load_dataset(dataset: str, data_root: Path) -> list[list[list[float]]]:
    if dataset == "cifar10":
        return load_cifar_images(PREPARE_CIFAR10.prepare(data_root), count=100)
    if dataset == "cifar100":
        return load_cifar_images(PREPARE_CIFAR100.prepare(data_root), count=100)
    return load_imagenet_images(PREPARE_IMAGENET.prepare(data_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "imagenet"), required=True)
    parser.add_argument("--kernel-size", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--serial-cli", type=Path, required=True)
    parser.add_argument("--basic-cli", type=Path, required=True)
    parser.add_argument("--optimized-cli", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    for executable in (args.serial_cli, args.basic_cli, args.optimized_cli):
        if not executable.is_file():
            parser.error(f"convolution executable does not exist: {executable}")

    expected_kernel_sizes = {
        "cifar10": {3},
        "cifar100": {3, 5},
        "imagenet": {3, 7},
    }
    if args.kernel_size not in expected_kernel_sizes[args.dataset]:
        parser.error(
            f"kernel size {args.kernel_size} is not in the {args.dataset} test matrix"
        )

    images = load_dataset(args.dataset, args.data_root)
    expected_count = 20 if args.dataset == "imagenet" else 100
    if len(images) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} {args.dataset} images, got {len(images)}"
        )
    kernel = kernel_for_size(args.kernel_size)
    thread_count = min(8, max(1, os.cpu_count() or 1))
    for image in images:
        check_case(
            args.serial_cli,
            args.basic_cli,
            args.optimized_cli,
            image,
            kernel,
            thread_count,
        )
    print(
        f"PASS: {args.dataset} {len(images)} image(s), "
        f"{args.kernel_size}x{args.kernel_size} kernel, {thread_count} thread(s)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
