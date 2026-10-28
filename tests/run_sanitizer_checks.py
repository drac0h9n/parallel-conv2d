#!/usr/bin/env python3
"""Configure and run the core suite under AddressSanitizer and UBSan."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


CORE_TEST_REGEX = (
    r"^(S1_T[1-4]_.*|S2_T[1-4]_.*|S2_T6_.*|S2_T7_.*|"
    r"S3_T1_.*|S3_T2_.*)$"
)


def run(command: list[str], environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def check_sanitizer(
    name: str,
    option: str,
    cmake: Path,
    source_dir: Path,
    build_root: Path,
    generator: str,
    compiler: Path,
) -> None:
    build_dir = build_root / name
    configure_command = [
        str(cmake),
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        generator,
        "-DCMAKE_BUILD_TYPE=Debug",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        "-DBUILD_TESTING=ON",
        "-DENABLE_SANITIZER_DRIVER=OFF",
        f"-D{option}=ON",
    ]
    run(configure_command)
    run([str(cmake), "--build", str(build_dir), "--config", "Debug", "--parallel", "2"])

    environment = os.environ.copy()
    if name == "address":
        environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
    else:
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--build-config",
            "Debug",
            "--output-on-failure",
            "-R",
            CORE_TEST_REGEX,
        ],
        environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    args = parser.parse_args()

    check_sanitizer(
        "address",
        "ENABLE_ADDRESS_SANITIZER",
        args.cmake,
        args.source_dir,
        args.build_root,
        args.generator,
        args.compiler,
    )
    check_sanitizer(
        "undefined",
        "ENABLE_UNDEFINED_SANITIZER",
        args.cmake,
        args.source_dir,
        args.build_root,
        args.generator,
        args.compiler,
    )
    print("PASS: AddressSanitizer and UndefinedBehaviorSanitizer core suites")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"FAILED: command exited with status {error.returncode}", file=sys.stderr)
        raise
