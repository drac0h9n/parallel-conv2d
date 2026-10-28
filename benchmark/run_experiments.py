#!/usr/bin/env python3
"""Run every performance experiment for the parallel 2-D convolution study.

benchmark/run_benchmark.sh builds the project (Release) and then calls this
script. It can also be run directly against an existing build:

    python3 benchmark/run_experiments.py --build-dir build --output-dir results

Steps
  1. Correctness gate: every CTest test labelled "correctness" must pass, or
     nothing is measured (exit status 1).
  2. Dataset: benchmark/prepare_dataset.py downloads and verifies the DIV2K
     validation photos (first use only) and converts them to N x N grayscale.
  3. Timing matrix: serial, basic and optimized for each image size N, kernel
     size K and OpenMP thread count p. Every configuration runs in a fresh
     process that loads the dataset images, makes W untimed warm-up calls, then
     R timed calls, each on the next image in turn.
  4. End-to-end pass: selected configurations timed as whole processes
     (launch + loading one image + OpenMP start-up + one cold call + exit).
Outputs in --output-dir: samples.csv, benchmark_data.csv, end_to_end.csv,
environment.json and run_info.json, plus logs/run_<time>.log. The CSV and JSON
files are replaced only when the whole run succeeds.

Needs Python 3.9 or newer; the dataset step needs Pillow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = ("serial", "basic", "optimized")
E2E_SEED = 750
KERNEL_DESCRIPTION = (
    "normalized K x K Gaussian blur, outer product of a 1-D Gaussian with sigma = 0.3 * ((K - 1) / 2 - 1) + 0.8 "
    "(OpenCV's default sigma rule); see benchmark/npy_image.hpp"
)
PRESETS = {
    "full": {
        "sizes": [256, 512, 1024, 2048, 4096],
        "kernels": [3, 5, 7, 11],
        "max_threads": None,
        "warmups": 3,
        "runs": 10,
        "e2e_kernels": [3, 5],
        "e2e_repeats": 5,
        "dataset": "div2k",
        "images": 10,
    },
    "smoke": {
        "sizes": [32, 64],
        "kernels": [3, 5],
        "max_threads": 2,
        "warmups": 1,
        "runs": 3,
        "e2e_kernels": [3],
        "e2e_repeats": 1,
        "dataset": "imagenet-sample",  # already downloaded by the correctness tests
        "images": 3,
    },
}
# Keys copied from `system_profiler SPHardwareDataType`. Serial numbers,
# hardware UUIDs and provisioning IDs are deliberately never recorded.
SAFE_HARDWARE_KEYS = {"Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory"}
SUMMARY_FIELDS = (
    "implementation", "image_size", "kernel", "threads", "warmups", "runs",
    "median_ms", "q1_ms", "q3_ms", "mean_ms", "std_ms", "min_ms", "max_ms",
    "speedup", "speedup_vs_serial", "efficiency", "gflops", "images_per_s",
)
SAMPLE_FIELDS = ("implementation", "image_size", "kernel", "threads", "run", "image", "runtime_ms")
E2E_FIELDS = ("implementation", "image_size", "kernel", "threads", "repeat", "end_to_end_ms", "cold_kernel_ms")

Config = Tuple[str, int, int, int]  # (implementation, N, K, threads)


class QualificationError(RuntimeError):
    """Raised when the correctness gate does not pass."""


class Tee:
    """Mirror everything to the console and to the run log."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = log_path
        self._stream = log_path.open("a", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self._stream.write(text)
        self._stream.flush()

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def banner(self, text: str) -> None:
        self.line("")
        self.line("=" * 78)
        self.line(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
        self.line("=" * 78)

    def close(self) -> None:
        self._stream.close()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def parse_int_list(text: str, label: str) -> List[int]:
    try:
        values = [int(token) for token in text.split(",") if token.strip()]
    except ValueError as error:
        raise SystemExit(f"error: {label} must be a comma-separated list of integers") from error
    if not values or any(value <= 0 for value in values):
        raise SystemExit(f"error: every {label} must be a positive integer")
    return list(dict.fromkeys(values))


def default_threads(logical_cores: int) -> List[int]:
    """1, 2, then every even count up to the logical core count (and that count)."""
    values = {1, 2, logical_cores}
    values.update(range(4, logical_cores + 1, 2))
    return sorted(value for value in values if 1 <= value <= logical_cores)


def quoted(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def output_of(command: Sequence[str], timeout: float = 30.0, strip: bool = True) -> Optional[str]:
    try:
        completed = subprocess.run([str(part) for part in command], capture_output=True, text=True,
                                   timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    text = completed.stdout.strip() if strip else completed.stdout.rstrip("\n")
    return text or None


def find_tool(name: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/Applications/CMake.app/Contents/bin"):
        candidate = Path(prefix) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise SystemExit(f"error: '{name}' was not found; install CMake or pass --{name} PATH")


def run_streaming(command: Sequence[str], tee: Tee, environment: Dict[str, str]) -> int:
    tee.line("+ " + quoted(command))
    process = subprocess.Popen([str(part) for part in command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment)
    assert process.stdout is not None
    for text in process.stdout:
        tee.write(text)
    return process.wait()


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:.6g}" if isinstance(value, float) else value)
                             for key, value in row.items()})


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------


def macos_environment() -> dict:
    info: dict = {
        "macos_version": output_of(("sw_vers", "-productVersion")),
        "macos_build": output_of(("sw_vers", "-buildVersion")),
    }
    sysctl: Dict[str, str] = {}
    for key in (
        "machdep.cpu.brand_string", "hw.model", "hw.ncpu", "hw.physicalcpu", "hw.logicalcpu",
        "hw.nperflevels", "hw.perflevel0.name", "hw.perflevel0.physicalcpu",
        "hw.perflevel0.l1dcachesize", "hw.perflevel0.l2cachesize", "hw.perflevel1.name",
        "hw.perflevel1.physicalcpu", "hw.perflevel1.l1dcachesize", "hw.perflevel1.l2cachesize",
        "hw.memsize", "hw.cachelinesize", "hw.pagesize",
    ):
        value = output_of(("sysctl", "-n", key))
        if value is not None:
            sysctl[key] = value
    info["sysctl"] = sysctl
    if sysctl.get("hw.memsize", "").isdigit():
        info["memory_gib"] = round(int(sysctl["hw.memsize"]) / 2**30, 1)
    overview: Dict[str, str] = {}
    hardware = output_of(("/usr/sbin/system_profiler", "SPHardwareDataType"), timeout=60)
    for raw in (hardware or "").splitlines():
        key, separator, value = raw.strip().partition(":")
        if separator and key in SAFE_HARDWARE_KEYS:
            overview[key] = value.strip()
    info["hardware_overview"] = overview
    battery = output_of(("pmset", "-g", "batt"))
    if battery:
        info["power_source"] = battery.splitlines()[0]
    for raw in (output_of(("pmset", "-g")) or "").splitlines():
        parts = raw.split()
        if len(parts) >= 2 and parts[0] in ("lowpowermode", "powermode"):
            info[parts[0]] = parts[1]
    for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if Path(prefix).exists():
            resolved = Path(prefix).resolve()
            info["libomp"] = {"prefix": prefix, "resolved": str(resolved), "version": resolved.name}
            break
    return info


def linux_environment() -> dict:
    info: dict = {}
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        for raw in os_release.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.startswith("PRETTY_NAME="):
                info["os_pretty_name"] = raw.split("=", 1)[1].strip().strip('"')
    keep = {"Architecture", "Vendor ID", "Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket",
            "Core(s) per cluster", "Socket(s)", "NUMA node(s)", "CPU max MHz", "L1d cache", "L2 cache",
            "L3 cache", "Hypervisor vendor"}
    fields: Dict[str, str] = {}
    for raw in (output_of(("lscpu",)) or "").splitlines():
        key, separator, value = raw.partition(":")
        if separator and key.strip() in keep:
            fields[key.strip()] = value.strip()
    info["lscpu"] = fields
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for raw in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.startswith("MemTotal:"):
                info["memory_gib"] = round(int(raw.split()[1]) / 2**20, 1)
    return info


def cmake_cache_values(build_dir: Path) -> Dict[str, str]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    values: Dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        values[key_and_type.split(":", 1)[0]] = value
    wanted = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
              "CMAKE_GENERATOR", "OpenMP_CXX_FLAGS", "OpenMP_CXX_LIB_NAMES", "OpenMP_CXX_SPEC_DATE",
              "OpenMP_omp_LIBRARY", "OpenMP_gomp_LIBRARY")
    return {key: values[key] for key in wanted if key in values}


def git_state() -> dict:
    def git(*arguments: str, strip: bool = True) -> Optional[str]:
        return output_of(("git", "-C", str(REPOSITORY_ROOT)) + arguments, strip=strip)

    top = git("rev-parse", "--show-toplevel")
    try:
        own_checkout = bool(top) and os.path.samefile(top, REPOSITORY_ROOT)
    except OSError:
        own_checkout = False
    if not own_checkout:
        return {}  # not a git checkout of this repository; never record an enclosing repository's state
    status = git("status", "--porcelain", "--untracked-files=no", strip=False) or ""
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_files_modified": bool(status.strip()),
        "status_porcelain": [line for line in status.splitlines() if line.strip()],
    }


def source_fingerprint() -> Dict[str, str]:
    """SHA-256 of every C++ source, so the exact measured code is identifiable."""
    paths = sorted(list((REPOSITORY_ROOT / "src").glob("*.[ch]pp"))
                   + list((REPOSITORY_ROOT / "benchmark").glob("*.cpp")))
    return {str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def read_metadata(executable: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in (output_of((str(executable), "--metadata")) or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            metadata[key] = value
    missing = {"compiler_id", "compiler_version", "build_type", "openmp_version"} - set(metadata)
    if missing:
        raise RuntimeError(f"{executable} --metadata did not report: {', '.join(sorted(missing))}")
    return metadata


def collect_environment(build_dir: Path, metadata: Dict[str, str], cmake: Optional[str]) -> dict:
    environment: dict = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "captured_at_local": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "logical_cores": os.cpu_count(),
        "load_average_1_5_15": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cmake_version": ((output_of((cmake, "--version")) or "").splitlines() or [None])[0] if cmake else None,
        "cmake_cache": cmake_cache_values(build_dir),
        "benchmark_metadata": metadata,
        "git": git_state(),
        "source_sha256": source_fingerprint(),
        "openmp_environment": {key: value for key, value in sorted(os.environ.items())
                               if key.startswith(("OMP_", "KMP_", "GOMP_"))},
    }
    compiler = environment["cmake_cache"].get("CMAKE_CXX_COMPILER")
    if compiler:
        banner = output_of((compiler, "--version"))
        environment["compiler_version_banner"] = banner.splitlines()[0] if banner else None
    if platform.system() == "Darwin":
        environment["macos"] = macos_environment()
    elif platform.system() == "Linux":
        environment["linux"] = linux_environment()
    return environment


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------


def run_correctness(ctest: str, build_dir: Path, build_type: str, tee: Tee, environment: Dict[str, str]) -> int:
    discovered = subprocess.run(
        [ctest, "--test-dir", str(build_dir), "--show-only=json-v1", "-L", "correctness"],
        check=True, capture_output=True, text=True, env=environment,
    )
    test_count = len(json.loads(discovered.stdout).get("tests", []))
    if test_count == 0:
        raise QualificationError("the build has no correctness-labelled tests; reconfigure it first")
    command = [ctest, "--test-dir", str(build_dir), "--output-on-failure", "-L", "correctness"]
    if build_type and build_type.lower() != "unknown":
        command += ["--build-config", build_type]
    tee.line(f"Correctness gate: {test_count} tests")
    if run_streaming(command, tee, environment) != 0:
        raise QualificationError("one or more correctness tests failed; nothing was measured")
    return test_count


# ---------------------------------------------------------------------------
# Timing matrix and end-to-end pass
# ---------------------------------------------------------------------------


def load_dataset_module():
    script = REPOSITORY_ROOT / "benchmark" / "prepare_dataset.py"
    specification = importlib.util.spec_from_file_location("prepare_dataset", script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def time_configuration(executable: Path, configuration: Config, warmups: int, runs: int,
                       images: Sequence[str]) -> List[float]:
    implementation, _, kernel, threads = configuration
    command = [str(executable), implementation, str(kernel), str(threads), str(warmups), str(runs), *images]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{quoted(command)} failed: {completed.stderr.strip()}")
    samples = [float(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(samples) != runs or any(not math.isfinite(s) or s <= 0.0 for s in samples):
        raise RuntimeError(f"{quoted(command)} printed invalid timings: {completed.stdout!r}")
    return samples


def matrix_configurations(sizes: Sequence[int], kernels: Sequence[int], threads: Sequence[int]) -> List[Config]:
    configurations: List[Config] = []
    for size in sizes:
        for kernel in kernels:
            configurations.append(("serial", size, kernel, 1))
            for implementation in ("basic", "optimized"):
                configurations.extend((implementation, size, kernel, p) for p in threads)
    return configurations


def summarize(samples: Dict[Config, List[float]], warmups: int) -> List[dict]:
    medians = {key: statistics.median(values) for key, values in samples.items()}
    rows = []
    for key, values in samples.items():
        implementation, size, kernel, threads = key
        ordered = sorted(values)
        median = medians[key]
        q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else (median,) * 3
        speedup = medians[(implementation, size, kernel, 1)] / median
        output = size - kernel + 1
        rows.append({
            "implementation": implementation,
            "image_size": size,
            "kernel": kernel,
            "threads": threads,
            "warmups": warmups,
            "runs": len(values),
            "median_ms": median,
            "q1_ms": q1,
            "q3_ms": q3,
            "mean_ms": statistics.fmean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "speedup": speedup,
            "speedup_vs_serial": medians[("serial", size, kernel, 1)] / median,
            "efficiency": speedup / threads,
            "gflops": 2.0 * kernel * kernel * output * output / (median * 1e6),
            "images_per_s": 1000.0 / median,
        })
    return rows


def end_to_end_configurations(sizes: Sequence[int], kernels: Sequence[int], threads: Sequence[int]) -> List[Config]:
    configurations: List[Config] = []

    def add(configuration: Config) -> None:
        if configuration not in configurations:
            configurations.append(configuration)

    p_max = max(threads)
    for size in sizes:
        for kernel in kernels:
            add(("serial", size, kernel, 1))
            add(("basic", size, kernel, p_max))
            add(("optimized", size, kernel, p_max))
    for kernel in kernels:
        for implementation in ("basic", "optimized"):
            for p in threads:
                add((implementation, max(sizes), kernel, p))
    return configurations


def end_to_end_pass(executable: Path, configurations: Sequence[Config], repeats: int,
                    first_image: Dict[int, str], tee: Tee) -> List[dict]:
    rows: List[dict] = []
    total = len(configurations) * repeats
    for repeat in range(1, repeats + 1):
        order = list(configurations)
        random.Random(E2E_SEED + repeat).shuffle(order)
        for implementation, size, kernel, threads in order:
            # One image, no warm-up and one timed call: the timed call is a cold call.
            command = [str(executable), implementation, str(kernel), str(threads), "0", "1", first_image[size]]
            start = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if completed.returncode != 0:
                raise RuntimeError(f"{quoted(command)} failed: {completed.stderr.strip()}")
            cold_ms = float([line for line in completed.stdout.splitlines() if line.strip()][-1])
            rows.append({"implementation": implementation, "image_size": size, "kernel": kernel,
                         "threads": threads, "repeat": repeat, "end_to_end_ms": elapsed_ms,
                         "cold_kernel_ms": cold_ms})
            if len(rows) % 25 == 0 or len(rows) == total:
                tee.line(f"[end-to-end {len(rows)}/{total}]")
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correctness-gated benchmark of the convolution implementations.")
    parser.add_argument("--build-dir", type=Path, default=REPOSITORY_ROOT / "build")
    parser.add_argument("--benchmark-executable", type=Path,
                        help="default: BUILD_DIR/convolution_benchmark")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("CONV_DATA_ROOT") or REPOSITORY_ROOT / "data"),
                        help="where datasets are downloaded and converted (default: $CONV_DATA_ROOT or data/)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--dataset", choices=("div2k", "imagenet-sample"),
                        help="image source (default: div2k; the smoke preset uses imagenet-sample)")
    parser.add_argument("--images", type=int, help="number of dataset photos (default: 10)")
    parser.add_argument("--sizes", help="image sizes N, e.g. 256,512,1024,2048,4096")
    parser.add_argument("--kernels", help="kernel sizes K, e.g. 3,5,7,11")
    parser.add_argument("--threads", help="OpenMP thread counts; default 1, 2 and every even count up to the cores")
    parser.add_argument("--warmups", type=int, help="untimed calls per configuration")
    parser.add_argument("--runs", type=int, help="timed calls per configuration")
    parser.add_argument("--e2e-kernels", help="kernel sizes for the end-to-end pass")
    parser.add_argument("--e2e-repeats", type=int, help="repeats of the end-to-end pass")
    parser.add_argument("--skip-e2e", action="store_true", help="skip the end-to-end pass")
    parser.add_argument("--ctest", help="path to ctest (auto-detected by default)")
    parser.add_argument("--allow-non-release", action="store_true", help="accept a non-Release build (tests only)")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    preset = PRESETS[args.preset]
    logical_cores = max(1, os.cpu_count() or 1)
    sizes = parse_int_list(args.sizes, "image size") if args.sizes else list(preset["sizes"])
    kernels = parse_int_list(args.kernels, "kernel size") if args.kernels else list(preset["kernels"])
    if args.threads:
        threads = parse_int_list(args.threads, "thread count")
    else:
        cap = preset["max_threads"] or logical_cores
        threads = default_threads(min(cap, logical_cores))
    warmups = preset["warmups"] if args.warmups is None else args.warmups
    runs = preset["runs"] if args.runs is None else args.runs
    e2e_kernels = parse_int_list(args.e2e_kernels, "end-to-end kernel size") if args.e2e_kernels \
        else [k for k in preset["e2e_kernels"] if k in kernels] or kernels[:1]
    e2e_repeats = preset["e2e_repeats"] if args.e2e_repeats is None else args.e2e_repeats
    dataset = args.dataset or preset["dataset"]
    image_count = preset["images"] if args.images is None else args.images
    if 1 not in threads:
        raise SystemExit("error: the thread list must include 1 (the speedup baseline)")
    if max(kernels + e2e_kernels) > min(sizes):
        raise SystemExit("error: every kernel must fit inside the smallest image")
    if warmups < 0 or runs < 1 or e2e_repeats < 1 or image_count < 1:
        raise SystemExit("error: need --warmups >= 0, --runs >= 1, --e2e-repeats >= 1 and --images >= 1")

    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    executable = (args.benchmark_executable or build_dir / "convolution_benchmark").resolve()
    if not executable.is_file():
        raise SystemExit(f"error: {executable} not found; build first (cmake --build build)")
    ctest = find_tool("ctest", args.ctest)
    cmake = shutil.which("cmake") or str(Path(ctest).with_name("cmake"))
    metadata = read_metadata(executable)
    if metadata["build_type"].lower() != "release" and not args.allow_non_release:
        raise SystemExit(f"error: benchmarks need a Release build, got {metadata['build_type']!r}")

    started = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(output_dir / "logs" / f"run_{started.strftime('%Y%m%dT%H%M%SZ')}.log")
    environment_vars = os.environ.copy()
    # Nested tests call `ctest` by name, so put the resolved CTest first on PATH.
    environment_vars["PATH"] = os.pathsep.join([str(Path(ctest).resolve().parent), environment_vars.get("PATH", "")])
    environment_vars["PYTHONUNBUFFERED"] = "1"
    caffeinate = None
    if platform.system() == "Darwin" and shutil.which("caffeinate"):
        caffeinate = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])  # no idle sleep

    configurations = matrix_configurations(sizes, kernels, threads)
    e2e_configurations = [] if args.skip_e2e else end_to_end_configurations(sizes, e2e_kernels, threads)
    phase_seconds: Dict[str, float] = {}
    tee.banner("Parallel 2-D convolution benchmark")
    tee.line(f"Dataset       : {dataset}, {image_count} images, in {args.data_root.resolve()}")
    tee.line(f"Image sizes N : {sizes}")
    tee.line(f"Kernel sizes K: {kernels} (Gaussian blur)")
    tee.line(f"Threads p     : {threads} (logical cores: {logical_cores})")
    tee.line(f"Repetitions   : {warmups} warm-up + {runs} timed calls per configuration")
    tee.line(f"Matrix        : {len(configurations)} configurations")
    tee.line(f"End-to-end    : {len(e2e_configurations)} configurations x {e2e_repeats} repeats")
    tee.line(f"Executable    : {executable} ({metadata['compiler_id']} {metadata['compiler_version']}, "
             f"{metadata['build_type']}, OpenMP {metadata['openmp_version']})")
    tee.line(f"Output        : {output_dir}")
    try:
        environment = collect_environment(build_dir, metadata, cmake)
        load = environment.get("load_average_1_5_15")
        if load and load[0] > 0.25 * logical_cores:
            tee.line(f"WARNING: 1-minute load average is {load[0]:.2f}; close busy apps for stable timings.")
        source = (environment.get("macos") or {}).get("power_source", "")
        if source and "AC Power" not in source:
            tee.line("WARNING: running on battery; plug in the charger for stable timings.")

        tee.banner("1/4 Correctness gate")
        phase_start = time.perf_counter()
        gate_tests = run_correctness(ctest, build_dir, metadata["build_type"], tee, environment_vars)
        phase_seconds["correctness_gate"] = time.perf_counter() - phase_start
        tee.line("Correctness PASS.")

        tee.banner("2/4 Dataset")
        phase_start = time.perf_counter()
        try:
            manifest = load_dataset_module().prepare(args.data_root, dataset, sizes, image_count, log=tee.line)
        except ImportError as error:
            raise RuntimeError(f"the dataset step needs Pillow (pip install -r requirements.txt): {error}") from error
        except RuntimeError:
            raise
        except Exception as error:  # noqa: BLE001 - download, checksum, archive or image errors
            raise RuntimeError(f"dataset preparation failed: {error}") from error
        image_ids = [entry["id"] for entry in manifest["images"]]
        image_paths = {int(size): paths for size, paths in manifest["paths"].items()}
        tee.line(f"Using {len(image_ids)} images ({image_ids[0]} .. {image_ids[-1]}) from {manifest['root']}")
        phase_seconds["dataset"] = time.perf_counter() - phase_start

        tee.banner("3/4 Timing matrix")
        phase_start = time.perf_counter()
        samples: Dict[Config, List[float]] = {}
        for index, configuration in enumerate(configurations, start=1):
            implementation, size, kernel, p = configuration
            tee.line(f"[{index}/{len(configurations)}] {implementation}: {size}x{size}, kernel {kernel}x{kernel}, "
                     f"{p} thread(s)")
            samples[configuration] = time_configuration(executable, configuration, warmups, runs,
                                                        image_paths[size])
        phase_seconds["matrix"] = time.perf_counter() - phase_start

        tee.banner("4/4 End-to-end pass")
        phase_start = time.perf_counter()
        first_image = {size: paths[0] for size, paths in image_paths.items()}
        e2e_rows = end_to_end_pass(executable, e2e_configurations, e2e_repeats, first_image, tee) \
            if e2e_configurations else []
        phase_seconds["end_to_end"] = time.perf_counter() - phase_start
    except (QualificationError, RuntimeError, subprocess.CalledProcessError) as error:
        tee.line("")
        tee.line(f"Benchmark CANCELLED: {error}" if isinstance(error, QualificationError) else f"FAILED: {error}")
        tee.line(f"Log: {tee.path}")
        tee.close()
        return 1
    finally:
        if caffeinate is not None:
            caffeinate.terminate()

    finished = datetime.now(timezone.utc)
    sample_rows = [
        {"implementation": key[0], "image_size": key[1], "kernel": key[2], "threads": key[3],
         "run": run, "image": image_ids[(run - 1) % len(image_ids)], "runtime_ms": value}
        for key, values in samples.items() for run, value in enumerate(values, start=1)
    ]
    dataset_record = {key: manifest[key] for key in ("source", "name", "url", "archive_sha256", "license",
                                                     "preprocessing") if key in manifest}
    dataset_record["images"] = [{key: entry.get(key) for key in ("id", "file", "sha256", "original_size")}
                                for entry in manifest["images"]]
    dataset_record["npy_pixel_sha256"] = {relative: digest for relative, digest in manifest["files"].items()
                                          if int(relative.split("/")[1]) in sizes and
                                          Path(relative).stem in image_ids}
    run_info = {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_seconds": (finished - started).total_seconds(),
        "phase_seconds": phase_seconds,
        "invocation": quoted([sys.executable] + sys.argv),
        "commands": {
            "timing_unit": quoted([str(executable), "IMPLEMENTATION", "K", "THREADS", str(warmups), str(runs),
                                   "IMAGE.npy", "..."]) + f" ({len(image_ids)} images of size N)",
            "end_to_end_unit": quoted([str(executable), "IMPLEMENTATION", "K", "THREADS", "0", "1", "IMAGE.npy"])
            + " (first image; timed as a whole process)",
        },
        "parameters": {
            "image_sizes": sizes,
            "kernel_sizes": kernels,
            "thread_counts": threads,
            "warmup_runs": warmups,
            "measured_runs": runs,
            "matrix_configurations": len(configurations),
            "correctness_tests": gate_tests,
            "end_to_end_kernels": [] if args.skip_e2e else e2e_kernels,
            "end_to_end_repeats": 0 if args.skip_e2e else e2e_repeats,
            "end_to_end_configurations": len(e2e_configurations),
            "end_to_end_shuffle_seed": E2E_SEED,
            "dataset": dataset_record,
            "image_order": "warm-up calls use the first image; timed call r uses image r mod the image count",
            "kernel": KERNEL_DESCRIPTION,
            "statistic": "median of the timed calls; speedup = T_impl(1) / T_impl(p); "
                         "speedup_vs_serial = T_serial / T_impl(p); efficiency = speedup / p",
        },
        "log": str(tee.path.relative_to(output_dir)),
    }
    # Publish atomically: every file is complete before any is replaced.
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=output_dir) as staging_name:
        staging = Path(staging_name)
        write_csv(staging / "samples.csv", SAMPLE_FIELDS, sample_rows)
        write_csv(staging / "benchmark_data.csv", SUMMARY_FIELDS, summarize(samples, warmups))
        if e2e_rows:
            write_csv(staging / "end_to_end.csv", E2E_FIELDS, e2e_rows)
        write_json(staging / "environment.json", environment)
        write_json(staging / "run_info.json", run_info)
        for item in sorted(staging.iterdir()):
            os.replace(item, output_dir / item.name)
    stale = output_dir / "end_to_end.csv"
    if not e2e_rows and stale.exists():
        stale.unlink()  # it belongs to an earlier run and would not match these samples
        tee.line("Removed end_to_end.csv from an earlier run (end-to-end pass skipped).")
    tee.banner(f"DONE in {run_info['wall_seconds'] / 60:.1f} min -> {output_dir}")
    tee.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; previous results were left untouched.", file=sys.stderr)
        sys.exit(130)
