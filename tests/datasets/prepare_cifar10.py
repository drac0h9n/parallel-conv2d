#!/usr/bin/env python3
"""Download and verify the CIFAR-10 Python test batch."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


# Download only the test batch the tests use instead of the full 163 MB archive.
# The checksum is the one published by Torchvision for the official test batch.
TEST_BATCH_URL = (
    "https://github.com/0Colin/cifar-10-batches-py/"
    "raw/refs/heads/master/test_batch"
)
TEST_BATCH_MD5 = "40351d587109b95175f43aff81a1287e"
EXTRACTED_NAME = "cifar-10-batches-py"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path) -> None:
    downloaded_bytes = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url)
    if downloaded_bytes:
        request.add_header("Range", f"bytes={downloaded_bytes}-")

    with urllib.request.urlopen(request, timeout=60) as response:
        is_partial_response = getattr(response, "status", None) == 206
        output_mode = "ab" if downloaded_bytes and is_partial_response else "wb"
        with destination.open(output_mode) as output:
            shutil.copyfileobj(response, output)


def prepare(data_root: Path) -> Path:
    dataset_dir = data_root / EXTRACTED_NAME
    test_batch = dataset_dir / "test_batch"
    if test_batch.is_file() and file_md5(test_batch) == TEST_BATCH_MD5:
        return test_batch

    dataset_dir.mkdir(parents=True, exist_ok=True)
    test_batch.unlink(missing_ok=True)
    temporary_path = dataset_dir / "test_batch.part"
    if not temporary_path.is_file() or file_md5(temporary_path) != TEST_BATCH_MD5:
        print(f"Downloading {TEST_BATCH_URL}")
        download_with_resume(TEST_BATCH_URL, temporary_path)

    actual_md5 = file_md5(temporary_path)
    if actual_md5 != TEST_BATCH_MD5:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "CIFAR-10 test-batch checksum mismatch: "
            f"expected {TEST_BATCH_MD5}, got {actual_md5}"
        )
    temporary_path.replace(test_batch)
    return test_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(prepare(args.data_root))


if __name__ == "__main__":
    main()
