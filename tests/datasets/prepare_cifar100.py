#!/usr/bin/env python3
"""Download, verify, and extract the official CIFAR-100 test batch."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


ARCHIVE_URLS = (
    "https://data.brainchip.com/dataset-mirror/cifar100/"
    "cifar-100-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
)
ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
TEST_BATCH_MD5 = "f0ef6b0ae62326f3e7ffdfab6717acfc"
ARCHIVE_NAME = "cifar-100-python.tar.gz"
EXTRACTED_NAME = "cifar-100-python"
TEST_MEMBER = f"{EXTRACTED_NAME}/test"


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
    test_batch = dataset_dir / "test"
    if test_batch.is_file() and file_md5(test_batch) == TEST_BATCH_MD5:
        return test_batch

    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / ARCHIVE_NAME
    if not archive.is_file() or file_md5(archive) != ARCHIVE_MD5:
        archive.unlink(missing_ok=True)
        partial_archive = data_root / f"{ARCHIVE_NAME}.part"
        errors = []
        for archive_url in ARCHIVE_URLS:
            try:
                print(f"Downloading {archive_url}")
                download_with_resume(archive_url, partial_archive)
                actual_md5 = file_md5(partial_archive)
                if actual_md5 != ARCHIVE_MD5:
                    raise RuntimeError(
                        "archive checksum mismatch: "
                        f"expected {ARCHIVE_MD5}, got {actual_md5}"
                    )
                partial_archive.replace(archive)
                break
            except Exception as error:
                errors.append(f"{archive_url}: {error}")
                partial_archive.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                "unable to prepare the verified CIFAR-100 archive:\n"
                + "\n".join(errors)
            )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    temporary_batch = dataset_dir / "test.part"
    temporary_batch.unlink(missing_ok=True)
    with tarfile.open(archive, "r:gz") as tar_stream:
        member = tar_stream.getmember(TEST_MEMBER)
        source = tar_stream.extractfile(member)
        if source is None:
            raise RuntimeError(f"archive member is not a file: {TEST_MEMBER}")
        with source, temporary_batch.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    actual_md5 = file_md5(temporary_batch)
    if actual_md5 != TEST_BATCH_MD5:
        temporary_batch.unlink(missing_ok=True)
        raise RuntimeError(
            "CIFAR-100 test-batch checksum mismatch: "
            f"expected {TEST_BATCH_MD5}, got {actual_md5}"
        )
    temporary_batch.replace(test_batch)
    return test_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(prepare(args.data_root))


if __name__ == "__main__":
    main()
