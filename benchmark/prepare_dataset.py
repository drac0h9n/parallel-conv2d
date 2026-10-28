#!/usr/bin/env python3
"""Download and preprocess the image dataset used by the performance benchmark.

    python3 benchmark/prepare_dataset.py                        # DIV2K, 10 images, N = 256..4096
    python3 benchmark/prepare_dataset.py --images 20 --sizes 512,1024

Default source: the DIV2K validation set, 100 high-resolution (2K) photos from
the NTIRE 2017 super-resolution challenge, https://data.vision.ee.ethz.ch/cvl/DIV2K/
(made available for academic research purposes only).

Steps
  1. Download DIV2K_valid_HR.zip (448,993,893 bytes) into DATA_ROOT/div2k/,
     resuming partial downloads, and verify its published SHA-256.
  2. Extract the first IMAGES photos in file-name order (0801.png, 0802.png, ...).
  3. For every size N: convert to 8-bit grayscale (ITU-R BT.601 luma, Pillow
     mode "L"), crop the central square and resize it to N x N with a Lanczos
     filter (upscaling when N exceeds the square), then save it as
     DATA_ROOT/div2k/npy/N/<id>.npy (uint8, row-major). The timing harness
     divides by 255 to get doubles in [0, 1].
  4. Write DATA_ROOT/div2k/manifest.json with the source, checksums and settings.

Work that is already done is reused. The source "imagenet-sample" applies the
same preprocessing to the 20 ImageNet photos the correctness tests download;
the benchmark smoke test uses it so the test suite never needs the DIV2K archive.

Requires Pillow (pip install -r requirements.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DIV2K_BYTES = 448_993_893
DIV2K_SHA256 = "20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc"
DEFAULT_SIZES = (256, 512, 1024, 2048, 4096)
DEFAULT_IMAGES = 10
FORMAT_VERSION = 1  # bump whenever the preprocessing changes, so caches are rebuilt
PREPROCESSING = (
    "8-bit grayscale (ITU-R BT.601 luma, Pillow mode L), central square crop, Lanczos resize to N x N "
    "(upscaling when N exceeds the square); stored as uint8 .npy and scaled to [0, 1] by the harness"
)
SOURCES = ("div2k", "imagenet-sample")

Log = Callable[[str], None]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path, expected_bytes: int, log: Log) -> None:
    done = destination.stat().st_size if destination.exists() else 0
    if done >= expected_bytes:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "parallel-conv2d-benchmark"})
    if done:
        request.add_header("Range", f"bytes={done}-")
    log(f"Downloading {url} ({expected_bytes / 2**20:.0f} MiB)" + (f", resuming at {done / 2**20:.0f} MiB" if done else ""))
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = done and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        written = done if resumed else 0
        next_report = written + 50 * 2**20
        with destination.open(mode) as output:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                output.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    log(f"  {written / 2**20:.0f} / {expected_bytes / 2**20:.0f} MiB")
                    next_report += 50 * 2**20


def fetch_div2k_archive(directory: Path, log: Log) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "DIV2K_valid_HR.zip"
    if archive.is_file() and archive.stat().st_size == DIV2K_BYTES and sha256_of(archive) == DIV2K_SHA256:
        return archive
    partial = directory / "DIV2K_valid_HR.zip.part"
    download_with_resume(DIV2K_URL, partial, DIV2K_BYTES, log)
    actual = sha256_of(partial)
    if actual != DIV2K_SHA256:
        partial.unlink()
        raise RuntimeError(f"DIV2K archive checksum mismatch: expected {DIV2K_SHA256}, got {actual}")
    partial.replace(archive)
    return archive


def extract_div2k(archive: Path, destination: Path, count: int, log: Log) -> List[Path]:
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            (info for info in bundle.infolist() if re.search(r"(^|/)\d{4}\.png$", info.filename)),
            key=lambda info: Path(info.filename).name,
        )
        if len(members) < count:
            raise RuntimeError(f"{archive.name} holds {len(members)} images; {count} requested")
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for info in members[:count]:
            target = destination / Path(info.filename).name
            if not target.is_file() or target.stat().st_size != info.file_size:
                log(f"Extracting {target.name}")
                temporary = target.with_name(target.name + ".part")
                with bundle.open(info) as source, temporary.open("wb") as output:  # CRC-checked
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
            paths.append(target)
    return paths


def imagenet_sample_images(data_root: Path, count: int) -> List[Path]:
    script = REPOSITORY_ROOT / "tests" / "datasets" / "prepare_imagenet_subset.py"
    specification = importlib.util.spec_from_file_location("bench_prepare_imagenet", script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    paths = list(module.prepare(data_root))
    if len(paths) < count:
        raise RuntimeError(f"the ImageNet sample holds {len(paths)} images; {count} requested")
    return paths[:count]


def write_npy_uint8(path: Path, rows: int, cols: int, pixels: bytes) -> None:
    """Write a C-order uint8 array in NumPy .npy format 1.0."""
    header = f"{{'descr': '|u1', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    padding = 64 - (10 + len(header) + 1) % 64
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        stream.write(b"\x93NUMPY\x01\x00")
        stream.write(len(header_bytes).to_bytes(2, "little"))
        stream.write(header_bytes)
        stream.write(pixels)
    temporary.replace(path)


def to_square_grayscale(image_path: Path, size: int):
    from PIL import Image  # imported here so --help works without Pillow

    lanczos = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(image_path) as source:
        width, height = source.size
        gray = source.convert("L")
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    square = gray.crop((left, top, left + side, top + side))
    resized = square if side == size else square.resize((size, size), lanczos)
    return resized.tobytes(), (width, height), side


def prepare(data_root: Path, source: str = "div2k", sizes: Sequence[int] = DEFAULT_SIZES,
            count: int = DEFAULT_IMAGES, log: Log = print) -> dict:
    """Make sure every (image, size) .npy exists; return the dataset manifest."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {', '.join(SOURCES)}")
    if count <= 0 or any(size <= 0 for size in sizes):
        raise ValueError("image count and sizes must be positive")
    root = data_root.resolve() / source
    if source == "div2k":
        archive = fetch_div2k_archive(root, log)
        originals = extract_div2k(archive, root / "images", count, log)
        origin = {"name": "DIV2K validation set (high-resolution photos)", "url": DIV2K_URL,
                  "archive_sha256": DIV2K_SHA256,
                  "license": "made available for academic research purposes only"}
    else:
        originals = imagenet_sample_images(data_root.resolve(), count)
        origin = {"name": "ImageNet sample images (correctness-test subset)",
                  "url": "https://github.com/EliSchwartz/imagenet-sample-images"}

    manifest_path = root / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    reusable = previous.get("format_version") == FORMAT_VERSION
    known = {entry["id"]: entry for entry in previous.get("images", [])} if reusable else {}
    previous_files = previous.get("files", {}) if reusable else {}

    images, files = [], {}
    for original in originals:
        image_id = original.stem
        original_sha = sha256_of(original)
        same_source = known.get(image_id, {}).get("sha256") == original_sha
        entry = {"id": image_id, "file": original.name, "sha256": original_sha}
        for size in sizes:
            relative = f"npy/{size}/{image_id}.npy"
            target = root / relative
            if same_source and target.is_file() and relative in previous_files:
                files[relative] = previous_files[relative]
                entry.setdefault("original_size", known[image_id].get("original_size"))
                entry.setdefault("square_side", known[image_id].get("square_side"))
                continue
            pixels, original_size, side = to_square_grayscale(original, size)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_npy_uint8(target, size, size, pixels)
            files[relative] = hashlib.sha256(pixels).hexdigest()
            entry["original_size"], entry["square_side"] = list(original_size), side
            log(f"Prepared {relative} from {original.name} ({original_size[0]}x{original_size[1]})")
        images.append(entry)

    manifest = {
        "format_version": FORMAT_VERSION,
        "source": source,
        **origin,
        "preprocessing": PREPROCESSING,
        "images": images,
        "sizes": sorted(set(int(size) for size in sizes) | set(int(s) for s in previous.get("sizes", []) if reusable)),
        "files": {**previous_files, **files},
        "pixel_sha256_note": "files maps each .npy to the SHA-256 of its raw pixel bytes",
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["root"] = str(root)
    manifest["paths"] = {int(size): [str(root / f"npy/{size}/{entry['id']}.npy") for entry in images]
                         for size in sizes}
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the benchmark image dataset.")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("CONV_DATA_ROOT") or REPOSITORY_ROOT / "data"),
                        help="default: $CONV_DATA_ROOT or data/")
    parser.add_argument("--source", choices=SOURCES, default="div2k")
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)), help="image sizes N")
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES, help="number of photos to use")
    args = parser.parse_args()
    sizes = [int(token) for token in args.sizes.split(",") if token.strip()]
    try:
        manifest = prepare(args.data_root, args.source, sizes, args.images)
    except ImportError:
        sys.exit("error: Pillow is required (pip install -r requirements.txt)")
    ids = [entry["id"] for entry in manifest["images"]]
    print(f"Ready: {len(ids)} images ({ids[0]}..{ids[-1]}) at N = {sizes} in {manifest['root']}")


if __name__ == "__main__":
    main()
