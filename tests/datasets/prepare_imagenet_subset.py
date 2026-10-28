#!/usr/bin/env python3
"""Download a deterministic 20-image, one-class-per-image ImageNet subset."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


# Pin the source commit so the regression input cannot change underneath us.
SOURCE_COMMIT = "bc69043e2862758a1906c6a24a792bb28c6bb2ff"
BASE_URL = (
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/"
    f"{SOURCE_COMMIT}"
)
IMAGE_NAMES = (
    "n01440764_tench.JPEG",
    "n01443537_goldfish.JPEG",
    "n01484850_great_white_shark.JPEG",
    "n01491361_tiger_shark.JPEG",
    "n01494475_hammerhead.JPEG",
    "n01496331_electric_ray.JPEG",
    "n01498041_stingray.JPEG",
    "n01514668_cock.JPEG",
    "n01514859_hen.JPEG",
    "n01518878_ostrich.JPEG",
    "n01530575_brambling.JPEG",
    "n01531178_goldfinch.JPEG",
    "n01532829_house_finch.JPEG",
    "n01534433_junco.JPEG",
    "n01537544_indigo_bunting.JPEG",
    "n01558993_robin.JPEG",
    "n01560419_bulbul.JPEG",
    "n01580077_jay.JPEG",
    "n01582220_magpie.JPEG",
    "n01592084_chickadee.JPEG",
)


def is_valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except (ImportError, OSError):
        return False


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def prepare(data_root: Path) -> list[Path]:
    subset_dir = data_root / "imagenet-sample-images" / SOURCE_COMMIT
    subset_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for image_name in IMAGE_NAMES:
        image_path = subset_dir / image_name
        if not is_valid_image(image_path):
            image_path.unlink(missing_ok=True)
            temporary_path = subset_dir / f"{image_name}.part"
            temporary_path.unlink(missing_ok=True)
            print(f"Downloading {BASE_URL}/{image_name}")
            download(f"{BASE_URL}/{image_name}", temporary_path)
            if not is_valid_image(temporary_path):
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(f"downloaded invalid image: {image_name}")
            temporary_path.replace(image_path)
        paths.append(image_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    for path in prepare(args.data_root):
        print(path)


if __name__ == "__main__":
    main()
