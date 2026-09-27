"""Validate and deduplicate the AIC image classification archives.

Run this file from any directory. Input and output default to the directory
containing this script. Original train.zip and test.zip are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError


ImageFile.LOAD_TRUNCATED_IMAGES = False
LABEL_PATTERN = re.compile(r"[0-9]{4}")
PROGRESS_INTERVAL = 5000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "MPO", "BMP"}


@dataclass
class Record:
    index: int
    name: str
    label: str | None
    digest: str | None = None
    reason: str | None = None
    image_format: str | None = None
    needs_conversion: bool = False


def check_name(name: str, split: str) -> tuple[str | None, str | None]:
    """Return (label, rejection reason) for one archive member."""
    if name.startswith("/") or "\\" in name or "\x00" in name:
        return None, "unsafe_path"
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None, "unsafe_path"
    if not name.lower().endswith(".jpg"):
        return None, "non_jpeg_file"
    if split == "train":
        if len(parts) != 2 or not LABEL_PATTERN.fullmatch(parts[0]):
            return None, "invalid_train_path"
        return parts[0], None
    if len(parts) != 1:
        return None, "invalid_test_path"
    return None, None


def scan_archive(path: Path, split: str) -> tuple[list[Record], Counter]:
    records: list[Record] = []
    modes: Counter = Counter()
    seen_paths: set[str] = set()
    start = time.monotonic()

    with zipfile.ZipFile(path, "r") as archive:
        for index, info in enumerate(archive.infolist()):
            if info.is_dir():
                continue
            label, reason = check_name(info.filename, split)
            record = Record(index=index, name=info.filename, label=label, reason=reason)
            records.append(record)
            if reason:
                continue
            if info.filename in seen_paths:
                record.reason = "duplicate_path"
                continue
            if info.file_size == 0:
                record.reason = "empty_file"
                continue

            try:
                with archive.open(info, "r") as member:
                    data = member.read()  # ZIP CRC is checked when the member is read.
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in SUPPORTED_FORMATS:
                        record.reason = "unsupported_image_format"
                        continue
                    image.load()  # Decode every pixel, including the end of the file.
                    mode = image.mode
                    record.image_format = image.format
                    record.needs_conversion = image.format != "JPEG" or mode != "RGB"
            except (OSError, ValueError, EOFError, RuntimeError,
                    zipfile.BadZipFile, UnidentifiedImageError):
                record.reason = "unreadable_image_or_zip_member"
                continue

            record.digest = hashlib.sha256(data).hexdigest()
            seen_paths.add(info.filename)
            modes[mode] += 1
            if len(records) % PROGRESS_INTERVAL == 0:
                elapsed = time.monotonic() - start
                print(f"{split}: checked {len(records):,} images in {elapsed:.0f}s", flush=True)

    return records, modes


def mark_duplicates(train: list[Record], test: list[Record]) -> None:
    test_hashes: set[str] = set()
    for record in test:
        if record.reason:
            continue
        assert record.digest is not None
        if record.digest in test_hashes:
            record.reason = "duplicate_in_test"
        else:
            test_hashes.add(record.digest)

    train_groups: dict[str, list[Record]] = defaultdict(list)
    for record in train:
        if not record.reason:
            assert record.digest is not None
            train_groups[record.digest].append(record)

    for digest, group in train_groups.items():
        labels = {record.label for record in group}
        if len(labels) > 1:
            for record in group:
                record.reason = "conflicting_train_labels"
        elif digest in test_hashes:
            for record in group:
                record.reason = "identical_image_in_test"
        else:
            for record in group[1:]:
                record.reason = "duplicate_in_train"


def convert_to_jpeg(data: bytes) -> bytes:
    """Convert the first image frame to an RGB JPEG on a white background."""
    with Image.open(io.BytesIO(data)) as image:
        image.seek(0)
        image.load()
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = image.convert("RGB")
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=95, subsampling=0)
        return output.getvalue()


def write_archive(source: Path, destination: Path, records: list[Record]) -> None:
    keep = {record.index: record for record in records if record.reason is None}
    start = time.monotonic()
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as cleaned:
        for index, info in enumerate(original.infolist()):
            record = keep.get(index)
            if record is None:
                continue
            with original.open(info, "r") as member:
                data = member.read()
            if record.needs_conversion:
                data = convert_to_jpeg(data)
            # JPEG is compressed already; ZIP_STORED avoids redundant compression.
            cleaned.writestr(info.filename, data, compress_type=zipfile.ZIP_STORED)
            if (index + 1) % PROGRESS_INTERVAL == 0:
                elapsed = time.monotonic() - start
                print(f"{source.stem}: saved {index + 1:,} entries in {elapsed:.0f}s", flush=True)


def make_report(train: list[Record], test: list[Record], modes: dict) -> dict:
    def summary(records: list[Record]) -> dict:
        reasons = Counter(record.reason for record in records if record.reason)
        return {
            "input_images": len(records),
            "kept_images": sum(record.reason is None for record in records),
            "removed_images": sum(reasons.values()),
            "removed_by_reason": dict(sorted(reasons.items())),
        }

    before = Counter(record.label for record in train if record.label is not None)
    after = Counter(record.label for record in train if record.label is not None and record.reason is None)
    converted = {
        split: dict(sorted(Counter(record.image_format for record in records
                                   if record.reason is None and record.needs_conversion).items()))
        for split, records in (("train", train), ("test", test))
    }
    return {
        "method": "Full image decode and ZIP CRC check; convert non-RGB JPEG and mislabeled supported image formats to RGB JPEG; remove invalid files, byte-identical duplicates, label conflicts, and train/test byte-identical overlap.",
        "notes": [
            "Original archives are unchanged. Already-RGB JPEG files keep their original bytes.",
            "Converted images use JPEG quality 95; transparency is placed on white, and only the first frame of animated images is kept.",
            "Duplicate detection uses SHA-256 of original bytes; visually similar or differently encoded images are not detected.",
            "No labels or statistics are learned from test images.",
        ],
        "train": summary(train),
        "test": summary(test),
        "train_class_count_before": dict(sorted(before.items())),
        "train_class_count_after": dict(sorted(after.items())),
        "image_modes": {name: dict(sorted(counts.items())) for name, counts in modes.items()},
        "converted_by_original_format": converted,
    }


def write_removed_csv(path: Path, train: list[Record], test: list[Record]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "archive_path", "reason"])
        for split, records in (("train", train), ("test", test)):
            for record in records:
                if record.reason:
                    writer.writerow([split, record.name, record.reason])


def temporary_path(directory: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    os.close(descriptor)
    return Path(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cleaned archives")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        parser.error(f"Output directory does not exist: {output_dir}")

    sources = {split: input_dir / f"{split}.zip" for split in ("train", "test")}
    outputs = {split: output_dir / f"{split}_clean.zip" for split in sources}
    for source in sources.values():
        if not source.is_file():
            parser.error(f"Missing input archive: {source}")
    for output in outputs.values():
        if output.exists() and not args.overwrite:
            parser.error(f"Output exists: {output}. Pass --overwrite to replace it.")

    print("Checking test.zip first, then train.zip...", flush=True)
    test, test_modes = scan_archive(sources["test"], "test")
    train, train_modes = scan_archive(sources["train"], "train")
    mark_duplicates(train, test)
    report = make_report(train, test, {"train": train_modes, "test": test_modes})
    for split in ("train", "test"):
        print(f"{split}: {report[split]['kept_images']:,} kept, "
              f"{report[split]['removed_images']:,} removed", flush=True)

    staged: dict[str, Path] = {}
    try:
        for split, records in (("train", train), ("test", test)):
            staged[split] = temporary_path(output_dir, f"{split}_clean_")
            print(f"Writing {outputs[split].name}...", flush=True)
            write_archive(sources[split], staged[split], records)

        for split in ("train", "test"):
            os.replace(staged[split], outputs[split])
        report_path = output_dir / "cleaning_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_removed_csv(output_dir / "removed_files.csv", train, test)
        print(f"Done. Report: {report_path}", flush=True)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
