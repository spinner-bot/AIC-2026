"""按 CAPR-CLIP V3 执行非破坏性数据审计与清洗。

原则：
  * 原始 ZIP 永不改写；可解码图片写入独立 cleaned 目录。
  * 可读截断图保留；不可读项记录到 anomaly.json，能读取的原始字节进入 quarantine。
  * 训练集重复/近重复图片不删除，只生成同组约束与固定划分。
  * 测试集只做完整性审计，不参与训练分组、类别统计或阈值计算。

示例：
    python scripts/clean_dataset.py ^
      --train-zip "D:\\AIC数据集\\train.zip" ^
      --test-zip "D:\\AIC数据集\\test.zip" ^
      --output-dir "D:\\AIC数据集\\cleaned" ^
      --expected-classes 500 --expected-train 103218 --expected-test 24967
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import imagehash
import pandas as pd
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.group_split import assign_head_mid_tail, build_groups, split_train_val

ImageFile.LOAD_TRUNCATED_IMAGES = True

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
_MANIFEST_COLUMNS = [
    "path",
    "archive_path",
    "orig_label",
    "class_idx",
    "width",
    "height",
    "aspect_ratio",
    "mode",
    "decodable",
    "file_size",
    "crc32",
    "file_hash",
    "phash",
    "error",
]


def _safe_archive_path(name: str) -> PurePosixPath:
    """规范化 ZIP 路径并拒绝绝对路径、盘符和 .. 路径穿越。"""
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"不安全的 ZIP 路径: {name!r}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, data: bytes) -> str:
    """原子写入；已有同内容文件复用，不覆盖内容不同的文件。"""
    if path.exists():
        if path.is_file() and path.stat().st_size == len(data):
            existing_hash = _sha256_file(path)
            if existing_hash == hashlib.sha256(data).hexdigest():
                return "reused"
        raise FileExistsError(f"目标已存在且内容不同，拒绝覆盖: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "written"


def _inspect_image(data: bytes, phash_size: int) -> dict[str, Any]:
    with Image.open(BytesIO(data)) as image:
        image.load()
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"非法图像尺寸: {width}x{height}")
        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 6),
            "mode": image.mode,
            "phash": str(imagehash.phash(image.convert("RGB"), hash_size=phash_size)),
        }


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    """由规范路径与文件 SHA256 生成与 ZIP 打包方式无关的数据指纹。"""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["archive_path"])):
        digest.update(str(row["archive_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("file_hash") or "").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _class_mapping(infos: list[zipfile.ZipInfo]) -> dict[str, int]:
    labels: set[str] = set()
    for info in infos:
        try:
            parts = _safe_archive_path(info.filename).parts
        except ValueError:
            continue
        if len(parts) == 2:
            labels.add(parts[0])
    return {label: index for index, label in enumerate(sorted(labels))}


def audit_archive(
    archive: Path,
    output_dir: Path,
    quarantine_dir: Path,
    kind: str,
    phash_size: int,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """流式校验、审计并提取一个训练或测试 ZIP。"""
    if kind not in {"train", "test"}:
        raise ValueError(f"未知数据类型: {kind}")
    if not archive.is_file():
        raise FileNotFoundError(f"ZIP 不存在: {archive}")

    rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    counters: Counter[str] = Counter()
    started = time.monotonic()

    with zipfile.ZipFile(archive, "r") as source:
        infos = [info for info in source.infolist() if not info.is_dir()]
        mapping = _class_mapping(infos) if kind == "train" else {}
        total = len(infos)
        print(f"[{kind}] {archive}：{total} 个文件", flush=True)

        for number, info in enumerate(infos, start=1):
            base: dict[str, Any] = {
                "path": "",
                "archive_path": info.filename.replace("\\", "/"),
                "orig_label": "",
                "class_idx": "",
                "width": "",
                "height": "",
                "aspect_ratio": "",
                "mode": "",
                "decodable": False,
                "file_size": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "file_hash": "",
                "phash": "",
                "error": "",
            }
            data: bytes | None = None
            relative: PurePosixPath | None = None

            try:
                relative = _safe_archive_path(info.filename)
                normalized = relative.as_posix()
                if normalized in seen_paths:
                    raise ValueError(f"ZIP 内路径重复: {normalized}")
                seen_paths.add(normalized)

                expected_parts = 2 if kind == "train" else 1
                if len(relative.parts) != expected_parts:
                    raise ValueError(
                        f"{kind} 路径层级应为 {expected_parts}，实际为 {len(relative.parts)}"
                    )
                if relative.suffix.lower() not in _IMAGE_EXTS:
                    raise ValueError(f"不支持的图片扩展名: {relative.suffix}")
                if info.flag_bits & 0x1:
                    raise ValueError("ZIP 条目已加密")
                if info.file_size <= 0:
                    raise ValueError("空文件")

                if kind == "train":
                    label = relative.parts[0]
                    base["orig_label"] = label
                    base["class_idx"] = mapping[label]

                data = source.read(info)
                if len(data) != info.file_size:
                    raise OSError(f"解压长度不符: {len(data)} != {info.file_size}")
                base["file_hash"] = hashlib.sha256(data).hexdigest()
                base.update(_inspect_image(data, phash_size))
                destination = output_dir.joinpath(*relative.parts)
                base["path"] = str(destination.resolve())
                counters[_write_once(destination, data)] += 1
                base["decodable"] = True
                counters["accepted"] += 1
            except Exception as error:
                base["error"] = f"{type(error).__name__}: {error}"
                counters["anomaly"] += 1
                if data is not None and relative is not None:
                    quarantine_path = (quarantine_dir / kind).joinpath(*relative.parts)
                    try:
                        _write_once(quarantine_path, data)
                        base["path"] = str(quarantine_path.resolve())
                    except Exception as quarantine_error:
                        base["error"] += (
                            f"; quarantine={type(quarantine_error).__name__}: {quarantine_error}"
                        )
                anomalies.append({"dataset": kind, **base})

            rows.append(base)
            if number % progress_every == 0 or number == total:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"[{kind}] {number}/{total} ({number / total:.1%})，"
                    f"可用 {counters['accepted']}，异常 {counters['anomaly']}，"
                    f"{number / elapsed:.1f} 张/秒",
                    flush=True,
                )

    summary = {
        "archive": str(archive.resolve()),
        "archive_size": archive.stat().st_size,
        "entries": len(rows),
        "accepted": counters["accepted"],
        "anomalies": counters["anomaly"],
        "written": counters["written"],
        "reused": counters["reused"],
        "dataset_sha256": _dataset_hash(rows),
        "classes": len({row["orig_label"] for row in rows if row["orig_label"] != ""}),
    }
    return rows, anomalies, summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _validate_expected(name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise RuntimeError(f"{name} 数量不符：期望 {expected}，实际 {actual}")


def clean_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    train_dir = output / "train"
    test_dir = output / "test"
    report_dir = output / "reports"
    quarantine_dir = output / "quarantine"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_anomalies, train_summary = audit_archive(
        args.train_zip.resolve(),
        train_dir,
        quarantine_dir,
        "train",
        args.phash_size,
        args.progress_every,
    )
    train_manifest = pd.DataFrame(train_rows, columns=_MANIFEST_COLUMNS)
    train_manifest.to_csv(report_dir / "train_manifest.csv", index=False)

    accepted_train = train_manifest[train_manifest["decodable"]].copy().reset_index(drop=True)
    print("[train] 构建全局重复/近重复 group…", flush=True)
    split = build_groups(accepted_train, near_dup_hamming=args.near_dup_hamming)
    split = split_train_val(
        split,
        val_ratio=args.val_ratio,
        min_samples_for_split=args.min_samples_for_split,
        oof_folds=args.oof_folds,
        seed=args.seed,
    )
    parts = assign_head_mid_tail(split, head_q=args.head_quantile, tail_q=args.tail_quantile)
    split["part"] = split["class_idx"].map(parts).fillna("middle")
    split.to_csv(report_dir / "split.csv", index=False)

    group_sizes = split.groupby("group_id").size()
    duplicate_ids = set(group_sizes[group_sizes > 1].index)
    duplicates = split[split["group_id"].isin(duplicate_ids)].copy()
    if not duplicates.empty:
        duplicates.insert(
            duplicates.columns.get_loc("group_id") + 1,
            "group_size",
            duplicates["group_id"].map(group_sizes),
        )
    duplicates.to_csv(report_dir / "duplicate_groups.csv", index=False)

    test_rows, test_anomalies, test_summary = audit_archive(
        args.test_zip.resolve(),
        test_dir,
        quarantine_dir,
        "test",
        args.phash_size,
        args.progress_every,
    )
    pd.DataFrame(test_rows, columns=_MANIFEST_COLUMNS).to_csv(
        report_dir / "test_manifest.csv", index=False
    )

    anomalies = train_anomalies + test_anomalies
    _write_json(report_dir / "anomaly.json", anomalies)

    train_summary.update(
        {
            "duplicate_groups": len(duplicate_ids),
            "duplicate_images": len(duplicates),
            "folds": {str(k): int(v) for k, v in split["fold"].value_counts().items()},
            "parts": {str(k): int(v) for k, v in split["part"].value_counts().items()},
        }
    )
    summary = {
        "schema_version": 1,
        "policy": "CAPR-CLIP V3 non-destructive audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parameters": {
            "seed": args.seed,
            "phash_size": args.phash_size,
            "near_dup_hamming": args.near_dup_hamming,
            "val_ratio": args.val_ratio,
            "min_samples_for_split": args.min_samples_for_split,
            "oof_folds": args.oof_folds,
            "head_quantile": args.head_quantile,
            "tail_quantile": args.tail_quantile,
        },
        "train": train_summary,
        "test": test_summary,
        "reports": {
            "train_manifest": str((report_dir / "train_manifest.csv").resolve()),
            "test_manifest": str((report_dir / "test_manifest.csv").resolve()),
            "split": str((report_dir / "split.csv").resolve()),
            "duplicates": str((report_dir / "duplicate_groups.csv").resolve()),
            "anomalies": str((report_dir / "anomaly.json").resolve()),
        },
    }
    _write_json(report_dir / "summary.json", summary)

    _validate_expected("训练集条目", train_summary["entries"], args.expected_train)
    _validate_expected("测试集条目", test_summary["entries"], args.expected_test)
    _validate_expected("训练集类别", train_summary["classes"], args.expected_classes)
    if train_summary["anomalies"] or test_summary["anomalies"]:
        print("[提示] 存在异常项，详情见 reports/anomaly.json；原 ZIP 未改动。", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAPR-CLIP V3 非破坏性数据清洗")
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--test-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-classes", type=int)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-test", type=int)
    parser.add_argument("--phash-size", type=int, default=8)
    parser.add_argument("--near-dup-hamming", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--min-samples-for-split", type=int, default=10)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--head-quantile", type=float, default=0.33)
    parser.add_argument("--tail-quantile", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    summary = clean_dataset(_parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
