"""Stage 0：数据审计与 manifest 生成。

职责（对应 V3 3.1）：
    - 扫描按类别文件夹组织的含噪训练集
    - 记录路径、原始类别、内部类别索引、尺寸、长宽比、解码状态、文件哈希、感知哈希
    - 无法读取的图像进入异常清单（不人工修改）
    - 可读取的截断图保留（Pillow LOAD_TRUNCATED_IMAGES）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image, ImageFile

from .config import Config

# 允许 Pillow 读取截断图（官方说明中存在此类图，能读则保留）
ImageFile.LOAD_TRUNCATED_IMAGES = True

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def _hash_file(path: Path, algo: str = "sha256") -> str:
    """分块读取文件计算哈希，避免一次性读入大文件。"""
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _phash(img: Image.Image, hash_size: int = 8) -> str:
    """感知哈希（64-bit，十六进制字符串）。phash 要求 RGB 输入。"""
    return str(imagehash.phash(img.convert("RGB"), hash_size=hash_size))


def build_manifest(cfg: Config) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """生成 manifest DataFrame 与异常清单。

    返回 (manifest, anomalies)。manifest 列：
        path, orig_label, class_idx, width, height, aspect_ratio,
        decodable, file_hash, phash
    """
    train_dir = cfg.path(cfg.data.train_dir)
    if train_dir is None or not train_dir.is_dir():
        raise FileNotFoundError(f"训练集目录不存在: {cfg.data.train_dir}")

    phash_size = int(cfg.audit.get("phash_size", 8))
    hash_algo = cfg.audit.get("file_hash", "sha256")

    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(f"训练集目录下没有类别文件夹: {train_dir}")

    rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for cidx, cdir in enumerate(class_dirs):
        for img_path in sorted(cdir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            rec: dict[str, Any] = {
                "path": str(img_path),
                "orig_label": cdir.name,
                "class_idx": cidx,
            }
            try:
                file_hash = _hash_file(img_path, hash_algo)
                with Image.open(img_path) as im:
                    w, h = im.size
                    ph = _phash(im, hash_size=phash_size)
                rec.update(
                    {
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(w / h, 6) if h else None,
                        "decodable": True,
                        "file_hash": file_hash,
                        "phash": ph,
                    }
                )
                rows.append(rec)
            except Exception as e:  # noqa: BLE001 —— 任何解码异常都进异常清单
                rec.update({"decodable": False, "error": f"{type(e).__name__}: {e}"})
                anomalies.append(rec)

    manifest = pd.DataFrame(rows)
    return manifest, anomalies


def save_manifest(
    cfg: Config, manifest: pd.DataFrame, anomalies: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """落盘 manifest.csv 与 anomaly.json，返回两个文件路径。"""
    out_dir = cfg.path(cfg.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cfg.path(cfg.audit.get("manifest_path")) or out_dir / "manifest.csv"
    anomaly_path = cfg.path(cfg.audit.get("anomaly_path")) or out_dir / "anomaly.json"

    manifest.to_csv(manifest_path, index=False)
    with anomaly_path.open("w", encoding="utf-8") as f:
        json.dump(anomalies, f, ensure_ascii=False, indent=2)

    return manifest_path, anomaly_path
