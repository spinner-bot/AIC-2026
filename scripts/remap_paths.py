"""路径重映射：把 CSV 产物里的 Windows 绝对路径换成当前机器路径。

场景：本地（Windows）产出的 split.csv / q0.csv / manifest.csv 里，path 字段
是 ``D:\\初赛数据集\\cleaned\\train\\...`` 这种 Windows 绝对路径。搬到 AutoDL
（Linux）后这些路径失效，Stage 3 无法按 path 读图。本脚本重写 path 列，其余
列与行序完全不动（features.npy 按行对齐 split.csv，动行序会错位）。

两种模式（二选一）：

1) 按 archive_path 重建（推荐，避免反斜杠转义）：
    python scripts/remap_paths.py \
        --csv outputs/capr_clip_v3/split.csv \
               outputs/capr_clip_v3/q0.csv \
               outputs/capr_clip_v3/manifest.csv \
        --archive-root /root/autodl-tmp/cleaned/train

2) 旧前缀 → 新前缀替换：
    python scripts/remap_paths.py \
        --csv outputs/capr_clip_v3/split.csv \
        --old-prefix "D:\\初赛数据集\\cleaned\\train" \
        --new-prefix "/root/autodl-tmp/cleaned/train"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def remap_by_archive_path(csv_path: str, root: str) -> int:
    """path = root / archive_path（archive_path 是跨平台相对路径，如 0000/xxx.jpg）。"""
    df = pd.read_csv(csv_path)
    for col in ("path", "archive_path"):
        if col not in df.columns:
            raise KeyError(f"{csv_path} 缺少 {col} 列，无法按 archive_path 重建 path")
    base = Path(root)
    df["path"] = [str(base / ap) for ap in df["archive_path"].astype(str)]
    df.to_csv(csv_path, index=False)
    return len(df)


def remap_by_prefix(csv_path: str, old_prefix: str, new_prefix: str) -> int:
    """把 path 列里以 old_prefix 开头的值替换为新前缀。"""
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise KeyError(f"{csv_path} 缺少 path 列")
    col = df["path"].astype(str)
    hits = int(col.str.startswith(old_prefix).sum())
    df["path"] = col.str.replace(old_prefix, new_prefix, regex=False)
    df.to_csv(csv_path, index=False)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="重写 CSV 产物的 path 前缀（本地 Windows → 目标机器）")
    ap.add_argument("--csv", nargs="+", required=True, help="一个或多个 CSV 产物路径")
    ap.add_argument("--archive-root", help="按 archive_path 重建 path 的根目录（推荐）")
    ap.add_argument("--old-prefix", help="旧路径前缀（与 --new-prefix 一起用）")
    ap.add_argument("--new-prefix", help="新路径前缀（与 --old-prefix 一起用）")
    args = ap.parse_args()

    use_archive = args.archive_root is not None
    use_prefix = args.old_prefix is not None and args.new_prefix is not None
    if use_archive == use_prefix:
        ap.error("必须且只能选一种：--archive-root，或 --old-prefix + --new-prefix")

    for p in args.csv:
        if use_archive:
            n = remap_by_archive_path(p, args.archive_root)
            print(f"  {p}: 按 archive_path 重建 {n} 行 path")
        else:
            n = remap_by_prefix(p, args.old_prefix, args.new_prefix)
            print(f"  {p}: 改写 {n} 行 path 前缀")


if __name__ == "__main__":
    main()
