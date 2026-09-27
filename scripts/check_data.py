"""数据完整性检查：确认下载的数据结构正确、图片可读。

用法：
    python scripts/check_data.py --config configs/v3.yaml

检查项：
    - data/train 目录是否存在、类别文件夹数量
    - 图片总数
    - 抽样检查图片能否被 Pillow 读取（含截断图）
    - 测试列表是否存在
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import load_config

ImageFile.LOAD_TRUNCATED_IMAGES = True

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
_SAMPLE_PER_CLASS = 5


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    train_dir = cfg.path(cfg.data.train_dir)

    if train_dir is None or not train_dir.is_dir():
        print(f"[错误] 训练集目录不存在: {cfg.data.train_dir}")
        print(f"请把下载的数据解压到 {train_dir}，结构为 train/类别文件夹/图片（见 DATA.md）")
        sys.exit(1)

    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    n_classes = len(class_dirs)
    total = 0
    unreadable = 0
    for cdir in class_dirs:
        imgs = [p for p in cdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        total += len(imgs)
        for p in imgs[:_SAMPLE_PER_CLASS]:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                unreadable += 1

    print(f"训练集目录: {train_dir}")
    print(f"类别数: {n_classes}")
    print(f"图片总数: {total}")
    print(f"抽样不可读: {unreadable}")

    expected = int(cfg.data.num_classes)
    if n_classes != expected:
        print(f"[警告] 类别数 {n_classes} 与配置 num_classes={expected} 不一致，请确认当前阶段/配置")
    else:
        print(f"[OK] 类别数与配置一致 ({n_classes})")

    test_list = cfg.path(cfg.data.test_list)
    if test_list and test_list.exists():
        n_lines = sum(1 for _ in test_list.open("r", encoding="utf-8", errors="ignore"))
        print(f"测试列表: {test_list} ({n_lines} 行)")
    else:
        print(f"[提示] 测试列表不存在: {cfg.data.test_list}（初赛可能由官方另发或走 val 划分）")

    print("检查完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据完整性检查")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
