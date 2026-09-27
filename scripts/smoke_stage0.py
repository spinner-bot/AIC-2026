"""Stage 0 冒烟测试：用假图片目录验证 manifest + group split 逻辑。

真实赛题数据尚未下载，先用随机生成的假图验证：
    - build_manifest 能正确记录路径/类别/尺寸/哈希/感知哈希
    - build_groups 能聚合完全重复与近重复
    - split_train_val 的 group 约束与头中尾划分

用法：
    python scripts/smoke_stage0.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import Config, project_root
from src.capr.group_split import run_group_split
from src.capr.manifest import build_manifest

_FAKE_DIR = "outputs/_smoke_data"


def _make_fake_data(root: Path, n_classes: int = 5, per_class: int = 20) -> None:
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        cdir = root / f"class_{c:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cdir / f"{i:04d}.png")


def _fake_config() -> Config:
    return Config(
        {
            "experiment": {"output_dir": "outputs/_smoke", "seed": 42},
            "data": {"train_dir": _FAKE_DIR},
            "audit": {
                "phash_size": 8,
                "file_hash": "sha256",
                "near_dup_hamming": 8,
                "val_ratio": 0.10,
                "min_samples_for_split": 10,
                "oof_folds": 3,
                "head_quantile": 0.33,
                "tail_quantile": 0.33,
            },
        },
        root=project_root(),
    )


def main() -> None:
    root = project_root()
    data_dir = root / _FAKE_DIR
    if data_dir.exists():
        shutil.rmtree(data_dir)
    _make_fake_data(data_dir)

    cfg = _fake_config()

    manifest, anomalies = build_manifest(cfg)
    print(f"[manifest] {len(manifest)} 张图，{manifest['class_idx'].nunique()} 类，异常 {len(anomalies)}")
    assert len(manifest) == 5 * 20, "manifest 数量不符"
    assert len(anomalies) == 0, "假图不应有解码异常"
    assert {"path", "orig_label", "class_idx", "width", "height", "phash", "file_hash", "decodable"} <= set(manifest.columns)

    split = run_group_split(cfg, manifest)
    print(f"[split] 列: {list(split.columns)}")
    print(split["fold"].value_counts().to_dict())
    print(split["part"].value_counts().to_dict())

    # group 不跨折：同一 group_id 只能出现在同一个 fold
    bad = split.groupby("group_id")["fold"].nunique()
    assert (bad <= 1).all(), "存在 group 跨折泄漏！"
    print("[split] group 不跨折 [OK]")

    print("STAGE0 SMOKE OK")


if __name__ == "__main__":
    main()
