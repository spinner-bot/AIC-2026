"""Stage 1 冒烟测试：用假图片验证冻结特征提取 + L2 归一化 + 缓存。

验证 transformers 5.x 的 CLIP API 可用性，以及特征形状/归一化正确。

用法：
    D:\\anaconda\\envs\\aic\\python.exe scripts/smoke_stage1.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import Config, project_root
from src.capr.features import extract_features, save_features
from src.capr.manifest import build_manifest

_FAKE_DIR = "outputs/_smoke_data1"


def _make_fake_data(root: Path, n_classes: int = 3, per_class: int = 5) -> None:
    rng = np.random.default_rng(1)
    for c in range(n_classes):
        cdir = root / f"class_{c:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cdir / f"{i:04d}.png")


def _fake_config() -> Config:
    return Config(
        {
            "experiment": {"output_dir": "outputs/_smoke1", "seed": 42},
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
            "features": {"batch_size": 8, "cache_dir": None},
        },
        root=project_root(),
    )


def main() -> None:
    data_dir = project_root() / _FAKE_DIR
    if data_dir.exists():
        shutil.rmtree(data_dir)
    _make_fake_data(data_dir)

    cfg = _fake_config()
    manifest, _ = build_manifest(cfg)

    feats, subset = extract_features(cfg, manifest, device="cuda")
    print(f"feats shape: {feats.shape}  subset: {len(subset)}")

    # L2 归一化验证：每行范数应约为 1
    norms = (feats**2).sum(axis=1)
    print(f"L2 norm mean={norms.mean():.6f}  max_dev={np.abs(norms - 1).max():.2e}")

    assert feats.shape == (len(subset), 512), "特征形状不符"
    assert np.abs(norms - 1).max() < 1e-4, "L2 归一化不正确"

    fp, ip = save_features(cfg, feats, subset)
    print(f"features -> {fp.name}  index -> {ip.name}")
    print("STAGE1 SMOKE OK")


if __name__ == "__main__":
    main()
