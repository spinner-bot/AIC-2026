"""Stage 1 入口：冻结 CLIP 特征缓存。

用法：
    python scripts/run_stage1.py --config configs/v3.yaml [--device cuda|cpu]

前置：
    先运行 run_stage0.py 生成 manifest.csv。

产物：
    {output_dir}/features/features.npy   L2 归一化冻结特征 [N, 512]
    {output_dir}/features/index.csv      特征与 manifest 的对齐索引
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import load_config, set_seed
from src.capr.features import extract_features, save_features


def main(config_path: str, device: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.experiment.get("seed", 42)))

    manifest_path = cfg.path(cfg.audit.get("manifest_path")) or cfg.path(cfg.experiment.output_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在，请先运行 run_stage0.py: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    print(f"== Stage 1 冻结特征缓存 ==")
    print(f"  manifest: {manifest_path} ({len(manifest)} 行)")

    feats, subset = extract_features(cfg, manifest, device=device)
    fp, ip = save_features(cfg, feats, subset)
    print(f"  features -> {fp}  形状 {feats.shape}")
    print(f"  index    -> {ip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1：冻结特征缓存")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="计算设备")
    args = parser.parse_args()
    main(args.config, args.device)
