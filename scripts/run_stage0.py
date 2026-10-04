"""Stage 0 入口：数据审计 + 防泄漏 group split + 头中尾划分。

用法：
    python scripts/run_stage0.py --config configs/v3.yaml

产物：
    {output_dir}/manifest.csv   图片审计清单（路径/类别/尺寸/哈希/感知哈希/解码状态）
    {output_dir}/anomaly.json   无法解码图片清单
    {output_dir}/split.csv      带 group_id / fold / part 的划分结果
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import load_config, set_seed
from src.capr.group_split import run_group_split
from src.capr.manifest import build_manifest, save_manifest


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.experiment.get("seed", 42)))

    print("== Stage 0.1 数据审计 ==")
    manifest, anomalies = build_manifest(cfg)
    mp, ap = save_manifest(cfg, manifest, anomalies)
    print(f"  manifest -> {mp}")
    print(f"  anomaly  -> {ap} ({len(anomalies)} 张无法解码)")

    print("== Stage 0.2 group split ==")
    split = run_group_split(cfg, manifest)
    out_dir = cfg.path(cfg.experiment.output_dir)
    sp = out_dir / "split.csv"
    split.to_csv(sp, index=False)
    print(f"  split -> {sp}")

    print("== 统计 ==")
    print(f"  总样本: {len(split)}  类别数: {split['class_idx'].nunique()}")
    print(f"  划分分布:\n{split['fold'].value_counts().to_string()}")
    print(f"  头中尾分布:\n{split['part'].value_counts().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0：数据审计 + group split")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
