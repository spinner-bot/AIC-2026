"""Stage 2 入口：q0 可靠度诊断。

用法：
    python scripts/run_stage2.py --config configs/v3.yaml

前置：
    先运行 run_stage0.py（生成 split.csv）与 run_stage1.py（生成 features.npy）。

产物：
    {output_dir}/q0.csv        带 q0 列的 split（训练样本有值，val 为 NaN）
    {output_dir}/signals.npz   各信号原始值（诊断用）
    控制台打印：信号相关性 / q0 分布 / GMM 启用情况
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import load_config
from src.capr.reliability import compute_q0


def _print_report(split: pd.DataFrame, q0: np.ndarray, signals: dict, num_classes: int) -> None:
    train = split[split["fold"] != "val"].copy()
    train["q0"] = q0

    print("\n== 信号相关性 ==")
    names = ["cls", "proto", "knn", "gmm"]
    arrs = {n: signals[n] for n in names if signals.get(n) is not None}
    if arrs:
        corr = pd.DataFrame({a: [np.corrcoef(arrs[a], arrs[b])[0, 1] for b in arrs] for a in arrs}, index=list(arrs))
        print(corr.round(3).to_string())

    print("\n== q0 分位数 ==")
    print(pd.Series(q0).quantile([0, 0.25, 0.5, 0.75, 1.0]).round(3).to_string())

    print("\n== q0 按头中尾 ==")
    print(train.groupby("part")["q0"].agg(["mean", "std", "count"]).round(3).to_string())

    print("\n== GMM 状态 ==")
    print("启用" if signals.get("gmm") is not None else "未启用（回退百分位）")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = cfg.path(cfg.experiment.output_dir)

    split_path = out_dir / "split.csv"
    feat_dir = cfg.path(cfg.features.get("cache_dir")) or out_dir / "features"
    feat_path = feat_dir / "features.npy"

    if not split_path.exists():
        raise FileNotFoundError(f"split.csv 不存在，先运行 run_stage0.py: {split_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"features.npy 不存在，先运行 run_stage1.py: {feat_path}")

    split = pd.read_csv(split_path)
    feats = np.load(feat_path)
    assert len(split) == len(feats), f"split({len(split)}) 与 features({len(feats)}) 行数不一致"

    train_mask = split["fold"] != "val"
    train_labels = split.loc[train_mask, "class_idx"].values.astype(int)
    train_feats = feats[train_mask]
    num_classes = int(split["class_idx"].max()) + 1

    print(f"== Stage 2 q0 诊断 ==")
    print(f"训练样本 {len(train_labels)}，类别数 {num_classes}")

    q0, signals = compute_q0(
        train_feats, train_labels, num_classes, cfg.trust.to_dict(), seed=int(cfg.experiment.get("seed", 42))
    )

    # 写回 q0（训练样本有值，val 为 NaN）
    split["q0"] = np.nan
    split.loc[train_mask, "q0"] = q0
    q0_path = out_dir / "q0.csv"
    split.to_csv(q0_path, index=False)
    print(f"\nq0 -> {q0_path}")

    np.savez(out_dir / "signals.npz", **{k: v for k, v in signals.items() if v is not None})
    print(f"signals -> {out_dir / 'signals.npz'}")

    _print_report(split, q0, signals, num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2：q0 可靠度诊断")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
