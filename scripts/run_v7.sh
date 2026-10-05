#!/usr/bin/env bash
# V7（448 全量微调）一键启动：复用 v6 的 manifest/split/q0，重跑 stage1（448 特征），再训练。
#
# 用法（服务器上，v6 训练结束后）：
#     nohup bash scripts/run_v7.sh > train_v7.log 2>&1 &
#
# 说明：split/q0 只依赖文件 hash/phash 不依赖分辨率，且 v7 走 ce_plain 主线
# （anchor=0 / hard_drop=false），q0 不参与 loss，故可整表复用 v6；仅 features.npy
# 需按 448 重算（stage1）。

set -euo pipefail

cd "$(dirname "$0")/.."

PY=/root/miniconda3/envs/aic/bin/python
CFG=configs/v7_448_dup.yaml
V6=outputs/capr_v6_384_dup
V7=outputs/capr_v7_448_dup

echo "== V7 前置：复用 v6 的 manifest/split/q0 =="
mkdir -p "$V7"
for f in manifest.csv split.csv q0.csv; do
  if [ -f "$V6/$f" ]; then
    cp "$V6/$f" "$V7/$f"
    echo "  复制 $V6/$f -> $V7/$f"
  else
    echo "  [跳过] 缺失 $V6/$f"
  fi
done

echo "== Stage1：448 冻结特征提取 =="
"$PY" scripts/run_stage1.py --config "$CFG"

echo "== Stage3：V7 训练 =="
"$PY" scripts/run_stage3.py --config "$CFG"
