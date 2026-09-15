"""训练入口。

用法:
    python src/train.py --config configs/baseline.yaml

说明:
    - 噪声筛选必须在本脚本内自动执行（赛题要求可复现，禁止人工清洗作为必要前置）
    - 训练流程含 OPSD warmup + λ 调度 + WiSE-FT
"""
import argparse


def main(config_path: str) -> None:
    # TODO:
    # 1. 加载 config（yaml）
    # 2. 构建 dataset / dataloader（含噪声筛选）
    # 3. 构建模型（CLIP ViT-B/32 + CoOp/LoRA）
    # 4. 构建损失（鲁棒损失 + OPSD 自蒸馏）
    # 5. 训练循环 + 保存 checkpoint
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算法4 训练入口")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
