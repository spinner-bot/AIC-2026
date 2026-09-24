"""推理入口。

用法:
    python src/inference.py --config configs/baseline.yaml --output pred_results.csv

说明:
    - 加载 checkpoint，对测试集推理
    - 输出 pred_results.csv（两列: 图片文件名, 类别编号，编号 4 位补零）
    - 最终压缩为 zip 提交
"""
import argparse


def main(config_path: str, output_path: str) -> None:
    # TODO:
    # 1. 加载 config + checkpoint
    # 2. 构建测试 dataloader
    # 3. 推理 -> 生成 pred_results.csv
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算法4 推理入口")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--output", required=True, help="输出 csv 路径")
    args = parser.parse_args()
    main(args.config, args.output)
