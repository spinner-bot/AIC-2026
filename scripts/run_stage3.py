"""Stage 3 入口：MVP 微调。

用法：
    python scripts/run_stage3.py --config configs/v3.yaml [--device cuda|cpu] [--epochs N]

前置：
    先运行 run_stage0.py（split.csv）、run_stage1.py（features.npy）、run_stage2.py（q0.csv）。

产物：
    {output_dir}/stage3.log   训练日志
    {output_dir}/best.pt      最优验证 checkpoint（仅可训练参数）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import load_config, set_seed
from src.capr.train_stage3 import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3：MVP 微调")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 robust_epochs（调试用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.experiment.get("seed", 42)))
    main(cfg, device=args.device, epochs=args.epochs)
