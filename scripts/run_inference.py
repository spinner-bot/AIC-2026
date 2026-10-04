"""Stage 6 推理：用训练好的 checkpoint 在测试集上生成提交文件。

用法：
    python scripts/run_inference.py --config configs/v4_fullft.yaml \
        --checkpoint outputs/capr_v4_fullft/best.pt \
        --test-dir /root/autodl-tmp/cleaned/test \
        --output pred_results.csv [--batch 128]

产物（写入 checkpoint 所在目录或 --output 指定路径）：
    pred_results.csv   每行 `文件名,四位类别编号`（无表头，与赛题示例一致）
    pred_results.zip   压缩包（提交格式）

约束：单模型、单裁剪、单次前向（不 TTA / 不集成）。
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import load_config, set_seed
from src.capr.model import build_model, load_checkpoint

_CLIP_MODEL_DIR = "models/clip-vit-base-patch32"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class _TestDataset(Dataset):
    """按文件路径批量读取测试图（中心裁剪，与验证口径一致）。"""

    def __init__(self, paths: list[str], processor: CLIPProcessor):
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.paths[i]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def list_test_images(test_dir: str) -> list[str]:
    """列出测试集所有图片（平铺目录），按文件名排序保证确定性顺序。"""
    p = Path(test_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"测试目录不存在: {test_dir}")
    files = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    if not files:
        raise ValueError(f"测试目录 {test_dir} 下没有图片")
    return files


@torch.no_grad()
def main(config_path: str, checkpoint: str, test_dir: str, output: str, batch: int, device: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.experiment.get("seed", 42)))

    device_obj = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    image_size = int(cfg.data.get("image_size", 224))
    num_classes = int(cfg.data.get("num_classes", 500))

    processor = CLIPProcessor.from_pretrained(
        str(Path(config_path).resolve().parents[1] / _CLIP_MODEL_DIR),
        size={"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
    )

    model = build_model(cfg, num_classes, device=device)
    load_checkpoint(model, Path(checkpoint))
    model = model.to(device_obj).eval()

    paths = list_test_images(test_dir)
    fnames = [Path(x).name for x in paths]
    print(f"== 推理：{len(paths)} 张测试图，checkpoint={checkpoint} ==")

    ds = _TestDataset(paths, processor)
    loader = DataLoader(ds, batch_size=batch, num_workers=0, shuffle=False)

    preds = np.zeros(len(paths), dtype=np.int64)
    idx = 0
    for px in loader:
        px = px.to(device_obj)
        _, logits = model(px)
        b = logits.size(0)
        preds[idx : idx + b] = logits.argmax(dim=1).cpu().numpy()
        idx += b
        if idx % (batch * 20) < batch:
            print(f"  {idx}/{len(paths)}", flush=True)

    # 输出：文件名,四位类别编号（无表头）
    out_csv = Path(output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        for fn, lab in zip(fnames, preds):
            f.write(f"{fn},{int(lab):04d}\n")

    out_zip = out_csv.with_suffix(".zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=out_csv.name)

    print(f"== 完成：{out_csv}（{len(preds)} 行） ==")
    print(f"== 提交包：{out_zip} ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6 推理 + 生成提交包")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径（best.pt）")
    parser.add_argument("--test-dir", required=True, help="测试集目录（平铺图片）")
    parser.add_argument("--output", default="pred_results.csv", help="输出 CSV 路径")
    parser.add_argument("--batch", type=int, default=128, help="推理 batch size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="计算设备")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.test_dir, args.output, args.batch, args.device)
