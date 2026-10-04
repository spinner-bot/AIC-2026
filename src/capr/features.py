"""Stage 1：冻结特征缓存。

职责（对应 V3 4.1）：
    - 使用官方 CLIP ViT-B/32 图像塔提取 L2 归一化特征 z_i^0
    - 分块写入 .npy，避免内存峰值
    - 特征只用于训练期可靠度 / 原型 / 锚定，不作为额外推理分支（不构成集成）
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from .config import Config, project_root

_CLIP_MODEL_DIR = "models/clip-vit-base-patch32"


class _ManifestImageDataset(Dataset):
    """按 manifest 的图片路径批量读取。"""

    def __init__(self, paths: list[str], processor: CLIPProcessor):
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.paths[i]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def _load_clip(cfg: Config, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    model_dir = project_root() / _CLIP_MODEL_DIR
    model = CLIPModel.from_pretrained(model_dir).to(device).eval()
    image_size = int(cfg.data.get("image_size", 224))
    processor = CLIPProcessor.from_pretrained(
        model_dir,
        size={"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
    )
    return model, processor


@torch.no_grad()
def extract_features(
    cfg: Config, manifest: pd.DataFrame, device: str = "cuda"
) -> tuple[np.ndarray, pd.DataFrame]:
    """提取冻结 CLIP 特征。

    只对 decodable 的样本提取；返回 (features[N,512], subset)。
    subset 为 decodable 子集，且额外含 manifest_idx 列用于对齐原始 manifest。
    """
    if torch.cuda.is_available() and device == "cuda":
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device("cpu")

    model, processor = _load_clip(cfg, device_obj)

    subset = manifest[manifest["decodable"]].copy()
    subset["manifest_idx"] = subset.index
    subset = subset.reset_index(drop=True)

    ds = _ManifestImageDataset(subset["path"].tolist(), processor)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.features.get("batch_size", 64)),
        num_workers=int(cfg.features.get("num_workers", 0)),
    )

    dim = int(model.config.projection_dim)  # ViT-B/32 -> 512
    feats = np.zeros((len(subset), dim), dtype=np.float32)

    total = len(subset)
    started = time.monotonic()
    start = 0
    for i, batch in enumerate(loader, start=1):
        batch = batch.to(device_obj)
        # transformers 5.x: get_image_features 返回 BaseModelOutputWithPooling，
        # 图像特征在 .pooler_output（512 维，即 CLIP image_embeds）
        z = model.get_image_features(pixel_values=batch, interpolate_pos_encoding=True).pooler_output
        z = z / z.norm(dim=-1, keepdim=True)  # L2 归一化
        end = start + batch.size(0)
        feats[start:end] = z.cpu().numpy()
        start = end
        if i % 100 == 0 or end == total:
            elapsed = max(time.monotonic() - started, 0.001)
            print(f"  {end}/{total} ({end / total:.1%})，{end / elapsed:.1f} 张/秒", flush=True)

    return feats, subset


def save_features(
    cfg: Config, feats: np.ndarray, subset: pd.DataFrame
) -> tuple[Path, Path]:
    """落盘 features.npy 与 index.csv，返回两个文件路径。"""
    out_dir = cfg.path(cfg.features.get("cache_dir")) or cfg.path(cfg.experiment.output_dir) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_dir / "features.npy"
    index_path = out_dir / "index.csv"

    np.save(feat_path, feats)
    subset.to_csv(index_path, index=False)
    return feat_path, index_path
