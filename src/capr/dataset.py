"""Stage 3：训练/验证数据集。

从 split.csv + q0.csv + features.npy 一次性拼好样本表，Dataset 按行返回。
特征缓存 z0 全量载入内存（约 190MB），训练时随 batch 转移到设备。
图像预处理与 Stage 1 一致（CLIPProcessor 默认，MVP 无随机增强）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class TrainDataset(Dataset):
    """训练样本：返回 (pixel_values, class_idx, q0, z0)。"""

    def __init__(
        self,
        items: pd.DataFrame,
        z0: np.ndarray,
        processor: CLIPProcessor,
    ) -> None:
        self.items = items.reset_index(drop=True)
        self.z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        row = self.items.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return (
            px,
            int(row["class_idx"]),
            float(row["q0"]),
            self.z0[i],
        )


class ValDataset(Dataset):
    """验证样本：返回 (pixel_values, class_idx)。"""

    def __init__(self, items: pd.DataFrame, processor: CLIPProcessor) -> None:
        self.items = items.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        row = self.items.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return px, int(row["class_idx"])


def _collate_train(batch):
    px = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    q0 = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    z0 = torch.stack([b[3] for b in batch])
    return px, labels, q0, z0


def _collate_val(batch):
    px = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return px, labels


def collate_fn(mode: str):
    return _collate_train if mode == "train" else _collate_val


def build_datasets(
    split: pd.DataFrame,
    q0: pd.DataFrame,
    feats: np.ndarray,
    processor: CLIPProcessor,
) -> tuple[TrainDataset, ValDataset, pd.DataFrame, pd.DataFrame]:
    """按 fold 划分训练/验证，并返回各自样本表（含 class_idx/q0 列）。"""
    train_mask = split["fold"] != "val"
    train_items = split[train_mask].copy()
    val_items = split[~train_mask].copy()

    # q0 列来自 q0.csv（训练样本有值，val 为 NaN）
    train_items["q0"] = q0.loc[train_mask, "q0"].values

    z0_train = feats[train_mask.values]

    train_ds = TrainDataset(train_items, z0_train, processor)
    val_ds = ValDataset(val_items, processor)
    return train_ds, val_ds, train_items, val_items
