"""Stage 3：训练/验证数据集。

从 split.csv + q0.csv + features.npy 一次性拼好样本表，Dataset 按行返回。
特征缓存 z0 全量载入内存（约 190MB），训练时随 batch 转移到设备。
图像预处理：训练施加轻量随机增强，验证保持 CLIPProcessor 默认中心裁剪。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
import torchvision.transforms as T


def make_train_aug(
    image_size: int = 224, crop_min: float = 0.5, ra_magnitude: int = 9
) -> T.Compose:
    """训练增强（分辨率/裁剪下界/RA 强度可配，供消融实验用）。

    细粒度识别里裁剪下界过低会切掉判别部位（鸟喙/花蕊等），
    RA magnitude 过高会破坏细粒度纹理，二者应作为消融变量。
    """
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(crop_min, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            # RandAugment：细粒度识别标配，比单一 ColorJitter 提供更强的变换多样性，
            # 同时作为正则化抑制噪声标签过拟合（含 Color/Contrast/Brightness 等子操作）。
            T.RandAugment(num_ops=2, magnitude=ra_magnitude),
        ]
    )


class TrainDataset(Dataset):
    """训练样本：返回 (pixel_values, class_idx, q0, z0)。

    训练时施加随机增强（RandAugment），防过拟合；
    验证集保持单一中心裁剪。z0 是 Stage 1 冻结特征（原始中心裁剪），
    锚定损失用它约束增强后的特征不漂移。
    """

    def __init__(
        self,
        items: pd.DataFrame,
        z0: np.ndarray,
        processor: CLIPProcessor,
        image_size: int = 224,
        crop_min: float = 0.5,
        ra_magnitude: int = 9,
        augment: bool = True,
    ) -> None:
        self.items = items.reset_index(drop=True)
        self.z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        self.processor = processor
        self.aug = (
            make_train_aug(image_size, crop_min=crop_min, ra_magnitude=ra_magnitude)
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        row = self.items.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
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


def compute_dup_conflict_mask(split: pd.DataFrame) -> pd.Series:
    """标记 exact-dup 标签冲突样本（确定性噪声，True=噪声）。

    完全重复（file_hash 相同）的图应属同一类；若同一 hash 出现在 2+ 个不同类别，
    则这些样本中至少有一份标签错误。整组标记为冲突，训练时可剔除。
    """
    hashes = split["file_hash"]
    dup = split[hashes.duplicated(keep=False) & hashes.notna()]
    if len(dup) == 0:
        return pd.Series(False, index=split.index)
    conflict_hashes = dup.groupby("file_hash")["class_idx"].nunique()
    conflict_hashes = conflict_hashes[conflict_hashes > 1].index
    return split["file_hash"].isin(conflict_hashes)


def compute_dup_sample_weight(split: pd.DataFrame) -> pd.Series:
    """exact-dup 标签一致组按 1/组大小 降权，避免过度采样重复样本。

    冲突组（同 hash 多标签）不在此处理——由 exclude_dup_conflict 整体剔除。
    返回与 split 行对齐的权重（非重复=1.0，一致重复组=1/group_size）。
    """
    w = pd.Series(1.0, index=split.index)
    hashes = split["file_hash"]
    dup = split[hashes.duplicated(keep=False) & hashes.notna()]
    if len(dup) == 0:
        return w
    consistent = dup.groupby("file_hash")["class_idx"].nunique()
    consistent_hashes = consistent[consistent == 1].index
    if len(consistent_hashes) == 0:
        return w
    sizes = split[split["file_hash"].isin(consistent_hashes)].groupby("file_hash").size()
    for h, gsize in sizes.items():
        w[split["file_hash"] == h] = 1.0 / gsize
    return w


def build_datasets(
    split: pd.DataFrame,
    q0: pd.DataFrame,
    feats: np.ndarray,
    processor: CLIPProcessor,
    image_size: int = 224,
    crop_min: float = 0.5,
    ra_magnitude: int = 9,
    train_keep: pd.Series | None = None,
    dup_downweight: bool = False,
) -> tuple[TrainDataset, ValDataset, pd.DataFrame, pd.DataFrame]:
    """按 fold 划分训练/验证，并返回各自样本表（含 class_idx/q0 列）。

    train_keep：可选训练保留掩码（True=保留，与 split 行对齐）。默认训练=所有
    非 val 样本；传入后可在训练集中剔除确定性噪声（如 exact-dup 标签冲突样本）。
    dup_downweight：对 exact-dup 标签一致组按 1/组大小 降权（写入 sample_weight 列）。
    """
    val_mask = split["fold"] == "val"
    if train_keep is None:
        train_keep = ~val_mask
    train_items = split[train_keep].copy()
    val_items = split[val_mask].copy()

    # q0 列来自 q0.csv（训练样本有值，val 为 NaN）
    train_items["q0"] = q0.loc[train_keep, "q0"].values

    # exact-dup 标签一致组降权（可选）
    if dup_downweight:
        train_items["sample_weight"] = compute_dup_sample_weight(split).loc[train_keep].values

    z0_train = feats[train_keep.values]

    train_ds = TrainDataset(
        train_items, z0_train, processor,
        image_size=image_size, crop_min=crop_min, ra_magnitude=ra_magnitude,
    )
    val_ds = ValDataset(val_items, processor)
    return train_ds, val_ds, train_items, val_items
