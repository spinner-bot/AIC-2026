"""Stage 0：防泄漏 group split 与头中尾划分。

职责（对应 V3 3.2/3.3）：
    - 完全重复与近重复图片在全局范围归入同一 group
    - group 不跨折，固定 90/10 训练/验证划分
    - 小类别不足以稳定划分时使用固定三折 OOF
    - 头中尾边界按当前训练数据固定并写入产物
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

import pandas as pd

from .config import Config


class DSU:
    """并查集。"""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class _HammingMultiIndex:
    """64-bit 哈希的精确半径检索，针对 V3 默认阈值 8 优化。

    把 64 bit 切成 21/21/22 三段。若全局汉明距离 <= 8，则至少一段
    距离 <= 2；枚举该段两位以内的翻转即可得到完整候选集，再用完整
    64-bit 距离复核。因此不是近似检索，不会漏掉阈值内样本。
    """

    _WIDTHS = (21, 21, 22)
    _SHIFTS = (0, 21, 42)

    def __init__(self, radius: int) -> None:
        if not 0 <= radius <= 8:
            raise ValueError("near_dup_hamming 当前支持 0..8")
        self.radius = radius
        local_radius = radius // len(self._WIDTHS)
        self.tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in self._WIDTHS]
        self.values: dict[int, int] = {}
        self.masks = [self._flip_masks(width, local_radius) for width in self._WIDTHS]

    @staticmethod
    def _flip_masks(width: int, radius: int) -> list[int]:
        masks = [0]
        for distance in range(1, radius + 1):
            for bits in combinations(range(width), distance):
                mask = 0
                for bit in bits:
                    mask |= 1 << bit
                masks.append(mask)
        return masks

    def query(self, value: int) -> list[int]:
        candidates: set[int] = set()
        for table, width, shift, masks in zip(
            self.tables, self._WIDTHS, self._SHIFTS, self.masks
        ):
            segment = (value >> shift) & ((1 << width) - 1)
            for mask in masks:
                bucket = table.get(segment ^ mask)
                if bucket:
                    candidates.update(bucket)
        return [
            index
            for index in candidates
            if (value ^ self.values[index]).bit_count() <= self.radius
        ]

    def add(self, value: int, index: int) -> None:
        self.values[index] = value
        for table, width, shift in zip(self.tables, self._WIDTHS, self._SHIFTS):
            segment = (value >> shift) & ((1 << width) - 1)
            table[segment].append(index)


def build_groups(manifest: pd.DataFrame, near_dup_hamming: int = 8) -> pd.DataFrame:
    """增加稳定的 group_id；全局发现完全重复与近重复图片。

    多索引哈希避免构造 100k 级全量两两矩阵，也不会因标签不同而漏掉
    可能由噪声标签造成的跨类别近重复。
    """
    n = len(manifest)
    dsu = DSU(n)
    indices = list(range(n))
    hashes = manifest["file_hash"].tolist()
    phashes = manifest["phash"].tolist()

    seen: dict[str, int] = {}
    for index in indices:
        digest = hashes[index]
        if digest in seen:
            dsu.union(seen[digest], index)
        else:
            seen[digest] = index

    indexer = _HammingMultiIndex(near_dup_hamming)
    phash_seen: dict[int, int] = {}
    for index in indices:
        value = int(phashes[index], 16)
        if value in phash_seen:
            dsu.union(phash_seen[value], index)
            continue
        for other in indexer.query(value):
            dsu.union(other, index)
        indexer.add(value, index)
        phash_seen[value] = index

    roots = [dsu.find(index) for index in indices]
    stable_ids: dict[int, int] = {}
    result = manifest.copy()
    result["group_id"] = [
        stable_ids.setdefault(root, len(stable_ids)) for root in roots
    ]
    return result


def split_train_val(
    manifest: pd.DataFrame,
    val_ratio: float = 0.10,
    min_samples_for_split: int = 10,
    oof_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """按全局 group 约束做近似类别分层划分。

    一个 group 即使横跨多个标签也只赋值一次，从结构上杜绝重复图跨折。
    大类别进入 train/val；纯小类别 group 进入固定 OOF 折。若 group 同时
    含大类和小类，防泄漏优先，整体服从大类的 train/val 赋值。
    """
    if oof_folds < 2:
        raise ValueError("oof_folds 必须至少为 2")

    rng = random.Random(seed)
    result = manifest.copy()
    result["fold"] = "train"

    class_totals = result.groupby("class_idx").size().to_dict()
    large_classes = {
        int(cid) for cid, count in class_totals.items() if count >= min_samples_for_split
    }
    targets = {
        cid: max(1, int(class_totals[cid] * val_ratio)) for cid in large_classes
    }
    val_counts: dict[int, int] = defaultdict(int)
    oof_counts: dict[int, list[int]] = defaultdict(lambda: [0] * oof_folds)

    groups: list[tuple[int, pd.DataFrame]] = list(result.groupby("group_id", sort=True))
    rng.shuffle(groups)
    for _, group in groups:
        counts = {int(k): int(v) for k, v in group.groupby("class_idx").size().items()}
        large_counts = {cid: n for cid, n in counts.items() if cid in large_classes}

        if large_counts:
            before = sum(abs(val_counts[cid] - targets[cid]) for cid in large_counts)
            after = sum(
                abs(val_counts[cid] + n - targets[cid])
                for cid, n in large_counts.items()
            )
            if after < before:
                result.loc[group.index, "fold"] = "val"
                for cid, n in large_counts.items():
                    val_counts[cid] += n
            continue

        fold_order = list(range(oof_folds))
        rng.shuffle(fold_order)
        best_fold = min(
            fold_order,
            key=lambda fold: sum(oof_counts[cid][fold] for cid in counts),
        )
        result.loc[group.index, "fold"] = f"oof_{best_fold}"
        for cid, n in counts.items():
            oof_counts[cid][best_fold] += n

    if (result.groupby("group_id")["fold"].nunique() > 1).any():
        raise RuntimeError("内部错误：存在重复 group 跨折")
    return result


def assign_head_mid_tail(
    manifest: pd.DataFrame, head_q: float = 0.33, tail_q: float = 0.33
) -> dict[int, str]:
    """按当前训练数据的类别样本量固定划分头中尾。"""
    train = manifest[manifest["fold"] != "val"]
    counts = train.groupby("class_idx").size().sort_values()
    n_classes = len(counts)
    if n_classes == 0:
        return {}

    head_n = int(n_classes * head_q)
    tail_n = int(n_classes * tail_q)
    head_ids = set(counts.index[-head_n:]) if head_n else set()
    tail_ids = set(counts.index[:tail_n]) if tail_n else set()

    parts: dict[int, str] = {}
    for cid in counts.index:
        if cid in head_ids:
            parts[int(cid)] = "head"
        elif cid in tail_ids:
            parts[int(cid)] = "tail"
        else:
            parts[int(cid)] = "middle"
    return parts


def run_group_split(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """完整执行 group split + 头中尾划分（仅可解码样本）。"""
    near = int(cfg.audit.get("near_dup_hamming", 8))
    val_ratio = float(cfg.audit.get("val_ratio", 0.10))
    min_split = int(cfg.audit.get("min_samples_for_split", 10))
    oof = int(cfg.audit.get("oof_folds", 3))
    head_q = float(cfg.audit.get("head_quantile", 0.33))
    tail_q = float(cfg.audit.get("tail_quantile", 0.33))

    result = manifest[manifest["decodable"]].copy().reset_index(drop=True)
    result = build_groups(result, near_dup_hamming=near)
    result = split_train_val(
        result,
        val_ratio=val_ratio,
        min_samples_for_split=min_split,
        oof_folds=oof,
        seed=int(cfg.experiment.get("seed", 42)),
    )
    parts = assign_head_mid_tail(result, head_q=head_q, tail_q=tail_q)
    result["part"] = result["class_idx"].map(parts).fillna("middle")
    return result
