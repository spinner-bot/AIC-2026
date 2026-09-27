"""Stage 0：防泄漏 group split 与头中尾划分。

职责（对应 V3 3.2/3.3）：
    - 完全重复（file_hash 相同）与近重复（phash 汉明距离 <= 阈值）归入同一 group
    - 按类别分层 + group 约束，固定 90/10 训练/验证划分
    - 小类别（样本量 < 阈值）不足以稳定划分时改用固定三折 OOF
    - 头中尾按训练集类别样本量分位数固定划分，写入产物（后续实验不得重划）

防泄漏约束：
    - 完全重复用全局 file_hash 并查集（跨类别也能捕获，噪声标签可能把同一图放到不同类）
    - 近重复在同类别内两两比较 phash（细粒度数据近重复几乎都落在同类别；
      跨类别近重复的极端情况由全局 file_hash 兜底）
"""

from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from .config import Config


class DSU:
    """并查集。"""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _hamming(a: str, b: str) -> int:
    """两个十六进制 phash 字符串的汉明距离（按 bit）。"""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def build_groups(manifest: pd.DataFrame, near_dup_hamming: int = 8) -> pd.DataFrame:
    """为 manifest 增加 group_id 列。

    完全重复按全局 file_hash 归组；近重复按同类别内 phash 汉明距离归组。
    """
    n = len(manifest)
    dsu = DSU(n)
    idx = list(range(n))
    hashes = manifest["file_hash"].tolist()
    phashes = manifest["phash"].tolist()
    classes = manifest["class_idx"].tolist()

    # 1) 完全重复：全局 file_hash -> 首次出现的样本索引
    seen: dict[str, int] = {}
    for i in idx:
        h = hashes[i]
        if h in seen:
            dsu.union(seen[h], i)
        else:
            seen[h] = i

    # 2) 近重复：同类别内两两比较 phash
    by_class: dict[int, list[int]] = defaultdict(list)
    for i in idx:
        by_class[classes[i]].append(i)

    for cid, members in by_class.items():
        m = len(members)
        for a in range(m):
            for b in range(a + 1, m):
                if _hamming(phashes[members[a]], phashes[members[b]]) <= near_dup_hamming:
                    dsu.union(members[a], members[b])

    manifest = manifest.copy()
    manifest["group_id"] = [dsu.find(i) for i in idx]
    return manifest


def split_train_val(
    manifest: pd.DataFrame,
    val_ratio: float = 0.10,
    min_samples_for_split: int = 10,
    oof_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """按 group 约束分层划分，返回增加 fold 列的 manifest。

    大类别：train / val（按 group 粒度贪心逼近 val_ratio）。
    小类别：oof_0 / oof_1 / oof_2（固定三折，供 q0 诊断使用）。
    """
    rng = random.Random(seed)
    manifest = manifest.copy()
    manifest["fold"] = "train"

    for cid, sub in manifest.groupby("class_idx"):
        groups = list(sub.groupby("group_id"))
        rng.shuffle(groups)
        n_c = len(sub)

        if n_c < min_samples_for_split:
            # 小类别：按 group 循环分配三折（保证 group 不跨折）
            for gi, (_, g) in enumerate(groups):
                manifest.loc[g.index, "fold"] = f"oof_{gi % oof_folds}"
        else:
            target = int(n_c * val_ratio)
            acc = 0
            for _, g in groups:
                if acc >= target:
                    break
                manifest.loc[g.index, "fold"] = "val"
                acc += len(g)

    return manifest


def assign_head_mid_tail(
    manifest: pd.DataFrame, head_q: float = 0.33, tail_q: float = 0.33
) -> dict[int, str]:
    """按训练集类别样本量分位数划分头中尾，返回 {class_idx: part}。

    只在训练折（fold == 'train'）上统计样本量，与 V3 3.3 一致。
    """
    train = manifest[manifest["fold"] == "train"]
    counts = train.groupby("class_idx").size().sort_values()

    n_cls = len(counts)
    if n_cls == 0:
        return {}

    # 头中尾按类别数量分位（不是样本量分位），简单可复现
    head_n = int(n_cls * head_q)
    tail_n = int(n_cls * tail_q)
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
    """完整执行 group split + 头中尾划分，返回带 group_id / fold / part 列的 manifest（仅可解码样本）。"""
    near = int(cfg.audit.get("near_dup_hamming", 8))
    val_ratio = float(cfg.audit.get("val_ratio", 0.10))
    min_split = int(cfg.audit.get("min_samples_for_split", 10))
    oof = int(cfg.audit.get("oof_folds", 3))
    head_q = float(cfg.audit.get("head_quantile", 0.33))
    tail_q = float(cfg.audit.get("tail_quantile", 0.33))

    # 只对可解码样本划分；异常样本（decodable=False）无 file_hash/phash，不进训练
    manifest = manifest[manifest["decodable"]].copy()
    manifest = build_groups(manifest, near_dup_hamming=near)
    manifest = split_train_val(
        manifest,
        val_ratio=val_ratio,
        min_samples_for_split=min_split,
        oof_folds=oof,
        seed=int(cfg.experiment.get("seed", 42)),
    )
    parts = assign_head_mid_tail(manifest, head_q=head_q, tail_q=tail_q)
    manifest["part"] = manifest["class_idx"].map(parts).fillna("middle")
    return manifest
