"""Stage 2：可靠度融合与尾类保护。

职责（对应 V3 4.4/4.5）：
    - 各信号类内转稳健秩分数
    - 计算信号冲突度 d_i = Var(s̃_cls, s̃_proto, s̃_knn, [s̃_gmm])
    - 融合 q0 = clip(Σ a_m·s̃_im - λ_d·d_i, q_min, 1)·g(n_{y_i})
    - 尾类最小可信质量（不用固定 floor，不把所有尾类样本无脑抬高）
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import rankdata

from .signals import (
    compute_gmm_signal,
    compute_mutual_knn_signal,
    compute_oof_signal,
    compute_prototype_signal,
)


def _rank_within_class(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """类内稳健秩分数，映射到 [0,1]（平均秩，抗离群）。"""
    out = np.zeros_like(values, dtype=np.float64)
    for c in np.unique(labels):
        mask = labels == c
        v = values[mask]
        n = len(v)
        if n <= 1:
            out[mask] = 0.5
        else:
            ranks = rankdata(v, method="average")
            out[mask] = (ranks - 1) / (n - 1)
    return out


def fuse_q0(
    signals: dict[str, np.ndarray],
    labels: np.ndarray,
    signal_weights: dict[str, float] | None = None,
    conflict_lambda: float = 0.1,
    q_min: float = 0.0,
) -> np.ndarray:
    """融合多信号为 q0（V3 4.4）。

    signals: {名称: 信号数组}，值为 None 的信号会被跳过（如 GMM 回退）。
    先类内转秩，再按权重求和，减去冲突惩罚，clip 到 [q_min, 1]。
    """
    if signal_weights is None:
        signal_weights = {}

    ranked = {
        name: _rank_within_class(v, labels)
        for name, v in signals.items()
        if v is not None and name != "losses"
    }
    if not ranked:
        raise ValueError("没有可用信号，无法计算 q0")

    n = len(labels)
    q = np.zeros(n, dtype=np.float64)
    for name, r in ranked.items():
        q += signal_weights.get(name, 1.0) * r
    q /= sum(signal_weights.get(name, 1.0) for name in ranked)

    # 信号冲突惩罚：多信号秩分数方差大 → 降低 q（不确定就保守）
    if len(ranked) >= 2:
        stacked = np.stack(list(ranked.values()), axis=1)
        conflict = stacked.var(axis=1)
        q = q - conflict_lambda * conflict

    return np.clip(q, q_min, 1.0)


def apply_tail_protection(
    q: np.ndarray, labels: np.ndarray, min_effective_count: float = 5.0
) -> np.ndarray:
    """尾类最小可信质量（V3 4.5）。

    保证每类总可信权重不低于 min_effective_count；不足时整体抬升该类（保持类内相对顺序），
    但不保证所有尾类样本高权重，也不使用固定 0.7 floor。
    """
    q = q.copy().astype(np.float64)
    for c in np.unique(labels):
        mask = labels == c
        qc = q[mask]
        total = qc.sum()
        if total < min_effective_count and len(qc) > 0:
            deficit = min_effective_count - total
            q[mask] = np.clip(qc + deficit / len(qc), 0.0, 1.0)
    return q


def compute_q0(
    feats: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    trust_cfg: dict,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """完整 q0 计算：四信号 + 融合 + 尾类保护。

    返回 (q0, signals)。signals 含 cls/proto/knn/gmm（gmm 可能为 None）与 losses。
    """
    use_gmm = bool(trust_cfg.get("use_gmm", False))
    knn_k = int(trust_cfg.get("knn_k", 20))
    oof_folds = int(trust_cfg.get("oof_folds", 3))
    conflict_lambda = float(trust_cfg.get("conflict_lambda", 0.1))
    q_min = float(trust_cfg.get("q_min", 0.0))
    min_eff = float(trust_cfg.get("min_effective_count", 5.0))
    weights = trust_cfg.get("signal_weights", None)

    started = time.monotonic()
    print(f"  [1/4] OOF 余弦分类器信号（{oof_folds} 折）...", flush=True)
    s_cls, losses = compute_oof_signal(
        feats, labels, num_classes, n_folds=oof_folds, seed=seed
    )
    print(f"  [1/4] OOF 完成，用时 {time.monotonic() - started:.1f}s", flush=True)

    print("  [2/4] 留一鲁棒原型信号...", flush=True)
    s_proto = compute_prototype_signal(feats, labels, num_classes)
    print(f"  [2/4] 原型完成，累计 {time.monotonic() - started:.1f}s", flush=True)

    print(f"  [3/4] mutual-kNN 信号（faiss，k={knn_k}）...", flush=True)
    s_knn = compute_mutual_knn_signal(feats, labels, k=knn_k)
    print(f"  [3/4] kNN 完成，累计 {time.monotonic() - started:.1f}s", flush=True)

    s_gmm = None
    if use_gmm:
        s_gmm = compute_gmm_signal(losses, min_component=int(trust_cfg.get("gmm_min_component", 100)))

    print("  [4/4] 融合 + 尾类保护...", flush=True)
    signal_map = {"cls": s_cls, "proto": s_proto, "knn": s_knn, "gmm": s_gmm}
    q0 = fuse_q0(signal_map, labels, signal_weights=weights, conflict_lambda=conflict_lambda, q_min=q_min)
    q0 = apply_tail_protection(q0, labels, min_effective_count=min_eff)
    print(f"  [4/4] 完成，总用时 {time.monotonic() - started:.1f}s", flush=True)

    signals = {"cls": s_cls, "proto": s_proto, "knn": s_knn, "gmm": s_gmm, "losses": losses}
    return q0, signals
