"""Stage 2 冒烟测试：用注入噪声的假特征验证 q0 可靠度系统。

验证点：
    - q0 形状与取值范围
    - 各信号形状正确
    - 关键：注入噪声（错标）样本的平均 q0 应低于干净样本
    - group-aware：OOF 不跨组；近重复样本在分组后可信度不被虚高（防泄漏）

用法：
    D:\\anaconda\\envs\\aic\\python.exe scripts/smoke_stage2.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.reliability import compute_q0
from src.capr.signals import _group_kfold

_TRUST = {
    "oof_folds": 3,
    "knn_k": 20,
    "use_gmm": False,
    "conflict_lambda": 0.1,
    "q_min": 0.0,
    "min_effective_count": 5.0,
    "signal_weights": {"cls": 1.0, "proto": 1.0, "knn": 1.0},
}


def _make_features(num_classes: int = 5, per_class: int = 40, dim: int = 512, noise_ratio: float = 0.2):
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((num_classes, dim)) * 2.0  # 类中心分开
    feats, labels = [], []
    for c in range(num_classes):
        fc = rng.standard_normal((per_class, dim)) + centers[c]
        feats.append(fc)
        labels.extend([c] * per_class)
    feats = np.vstack(feats)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    labels = np.array(labels)

    # 注入噪声标签
    n = len(labels)
    noise_mask = rng.random(n) < noise_ratio
    labels_noisy = labels.copy()
    labels_noisy[noise_mask] = rng.integers(0, num_classes, size=int(noise_mask.sum()))

    return feats, labels_noisy, noise_mask


def _make_features_groups(
    num_classes: int = 5, per_class: int = 40, dim: int = 512,
    noise_ratio: float = 0.2, dup_ratio: float = 0.15,
):
    """带重复组的特征：复制部分样本为近重复 twin（同组、同标签）。

    同组同标正是泄漏场景：不排除同组时，twin 会借其近重复源样本在
    OOF/原型/kNN 里被「带高」可信度。
    """
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((num_classes, dim)) * 2.0
    feats, labels = [], []
    for c in range(num_classes):
        fc = rng.standard_normal((per_class, dim)) + centers[c]
        feats.append(fc)
        labels.extend([c] * per_class)
    feats = np.vstack(feats)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    labels = np.array(labels)
    n_orig = len(labels)
    groups = np.arange(n_orig)

    # 随机翻转噪声（仅原始样本）
    flip = rng.random(n_orig) < noise_ratio
    labels_noisy = labels.copy()
    labels_noisy[flip] = rng.integers(0, num_classes, size=int(flip.sum()))
    is_noise = flip.copy()

    # 近重复 twin（同标签，继承源样本的噪声状态）
    dup_src = rng.choice(n_orig, size=int(n_orig * dup_ratio), replace=False)
    twin_feats, twin_labels, twin_groups = [], [], []
    for src in dup_src:
        twin = feats[src] + rng.standard_normal(dim) * 0.001
        twin = twin / (np.linalg.norm(twin) + 1e-8)
        twin_feats.append(twin)
        twin_labels.append(int(labels_noisy[src]))
        twin_groups.append(int(groups[src]))
    feats = np.vstack([feats, np.asarray(twin_feats)])
    labels_noisy = np.concatenate([labels_noisy, np.asarray(twin_labels)])
    groups = np.concatenate([groups, np.asarray(twin_groups)])
    is_noise = np.concatenate([is_noise, flip[dup_src]])

    dup_mask = np.zeros(len(labels_noisy), dtype=bool)
    dup_mask[n_orig:] = True

    return feats, labels_noisy, groups, is_noise, dup_mask


def _test_group_kfold() -> None:
    feats, labels, groups, _, _ = _make_features_groups(per_class=20)
    splits = _group_kfold(feats, labels, groups, n_folds=3, seed=42)
    fold_of = np.zeros(len(labels), dtype=int)
    for fi, (_, test_idx) in enumerate(splits):
        fold_of[test_idx] = fi
    for g in np.unique(groups):
        folds = fold_of[groups == g]
        assert folds.min() == folds.max(), f"组 {g} 被拆到多个折"
    print("OOF group 约束通过：无同组跨折")


def main() -> None:
    feats, labels, noise_mask = _make_features()
    num_classes = int(labels.max()) + 1

    q0, signals = compute_q0(feats, labels, num_classes, _TRUST, seed=42)

    print(f"q0 shape: {q0.shape}")
    assert q0.shape == (len(labels),)
    assert q0.min() >= 0.0 and q0.max() <= 1.0, "q0 超出 [0,1]"
    for name in ("cls", "proto", "knn"):
        assert signals[name].shape == (len(labels),), f"{name} 形状不符"
    print("q0 范围: [{:.3f}, {:.3f}]".format(q0.min(), q0.max()))

    clean_q = q0[~noise_mask].mean()
    noise_q = q0[noise_mask].mean()
    print(f"干净样本平均 q0: {clean_q:.3f}   噪声样本平均 q0: {noise_q:.3f}")
    assert clean_q > noise_q, "可靠度系统未能区分干净/噪声样本"

    print("STAGE2 SMOKE OK")


def main_group() -> None:
    _test_group_kfold()

    feats, labels, groups, is_noise, dup_mask = _make_features_groups()
    num_classes = int(labels.max()) + 1

    q0_blind, _ = compute_q0(feats, labels, num_classes, _TRUST, seed=42)
    q0_group, signals = compute_q0(feats, labels, num_classes, _TRUST, groups=groups, seed=42)

    assert q0_group.shape == (len(labels),)
    assert q0_group.min() >= 0.0 and q0_group.max() <= 1.0, "q0 超出 [0,1]"

    clean_q = q0_group[~is_noise].mean()
    noise_q = q0_group[is_noise].mean()
    print(f"[分组] 干净样本平均 q0: {clean_q:.3f}   噪声样本平均 q0: {noise_q:.3f}")
    assert clean_q > noise_q, "分组可靠度系统未能区分干净/噪声样本"

    dup_blind = q0_blind[dup_mask].mean()
    dup_group = q0_group[dup_mask].mean()
    print(f"近重复样本平均 q0：不分组 {dup_blind:.3f}   分组 {dup_group:.3f}")
    assert dup_group <= dup_blind + 1e-6, "分组后近重复样本可信度应被压低（防泄漏）"

    print("STAGE2 GROUP SMOKE OK")


if __name__ == "__main__":
    main()
    main_group()
