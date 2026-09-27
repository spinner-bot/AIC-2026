"""Stage 2 冒烟测试：用注入噪声的假特征验证 q0 可靠度系统。

验证点：
    - q0 形状与取值范围
    - 各信号形状正确
    - 关键：注入噪声（错标）样本的平均 q0 应低于干净样本
      （可靠度系统应当能把可疑样本识别为低可信）

用法：
    D:\\anaconda\\envs\\aic\\python.exe scripts/smoke_stage2.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.reliability import compute_q0

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


if __name__ == "__main__":
    main()
