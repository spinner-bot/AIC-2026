"""Stage 2：可靠度信号计算。

职责（对应 V3 4.2/4.3）：
    - OOF 分类信号：余弦分类器 + 留出法，s_cls = P_{-fold(i)}(y_i | x_i)
    - 留一鲁棒原型信号：s_proto = cos(z_i, p_{y_i}^{-i}) - max_{c≠y_i} cos(z_i, p_c)
    - mutual-kNN 局部信号：互惠近邻图上的邻域一致率
    - 条件 GMM 信号：干净/噪声双分量后验，满足条件才启用，否则返回 None（回退百分位）

所有信号只从训练折的冻结特征计算，验证折仅用于评估。
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold

_EPS = 1e-8


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + _EPS)


def _cosine(feats: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """feats [N,D] 与 protos [C,D] 的余弦相似度 -> [N,C]（两边均已 L2 归一化）。"""
    return _l2_normalize(feats) @ _l2_normalize(protos).T


def _prototypes(feats: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """每类 L2 归一化原型（普通均值；空类返回零向量）。"""
    protos = np.zeros((num_classes, feats.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            protos[c] = _l2_normalize(feats[mask].mean(axis=0, keepdims=True))[0]
    return protos


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / (exp.sum(axis=1, keepdims=True) + _EPS)


def compute_oof_signal(
    feats: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_folds: int = 3,
    scale: float = 20.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF 余弦分类器信号。

    用 K 折（优先分层）划分，对每个折用其余折训练余弦分类器，预测本折样本，
    取 s_cls[i] = P(y_i | x_i)。同时返回 losses = -log P(y_i | x_i)（供 GMM 使用）。

    该信号仍继承标签噪声，仅作为带偏差的判别证据（V3 4.2）。
    """
    n = len(labels)
    s_cls = np.zeros(n, dtype=np.float32)
    losses = np.zeros(n, dtype=np.float32)

    try:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(feats, labels))
    except ValueError:
        # 某些类别样本量 < n_folds，退化为随机 KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(feats))

    for train_idx, test_idx in splits:
        protos = _prototypes(feats[train_idx], labels[train_idx], num_classes)
        logits = _cosine(feats[test_idx], protos) * scale
        probs = _softmax(logits)
        test_labels = labels[test_idx]
        s_cls[test_idx] = probs[np.arange(len(test_idx)), test_labels]
        losses[test_idx] = -np.log(probs[np.arange(len(test_idx)), test_labels] + _EPS)

    return s_cls, losses


def compute_prototype_signal(
    feats: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    shrink_tau: float = 10.0,
) -> np.ndarray:
    """留一鲁棒原型信号。

    原型收缩：p_c = ρ_c·p_c^local + (1-ρ_c)·p_global，ρ_c = n_c/(n_c + τ)
    随有效样本量增加而增加；小类别向全局原型收缩（V3 4.2）。
    留一：p_{y_i}^{-i} 通过减去样本自身贡献精确计算。

    返回 s_proto[i] = cos(z_i, p_{y_i}^{-i}) - max_{c≠y_i} cos(z_i, p_c)。
    """
    n, d = feats.shape
    z = _l2_normalize(feats)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    # 类内未归一化原型和（用于精确留一）与全局原型
    local_sum = np.zeros((num_classes, d), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            local_sum[c] = feats[mask].sum(axis=0)

    global_proto = _l2_normalize(feats.sum(axis=0, keepdims=True))[0]

    # 收缩后的每类原型 p_c
    rho = counts / (counts + shrink_tau)  # [C]
    protos = np.zeros((num_classes, d), dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            p_local = _l2_normalize(local_sum[c : c + 1])[0]
            protos[c] = _l2_normalize(
                (rho[c] * p_local + (1 - rho[c]) * global_proto).reshape(1, -1)
            )[0]

    s_proto = np.zeros(n, dtype=np.float32)
    for i in range(n):
        c = labels[i]
        n_c = counts[c]
        if n_c <= 1:
            # 类别只有一个样本，留一原型退化为全局原型
            p_leave = global_proto
        else:
            # 留一：从类内和里减去样本自身，再收缩
            leave_sum = local_sum[c] - feats[i]
            p_leave_local = _l2_normalize(leave_sum.reshape(1, -1))[0]
            p_leave = _l2_normalize(
                (rho[c] * p_leave_local + (1 - rho[c]) * global_proto).reshape(1, -1)
            )[0]

        sim_all = z[i] @ protos.T
        sim_own = float(z[i] @ p_leave)
        # 异类最大余弦（排除自身类别）
        other = np.delete(sim_all, c)
        s_proto[i] = sim_own - (other.max() if other.size else 0.0)

    return s_proto


def _mutual_knn_topk(
    feats: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """用 faiss 找每个样本的 top-(k+1) 近邻（含自身），返回 (dist, idx)。"""
    import faiss

    feats = _l2_normalize(feats).astype(np.float32)
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)  # 内积 == 余弦（已归一化）
    index.add(feats)
    dist, idx = index.search(feats, k + 1)
    return dist, idx


def compute_mutual_knn_signal(feats: np.ndarray, labels: np.ndarray, k: int = 20) -> np.ndarray:
    """mutual-kNN 局部信号。

    互惠约束：仅当 j 是 i 的 top-k 近邻且 i 也是 j 的 top-k 近邻时才建立连接。
    邻域一致率（V3 4.2）：
        s_knn[i] = Σ_{j∈N_i} w_ij·1(y_j=y_i) / (Σ_{j∈N_i} w_ij + ε)，w_ij = max(cos, 0)
    """
    n = len(labels)
    z = _l2_normalize(feats)
    dist, idx = _mutual_knn_topk(feats, k)

    # 构建每个样本的 top-k 邻居集合（排除自身）
    neighbor_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in idx[i]:
            if j != i:
                neighbor_sets[i].add(int(j))

    s_knn = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mutual = [j for j in neighbor_sets[i] if i in neighbor_sets[j]]
        if not mutual:
            s_knn[i] = 0.0
            continue
        w = np.clip(z[i] @ z[mutual].T, 0.0, None)
        same = (labels[mutual] == labels[i]).astype(np.float32)
        s_knn[i] = float((w * same).sum() / (w.sum() + _EPS))

    return s_knn


def compute_gmm_signal(
    losses: np.ndarray,
    min_component: int = 100,
    n_init: int = 3,
    separation_gain: float = 0.05,
) -> np.ndarray | None:
    """条件 GMM 信号。

    对 OOF 损失拟合干净/噪声双分量 GMM，返回干净分量后验 s_gmm。
    不满足条件时返回 None，由融合层回退到类内百分位（V3 4.3）。
    启用条件：分量样本量、方差稳定、多次初始化一致、分离度优于百分位基线。
    """
    losses = np.asarray(losses, dtype=np.float64).reshape(-1, 1)
    n = len(losses)
    if n < 2 * min_component:
        return None

    # 多次初始化，取 log-likelihood 最优
    best = None
    for _ in range(n_init):
        gmm = GaussianMixture(
            n_components=2, covariance_type="full", random_state=None, max_iter=200
        ).fit(losses)
        if best is None or gmm.lower_bound_ > best.lower_bound_:
            best = gmm

    means = best.means_.ravel()
    # 干净分量 = 损失均值较低的那个
    clean_comp = int(np.argmin(means))
    post = best.predict_proba(losses)[:, clean_comp]

    # 条件 1：两个分量都有足够样本（按后验归属）
    hard = best.predict(losses)
    if (hard == clean_comp).sum() < min_component or (hard != clean_comp).sum() < min_component:
        return None

    # 条件 2：分离度优于简单百分位基线
    # 基线：按损失二分，低损失一半视为干净；GMM 优于基线才算有效
    baseline = (losses.ravel() <= np.median(losses)).astype(np.float64)
    gmm_acc = ((post > 0.5) == baseline).mean()
    if gmm_acc <= 0.5 + separation_gain:
        return None

    return post.astype(np.float32)
