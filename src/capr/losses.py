"""Stage 3：损失函数。

对应 V3 5.2：
    - 监督项：可靠度加权 CE/GCE（第一轮只比较 CE 与 GCE，默认 CE）
    - 锚定项：L_anchor,i = λ_i (1 - cos(z_i, z_i^0))，λ_i = λ_min + λ_uncertain(1 - q_i)
    - 总损失：L = L_sup + L_anchor
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-8


def weighted_ce(logits: torch.Tensor, targets: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
    """可靠度加权交叉熵：Σ q_i · CE_i / Σ q_i。"""
    ce = F.cross_entropy(logits, targets, reduction="none")
    q0 = q0.clamp(min=0.0)
    denom = q0.sum() + _EPS
    return (q0 * ce).sum() / denom


def gce_loss(
    logits: torch.Tensor, targets: torch.Tensor, q: float = 0.7
) -> torch.Tensor:
    """广义交叉熵：L_GCE = (1 - p_y^q) / q（q 越小越鲁棒，0 < q <= 1）。"""
    probs = F.softmax(logits, dim=-1)
    p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return ((1.0 - p_y.clamp(min=_EPS) ** q) / q).mean()


def weighted_gce(
    logits: torch.Tensor, targets: torch.Tensor, q0: torch.Tensor, q: float = 0.7
) -> torch.Tensor:
    """可靠度加权 GCE。"""
    probs = F.softmax(logits, dim=-1)
    p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    per_sample = (1.0 - p_y.clamp(min=_EPS) ** q) / q
    q0 = q0.clamp(min=0.0)
    return (q0 * per_sample).sum() / (q0.sum() + _EPS)


def anchor_loss(
    z: torch.Tensor,
    z0: torch.Tensor,
    q0: torch.Tensor,
    lambda_min: float = 0.01,
    lambda_uncertain: float = 0.05,
) -> torch.Tensor:
    """自适应特征锚定：低可信样本受到更强冻结先验保护。

    z  : 当前模型输出特征（已 L2 归一化）
    z0 : 冻结特征（Stage 1，已 L2 归一化）
    q0 : 样本可靠度
    """
    cos = (z * z0).sum(dim=-1)  # 两者均已归一化，点积即余弦
    lam = lambda_min + lambda_uncertain * (1.0 - q0.clamp(0.0, 1.0))
    return (lam * (1.0 - cos)).mean()


def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z: torch.Tensor,
    z0: torch.Tensor,
    q0: torch.Tensor,
    supervised: str = "ce",
    gce_q: float = 0.7,
    anchor_min: float = 0.01,
    anchor_uncertain: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """统一损失入口，返回 (总损失, 分项标量用于日志)。"""
    if supervised == "ce":
        l_sup = weighted_ce(logits, targets, q0)
    elif supervised == "gce":
        l_sup = weighted_gce(logits, targets, q0, q=gce_q)
    else:
        raise ValueError(f"未知监督损失: {supervised}（支持 ce/gce）")

    l_anchor = anchor_loss(z, z0, q0, lambda_min=anchor_min, lambda_uncertain=anchor_uncertain)
    total = l_sup + l_anchor

    with torch.no_grad():
        parts = {
            "sup": float(l_sup.detach().cpu()),
            "anchor": float(l_anchor.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
    return total, parts
