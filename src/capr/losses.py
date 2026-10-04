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


def plain_ce(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """无可靠度加权的交叉熵。

    targets 为 int 硬标签时走 F.cross_entropy（可带 label smoothing）；
    targets 为 float 软标签（mixup 后的 one-hot 混合）时走 -Σ y·log_softmax。
    """
    if targets.is_floating_point():
        logp = F.log_softmax(logits, dim=-1)
        return -(targets * logp).sum(dim=-1).mean()
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


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
    hard_threshold: float | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """统一损失入口，返回 (总损失, 分项标量用于日志)。

    hard_threshold：q0 硬丢弃阈值。启用时 q0 低于该值的样本监督损失权重
    置 0（视为噪声，不参与分类监督）；anchor 损失不受影响，仍对这些样本
    施加更强锚定（低 q0 → λ 更大），两者互补。

    supervised 取值：
      - "ce"       可靠度加权 CE（v2 默认，按 q0 加权）
      - "gce"      可靠度加权 GCE（v3 默认）
      - "ce_plain" 无可靠度加权 CE + label smoothing（全量微调基线用，
                   支持 mixup 软标签 targets）
    """
    # 硬丢弃：仅在监督项上对低可靠度样本置 0 权重
    q_sup = q0
    if hard_threshold is not None:
        q_sup = q0.clone()
        q_sup[q0 < hard_threshold] = 0.0

    if supervised == "ce":
        l_sup = weighted_ce(logits, targets, q_sup)
    elif supervised == "gce":
        l_sup = weighted_gce(logits, targets, q_sup, q=gce_q)
    elif supervised == "ce_plain":
        l_sup = plain_ce(logits, targets, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"未知监督损失: {supervised}（支持 ce/gce/ce_plain）")

    l_anchor = anchor_loss(z, z0, q0, lambda_min=anchor_min, lambda_uncertain=anchor_uncertain)
    total = l_sup + l_anchor

    with torch.no_grad():
        parts = {
            "sup": float(l_sup.detach().cpu()),
            "anchor": float(l_anchor.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
    return total, parts
