"""损失函数。

职责:
    - 鲁棒损失：CE / SCE / GCE / ELR
    - OPSD 自蒸馏损失：KL(p(x1) || p(x2))，带 λ 调度（linear ramp）
    - 可选：class-balanced / Focal（长尾阶段）
"""

# TODO: 实现 sce_loss / gce_loss / elr_loss / self_distill_loss / class_balanced_loss
