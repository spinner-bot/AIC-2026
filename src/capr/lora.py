"""LoRA 低秩适配：注入 CLIP 视觉塔最后若干层的 Q/V 投影。

对应 V3 5.1：主线使用视觉 LoRA——最后四层 Attention 的 Q/V 投影，
rank=8、alpha=16、dropout=0.05。手写实现（无 peft 依赖）。

A 用 kaiming 初始化，B 零初始化，保证初始时 LoRA 分支输出为 0，
等价于原始冻结模型，从零开始学习残差。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包装 nn.Linear 的低秩适配层。

    y = Wx + (alpha / rank) * (B @ A) x
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 原权重冻结
        for p in linear.parameters():
            p.requires_grad = False

        self._reset_lora_parameters()

    def _reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        h = x @ self.lora_A.T  # [..., rank]
        h = self.lora_dropout(h)
        delta = h @ self.lora_B.T  # [..., out_features]
        return base + self.scaling * delta


def resolve_layers(num_layers: int, spec) -> list[int]:
    """把 layers 配置解析为层索引列表。

    支持：None / "last_4"（默认）/ "all" / 整数 n（最后 n 层）/ 显式列表。
    """
    if spec is None or spec == "last_4":
        return list(range(max(0, num_layers - 4), num_layers))
    if spec == "all":
        return list(range(num_layers))
    if isinstance(spec, int):
        return list(range(max(0, num_layers - spec), num_layers))
    if isinstance(spec, (list, tuple)):
        return [int(i) for i in spec]
    raise ValueError(f"无法解析 lora.layers 配置: {spec!r}")


def apply_lora(
    vision_encoder: nn.Module,
    layers: list[int] | None = None,
    targets: list[str] | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> None:
    """把 LoRA 注入 vision encoder 指定层的 attention 投影。

    layers: 层索引列表（默认最后 4 层）
    targets: 要注入的投影名（默认 ["q_proj", "v_proj"]）
    """
    num_layers = len(vision_encoder.layers)
    if layers is None:
        layers = resolve_layers(num_layers, "last_4")
    if targets is None:
        targets = ["q_proj", "v_proj"]

    for idx in layers:
        attn = vision_encoder.layers[idx].self_attn
        for name in targets:
            orig = getattr(attn, name)
            if isinstance(orig, nn.Linear) and not isinstance(orig, LoRALinear):
                setattr(attn, name, LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout))


def unfreeze_layer_norms(vision_encoder: nn.Module, layers: list[int]) -> None:
    """解冻指定层的 LayerNorm（V3 5.1：最后四层 LayerNorm 可训练）。"""
    for idx in layers:
        block = vision_encoder.layers[idx]
        for name in ("layer_norm1", "layer_norm2"):
            ln = getattr(block, name)
            for p in ln.parameters():
                p.requires_grad = True
