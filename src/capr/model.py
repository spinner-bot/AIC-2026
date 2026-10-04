"""Stage 3：CAPR 训练模型。

结构：CLIP 视觉塔（冻结主干 + 最后 4 层 LoRA）+ visual_projection（冻结）
     + cosine 分类头（可训练）+ 可学习 logit_scale。

特征口径与 Stage 1 完全一致：
    z = L2_norm(visual_projection(vision_model(pixel_values).pooler_output))
锚定损失因此能与冻结特征 z0 对齐（V3 5.2）。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPModel

from .config import Config, project_root
from .lora import apply_lora, resolve_layers, unfreeze_layer_norms

_CLIP_MODEL_DIR = "models/clip-vit-base-patch32"
_EPS = 1e-8


class CAPRModel(nn.Module):
    """CLIP + LoRA + cosine 分类头。"""

    def __init__(
        self,
        clip_model: CLIPModel,
        num_classes: int,
        init_logit_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.num_classes = num_classes
        self.dim = int(clip_model.config.projection_dim)  # 512

        # cosine 分类头：权重 [num_classes, dim]，forward 中做 L2 归一化
        self.classifier_weight = nn.Parameter(torch.empty(num_classes, self.dim))
        nn.init.normal_(self.classifier_weight, std=0.01)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))

    def _image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """图像 -> L2 归一化特征 z。与 Stage 1 口径一致。

        interpolate_pos_encoding=True：高分辨率（如 288）时 transformers 自动
        双线性插值位置编码（224 的 7×7 → 288 的 9×9），无需手动改 position_embedding。
        """
        out = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=True)
        z = self.visual_projection(out.pooler_output)
        return z / (z.norm(dim=-1, keepdim=True) + _EPS)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """从（已归一化）特征计算 cosine logits。head warmup 直接复用。"""
        w = self.classifier_weight / (self.classifier_weight.norm(dim=-1, keepdim=True) + _EPS)
        return z @ w.T * self.logit_scale.exp()

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (z, logits)。z 用于锚定损失。"""
        z = self._image_embeds(pixel_values)
        return z, self.classify(z)

    def param_groups(self, lr_cfg: dict, weight_decay: float = 0.05) -> list[dict]:
        """按 lr_cfg（classifier/lora/layer_norm）分组可训练参数。

        logit_scale 与 LayerNorm 不施加 weight_decay（前者衰减会导致
        exp(logit_scale) 塌缩到 0，后者本身无需正则）。
        """
        cls_params: list[nn.Parameter] = []
        scale_params: list[nn.Parameter] = []
        lora_params: list[nn.Parameter] = []
        ln_params: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("logit_scale"):
                scale_params.append(p)
            elif "classifier_weight" in name:
                cls_params.append(p)
            elif "lora_A" in name or "lora_B" in name:
                lora_params.append(p)
            elif "layer_norm" in name:
                ln_params.append(p)
            else:
                # 兜底：任何遗漏的可训练参数并入 LoRA 组，避免静默漏训
                lora_params.append(p)

        groups: list[dict] = []
        if cls_params:
            groups.append({"params": cls_params, "lr": float(lr_cfg.get("classifier", 0.0005)), "weight_decay": weight_decay})
        if scale_params:
            groups.append({"params": scale_params, "lr": float(lr_cfg.get("classifier", 0.0005)), "weight_decay": 0.0})
        if lora_params:
            groups.append({"params": lora_params, "lr": float(lr_cfg.get("lora", 0.0001)), "weight_decay": weight_decay})
        if ln_params:
            groups.append({"params": ln_params, "lr": float(lr_cfg.get("layer_norm", 0.00001)), "weight_decay": 0.0})
        return groups


def build_model(
    cfg: Config, num_classes: int, device: str = "cuda"
) -> CAPRModel:
    """加载 CLIP、注入 LoRA、解冻 LayerNorm，返回训练模型。

    只有 vision_model / visual_projection 上 GPU，text tower 保留在 CPU
    （不参与训练，避免占用显存）。
    """
    model_dir = project_root() / _CLIP_MODEL_DIR
    clip = CLIPModel.from_pretrained(str(model_dir))

    model = CAPRModel(clip, num_classes, init_logit_scale=20.0)

    # 1) 冻结主干与视觉投影
    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.visual_projection.parameters():
        p.requires_grad = False

    # 2) LoRA 注入（最后 4 层 Q/V）
    lora_cfg = cfg.model.lora
    layers_spec = lora_cfg.get("layers", "last_4")
    targets = list(lora_cfg.get("targets", ["q_proj", "v_proj"]))
    rank = int(lora_cfg.get("rank", 8))
    alpha = int(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.05))

    num_layers = len(model.vision_model.encoder.layers)
    layer_idxs = resolve_layers(num_layers, layers_spec)
    apply_lora(model.vision_model.encoder, layer_idxs, targets, rank, alpha, dropout)

    # 3) 解冻最后四层 LayerNorm
    unfreeze_layer_norms(model.vision_model.encoder, layer_idxs)

    # 4) 上设备（仅模型持有的部分；text tower 已在 clip 上，del 后释放 CPU 内存）
    device_obj = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(device_obj)
    del clip
    return model


def trainable_state_dict(model: CAPRModel) -> dict[str, torch.Tensor]:
    """仅收集可训练参数（LoRA / 分类头 / LayerNorm），省去冻结主干。"""
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def save_checkpoint(model: CAPRModel, path: Path) -> None:
    """保存可训练参数 + 元信息。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": trainable_state_dict(model), "num_classes": model.num_classes},
        path,
    )


def load_checkpoint(model: CAPRModel, path: Path) -> None:
    """加载可训练参数（strict=False，冻结参数由当前模型提供）。"""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
