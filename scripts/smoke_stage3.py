"""Stage 3 冒烟测试：用小规模假数据验证训练链路。

验证点：
    - CAPRModel 前向输出 z/logits 形状正确
    - 总损失（加权 CE + 锚定）可计算、可反向，LoRA 梯度非空且有限
    - 分类头预热（缓存特征）1 epoch 不崩
    - robust 微调（假 DataLoader）1 epoch 不崩，evaluate 返回合法值
    - checkpoint 可存可读

用法：
    D:\\anaconda\\envs\\aic\\python.exe scripts/smoke_stage3.py
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capr.config import Config
from src.capr.dataset import collate_fn
from src.capr.losses import compute_total_loss
from src.capr.model import build_model, load_checkpoint, save_checkpoint
from src.capr.train_stage3 import train_head_warmup, train_robust

_NUM_CLASSES = 8
_DIM = 512


class _FakeTrain(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, i):
        z0 = torch.randn(_DIM)
        z0 = z0 / z0.norm()
        return torch.randn(3, 224, 224), i % _NUM_CLASSES, float(np.random.rand()), z0


class _FakeVal(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, i):
        return torch.randn(3, 224, 224), i % _NUM_CLASSES


def _make_cfg() -> Config:
    return Config(
        {
            "model": {
                "lora": {
                    "layers": "last_4",
                    "targets": ["q_proj", "v_proj"],
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                }
            },
            "train": {
                "head_warmup_epochs": 1,
                "robust_epochs": 1,
                "batch_size": 8,
                "grad_accumulation_steps": 1,
                "amp": False,
                "weight_decay": 0.05,
                "grad_clip_norm": 1.0,
                "lr": {"classifier": 1e-3, "lora": 1e-3, "layer_norm": 1e-4},
                "scheduler": {"name": "cosine", "warmup_ratio": 0.05},
            },
            "loss": {
                "supervised": "ce",
                "gce_q": 0.7,
                "anchor": {"min_weight": 0.01, "uncertain_weight": 0.05},
            },
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("smoke_stage3")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    cfg = _make_cfg()

    model = build_model(cfg, _NUM_CLASSES, device=device)
    model.train()

    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_trainable > 0, "没有可训练参数"
    print(f"可训练参数: {n_trainable}")

    # 1) 前向
    px = torch.randn(2, 3, 224, 224).to(device_obj)
    z, logits = model(px)
    assert z.shape == (2, _DIM), f"z 形状 {z.shape}"
    assert logits.shape == (2, _NUM_CLASSES), f"logits 形状 {logits.shape}"
    print(f"前向 OK: z {tuple(z.shape)} logits {tuple(logits.shape)}")

    # 2) 损失 + 反向
    labels = torch.randint(0, _NUM_CLASSES, (2,)).to(device_obj)
    q0 = torch.rand(2).to(device_obj)
    z0 = torch.randn(2, _DIM).to(device_obj)
    z0 = z0 / z0.norm(dim=-1, keepdim=True)
    loss, parts = compute_total_loss(logits, labels, z, z0, q0)
    assert torch.isfinite(loss), "损失非有限"
    model.zero_grad(set_to_none=True)
    loss.backward()
    lora_grads = [
        p.grad
        for n, p in model.named_parameters()
        if p.requires_grad and ("lora_A" in n or "lora_B" in n)
    ]
    assert len(lora_grads) > 0 and all(g is not None and torch.isfinite(g).all() for g in lora_grads), "LoRA 梯度异常"
    print(f"反向 OK: loss={float(loss.detach()):.4f} (sup={parts['sup']:.4f} anchor={parts['anchor']:.4f})")

    # 3) checkpoint 存/读
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "best.pt"
        save_checkpoint(model, ckpt)
        assert ckpt.exists()
        load_checkpoint(model, ckpt)
        print("checkpoint 存/读 OK")

    # 4) 分类头预热（缓存特征）
    z0_np = np.random.randn(64, _DIM).astype(np.float32)
    z0_np /= np.linalg.norm(z0_np, axis=1, keepdims=True)
    labels_np = np.random.randint(0, _NUM_CLASSES, (64,))
    train_head_warmup(model, z0_np, labels_np, cfg, device_obj, logger)
    print("head warmup OK")

    # 5) robust 微调 1 epoch（假数据）
    train_loader = DataLoader(
        _FakeTrain(), batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn("train")
    )
    val_loader = DataLoader(
        _FakeVal(), batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn("val")
    )
    with tempfile.TemporaryDirectory() as tmp:
        train_robust(model, train_loader, val_loader, cfg, device_obj, 1, Path(tmp), logger)

    print("STAGE3 SMOKE OK")


if __name__ == "__main__":
    main()
