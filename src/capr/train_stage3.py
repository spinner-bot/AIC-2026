"""Stage 3：MVP 微调训练循环。

流程（V3 8 / 5.2）：
    1) 分类头预热：直接用 Stage 1 缓存特征训 cosine 分类头（不加载图像，秒级）
    2) robust 微调：LoRA + 分类头 + 最后 4 层 LayerNorm，
       损失 = 可靠度加权 CE/GCE + 自适应特征锚定
    3) 每个 epoch 在 val 上评估 Top-1，保存 best checkpoint
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from .config import Config, project_root
from .dataset import build_datasets, collate_fn
from .losses import compute_total_loss
from .model import build_model, save_checkpoint

_CLIP_MODEL_DIR = "models/clip-vit-base-patch32"


def _configure_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(out_dir / "stage3.log", encoding="utf-8", mode="w")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def train_head_warmup(
    model, z0: np.ndarray, labels: np.ndarray, cfg: Config, device: torch.device, logger: logging.Logger
) -> None:
    """用冻结特征训练分类头（普通 CE）。"""
    epochs = int(cfg.train.get("head_warmup_epochs", 5))
    lr = float(cfg.train.lr.get("classifier", 0.0005))
    wd = float(cfg.train.get("weight_decay", 0.05))
    bs = int(cfg.train.get("batch_size", 64))

    z0_t = torch.from_numpy(np.asarray(z0, dtype=np.float32)).to(device)
    labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    n = len(labels_t)

    opt = torch.optim.AdamW(
        [
            {"params": [model.classifier_weight], "lr": lr, "weight_decay": wd},
            {"params": [model.logit_scale], "lr": lr, "weight_decay": 0.0},
        ]
    )

    model.train()
    logger.info(f"== 分类头预热：{epochs} epoch，样本 {n}，lr {lr} ==")
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            logits = model.classify(z0_t[idx])
            loss = F.cross_entropy(logits, labels_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            cnt += len(idx)
        logger.info(f"  [head warmup] epoch {ep + 1}/{epochs}  loss={total / cnt:.4f}")


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader: DataLoader, device: torch.device) -> float:
    """val Top-1（单模型单裁剪单次前向）。"""
    model.eval()
    correct = total = 0
    for px, labels in val_loader:
        px = px.to(device)
        labels = labels.to(device)
        _, logits = model(px)
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += len(labels)
    model.train()
    return correct / total if total else 0.0


def train_robust(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    epochs: int,
    out_dir: Path,
    logger: logging.Logger,
) -> float:
    """完整 LoRA 微调：可靠度加权 CE/GCE + 锚定。返回 best val Top-1。"""
    lr_cfg = cfg.train.lr.to_dict()
    wd = float(cfg.train.get("weight_decay", 0.05))
    grad_clip = float(cfg.train.get("grad_clip_norm", 1.0))
    accum = int(cfg.train.get("grad_accumulation_steps", 4))
    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"

    supervised = str(cfg.loss.get("supervised", "ce"))
    gce_q = float(cfg.loss.get("gce_q", 0.7))
    anchor_min = float(cfg.loss.anchor.get("min_weight", 0.01))
    anchor_uncertain = float(cfg.loss.anchor.get("uncertain_weight", 0.05))

    # 噪声硬处理：q0 硬丢弃阈值（启用时低可靠度样本不参与分类监督）
    noise_cfg = cfg.get("noise", None)
    hard_threshold = None
    if noise_cfg is not None and noise_cfg.get("hard_drop", False):
        hard_threshold = float(noise_cfg.get("hard_threshold", 0.3))

    # sched.step() 每 accum 个 batch 调用一次，故调度步数按 optimizer step 计算
    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / train_loader.batch_size))
    opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / accum))
    total_steps = epochs * opt_steps_per_epoch
    warmup_ratio = float(cfg.train.scheduler.get("warmup_ratio", 0.05))
    warmup_steps = int(total_steps * warmup_ratio)

    opt = torch.optim.AdamW(model.param_groups(lr_cfg, weight_decay=wd))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps))
    # fp16 下梯度 × init_scale 过大时会溢出产生 NaN（而非 inf），GradScaler 只检测 inf
    # 无法自愈，因此用保守初始值（scaler 训练中会自适应增长到合适值）。
    init_scale = float(cfg.train.get("grad_scaler_init_scale", 2048.0))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=init_scale)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    logger.info(
        f"== robust 微调：{epochs} epoch，batch {train_loader.batch_size}×{accum}，"
        f"loss={supervised}+anchor，amp={use_amp} =="
    )

    best_acc = 0.0
    best_path = out_dir / "best.pt"
    patience = int(cfg.train.get("early_stop_patience", 0))
    epochs_no_improve = 0
    global_step = 0

    for ep in range(epochs):
        model.train()
        t0 = time.monotonic()
        run_sup = run_anchor = 0.0
        opt.zero_grad(set_to_none=True)

        for bi, (px, labels, q0, z0) in enumerate(train_loader):
            px = px.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            q0 = q0.to(device, non_blocking=True)
            z0 = z0.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                z, logits = model(px)
                loss, parts = compute_total_loss(
                    logits, labels, z, z0, q0,
                    supervised=supervised, gce_q=gce_q,
                    anchor_min=anchor_min, anchor_uncertain=anchor_uncertain,
                    hard_threshold=hard_threshold,
                )
            loss = loss / accum

            scaler.scale(loss).backward()

            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            run_sup += float(parts["sup"]) * len(labels)
            run_anchor += float(parts["anchor"]) * len(labels)
            global_step += 1

            if global_step % 200 == 0:
                logger.info(
                    f"  [ep {ep + 1}/{epochs} step {global_step}] "
                    f"sup={parts['sup']:.4f} anchor={parts['anchor']:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}"
                )

        n_train = len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        logger.info(
            f"  [ep {ep + 1}/{epochs}] 用时 {time.monotonic() - t0:.1f}s  "
            f"sup={run_sup / n_train:.4f} anchor={run_anchor / n_train:.4f}  val_top1={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, best_path)
            epochs_no_improve = 0
            logger.info(f"  -> 保存 best checkpoint ({best_acc:.4f})")
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"  == 早停：连续 {patience} epoch 无提升，停止于 epoch {ep + 1} ==")
            break

    logger.info(f"== robust 完成，best val Top-1 = {best_acc:.4f} ==")
    return best_acc


def main(cfg: Config, device: str = "cuda", epochs: int | None = None) -> float:
    out_dir = cfg.path(cfg.experiment.output_dir)
    logger = _configure_logging(out_dir)

    device_obj = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    split_path = out_dir / "split.csv"
    q0_path = out_dir / "q0.csv"
    feat_dir = cfg.path(cfg.features.get("cache_dir")) or out_dir / "features"
    feat_path = feat_dir / "features.npy"

    for p in (split_path, q0_path, feat_path):
        if not p.exists():
            raise FileNotFoundError(f"前置产物缺失: {p}（先运行 run_stage0/1/2）")

    split = pd.read_csv(split_path)
    q0 = pd.read_csv(q0_path)
    feats = np.load(feat_path)
    assert len(split) == len(feats), "split 与 features 行数不一致"

    num_classes = int(split["class_idx"].max()) + 1
    image_size = int(cfg.data.get("image_size", 224))
    processor = CLIPProcessor.from_pretrained(
        str(project_root() / _CLIP_MODEL_DIR),
        size={"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
    )

    train_ds, val_ds, train_items, _ = build_datasets(
        split, q0, feats, processor, image_size=image_size
    )
    logger.info(f"== Stage 3 MVP 微调 ==")
    logger.info(f"  训练 {len(train_ds)} / 验证 {len(val_ds)} / 类别 {num_classes}")

    model = build_model(cfg, num_classes, device=device)

    # 1) 分类头预热（缓存特征）
    train_mask = (split["fold"] != "val").values
    z0_train = feats[train_mask]
    labels_train = split.loc[train_mask, "class_idx"].values.astype(int)
    train_head_warmup(model, z0_train, labels_train, cfg, device_obj, logger)

    # 2) robust 微调
    bs = int(cfg.train.get("batch_size", 64))
    num_workers = int(cfg.train.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=collate_fn("train")
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate_fn("val")
    )
    # epoch 0 基线：分类头预热后（冻结特征 + 余弦分类头），尚未 LoRA 微调
    baseline_acc = evaluate(model, val_loader, device_obj)
    logger.info(f"== epoch 0 基线（冻结特征 + 分类头预热）val_top1 = {baseline_acc:.4f} ==")
    robust_epochs = epochs if epochs is not None else int(cfg.train.get("robust_epochs", 35))
    return train_robust(model, train_loader, val_loader, cfg, device_obj, robust_epochs, out_dir, logger)
