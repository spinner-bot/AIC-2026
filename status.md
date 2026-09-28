# Project Status

> 更新于 2026-09-28

## Current Phase

算法4 — 面向噪声标签数据的细粒度图像识别鲁棒微调
（CLIP ViT-B/32 · 指标 Top-1 Accuracy · 技术路线 CAPR-CLIP V3）

## Completed

- [x] 组队 / 能力评估 / 赛题收集 / 选题 → 算法4（噪声图像识别）
- [x] 技术路线定稿 → CAPR-CLIP V3（LoRA + 可靠度加权 + 自适应锚定 + cRT）
- [x] 环境搭建（conda `aic` / torch 2.11+cu128 / CLIP ViT-B/32 权重下载）
- [x] **Stage 0 数据清洗**：103218 训练 / 24967 测试，异常 0，500 类；5762 重复组（13062 张）；train/val 93097/10121；head/mid/tail 34715/35083/33420
- [x] **Stage 1 冻结特征**：`features.npy` (103218, 512)，L2 归一化

## In Progress

- [ ] **Stage 2 q0 可靠度诊断**（OOF / 鲁棒原型 / mutual-kNN → `q0.csv`）

## Upcoming

- [ ] **Stage 3 MVP 微调**（分类头预热 + LoRA + 可靠度加权 CE/GCE + 自适应锚定）— 需新写训练模块
- [ ] Stage 5 延迟 cRT 校准
- [ ] Stage 6 容量与分辨率 / Stage 7 阶段重训
- [ ] Final submission

## 数据与产物位置

- 原始数据：`D:\初赛数据集\`（train.zip / test.zip，**不写入仓库**）
- 清洗产物：`D:\初赛数据集\cleaned\`（`train/` `test/` `reports/`）
- 特征缓存：`outputs/capr_clip_v3/features/features.npy`
- 划分产物：`outputs/capr_clip_v3/split.csv` / `manifest.csv`
