# Project Status

> 更新于 2026-09-30

## Current Phase

算法4 — 面向噪声标签数据的细粒度图像识别鲁棒微调
（CLIP ViT-B/32 · 指标 Top-1 Accuracy · 技术路线 CAPR-CLIP V3）

## Completed

- [x] 组队 / 能力评估 / 赛题收集 / 选题 → 算法4（噪声图像识别）
- [x] 技术路线定稿 → CAPR-CLIP V3（LoRA + 可靠度加权 + 自适应锚定 + cRT）
- [x] 环境搭建（conda `aic` / torch 2.11+cu128 / CLIP ViT-B/32 权重下载）
- [x] **Stage 0 数据清洗**：103218 训练 / 24967 测试，异常 0，500 类；5762 重复组（13062 张）；train/val 93097/10121；head/mid/tail 34715/35083/33420
- [x] **Stage 1 冻结特征**：`features.npy` (103218, 512)，L2 归一化
- [x] **Stage 2 q0 可靠度诊断**：OOF + 留一鲁棒原型 + mutual-kNN 融合 → `q0.csv` + `signals.npz`
- [x] **Stage 3 MVP 微调**：分类头预热（缓存特征 5 epoch）+ 可靠度加权 LoRA 微调 + 自适应锚定
      · AutoDL RTX 5090 完成 35 epoch，全程约 2 小时 29 分
      · 最佳 **val Top-1 = 68.20%**（epoch 1），末轮 64.77%
      · best checkpoint 已落盘 `best.pt`；后期训练损失继续下降但验证准确率回落

## In Progress

- Stage 3 结果复盘与短程早停实验设计

## Upcoming

- [ ] 评估并保存分类头预热后的 epoch-0 基线，加入 early stopping
- [ ] 修复 Stage 2 重复组在 OOF / 原型 / kNN 中的组内泄漏后重算 q0
- [ ] 对 B0 / B1 / R1 / R2 做固定划分、短程、多种子消融
- [ ] **Stage 5 延迟分类器校准**（数据近似均衡，作为条件分支而非默认主线）
- [ ] Stage 6 容量与分辨率 / Stage 7 阶段重训
- [ ] Final submission

## 数据与产物位置

- 原始数据：`D:\初赛数据集\`（train.zip / test.zip，**不写入仓库**）
- 清洗产物：`D:\初赛数据集\cleaned\`（`train/` `test/` `reports/`）
- 特征缓存：`outputs/capr_clip_v3/features/features.npy`
- 划分产物：`outputs/capr_clip_v3/split.csv` / `manifest.csv`
- 可靠度：`outputs/capr_clip_v3/q0.csv` / `signals.npz`
- Stage 3 模型：`outputs/capr_clip_v3/best.pt`（仅可训练参数）

## 关键实现要点（Stage 3）

- LoRA 手写注入最后 4 层 Q/V（无 peft 依赖），`src/capr/lora.py`
- 特征口径与 Stage 1 对齐：`z = L2_norm(visual_projection(vision_pooler))`，锚定损失成立
- cosine 分类头随机初始化（类别为 4 位编号无文本名）+ 可学习 `logit_scale`
- **AMP `grad_scaler_init_scale: 2048`**（默认 65536 会使 fp16 梯度溢出产生 NaN，GradScaler 只检测 inf 无法自愈）
