# AutoDL 云端训练部署指南（算法4 · Stage 3 全量）

> 目标：把本地已打通的 Stage 0~2 产物搬到 AutoDL，跑 Stage 3 全量 35 epoch。
> 本地 Stage 3 的 val Top-1 已到 67.87%，全量训练瓶颈在算力（本地约 20h）。

## 1. 选卡与镜像

- **GPU**：RTX 4090 / 4090D（单卡即可）。显存 8GB 就够，瓶颈是算力，4090 性价比最高；想省钱可选 3090。
- **镜像**：选 **Miniconda 基础镜像**（纯净）。不要选带 torch 的框架镜像——本项目锁
  `torch 2.11.0+cu128`，现成镜像对不上，反正要按 `environment.yml` 重装。
- **数据盘**：开 **50GB**（训练集约 10–30GB，留余量）。AutoDL 数据盘一般挂载在 `/root/autodl-tmp`。

## 2. 上传文件

把下面东西传到数据盘（JupyterLab 拖拽 / `scp` / 网盘均可）：

| 文件/目录 | 来源 | 用途 |
|---|---|---|
| 整个仓库（不含 `data/`、`outputs/`、`.git` 可留） | 本地 `git clone` 或压缩上传 | 代码 |
| 清洗后训练集 | `D:\初赛数据集\cleaned\train\` | Stage 3 需原始图像 |
| `features.npy` | `outputs/capr_clip_v3/features/` | 前置产物 |
| `split.csv` / `q0.csv` / `manifest.csv` | `outputs/capr_clip_v3/` | 前置产物 |

> 前置产物缺失会强制重跑 Stage 0/1/2（Stage 1 约 35min），建议直接传。
> CLIP 权重不用传，用 `download_model.sh` 走 hf-mirror 重新下载。

## 3. 建环境

```bash
cd AIC-2026
conda env create -f environment.yml
conda activate aic
bash scripts/download_model.sh          # 下载 CLIP ViT-B/32 权重
```

## 4. 重映射 path（关键，否则读不到图）

本地产物里的 `path` 是 Windows 绝对路径，AutoDL 上失效。用脚本按 `archive_path` 重建：

```bash
python scripts/remap_paths.py \
    --csv outputs/capr_clip_v3/split.csv \
           outputs/capr_clip_v3/q0.csv \
           outputs/capr_clip_v3/manifest.csv \
    --archive-root /root/autodl-tmp/cleaned/train
```

> 只改 `path` 列，不动行序（`features.npy` 与 `split.csv` 按行对齐）。

## 5. 改配置

`configs/v3.yaml`：

- `train.num_workers` 改成 `4`（Linux 多进程安全，加速数据加载）
- 若需重跑 Stage 1，`features.num_workers` 同样改成 `4`

## 6. 运行

```bash
python scripts/run_stage3.py --config configs/v3.yaml
```

35 epoch 在 4090 上约 4–6h。日志在 `outputs/capr_clip_v3/stage3.log`，最佳权重 `best.pt`。

## 7. 注意事项

- **关机释放 GPU**：AutoDL 按小时计费，跑完记得关机；数据盘关机后仍计费，跑完记得导出/释放。
- **transformers 版本**：`environment.yml` 里 transformers 是 loose 版，装最新即可；若
  `CLIPModel` API 报错，`pip install transformers==5.17.0` 与本地对齐。
- **CUDA 驱动**：torch 2.11+cu128 需宿主机驱动支持 CUDA 12.8；4090/3090 实例驱动足够新，无需处理。
- **数据安全**：数据路径、下载链接严禁写入公开仓库，`.gitignore` 已排除 `data/`、`outputs/`。
