# 数据集说明

> ⚠️ 数据下载链接**通过团队私有渠道（QQ 群公告 / 私聊）分发，不写入本公开仓库**。
> 赛题要求数据仅限比赛使用、禁止外泄传播，请勿将链接提交到 GitHub。

## 数据规模（三阶段）

| 阶段 | 类别数 | 训练图 | 测试图 | 特点 |
|---|---|---|---|---|
| 初赛 | 500 | 103,218 | 24,967 | 含噪声标签 |
| 复赛 | 750 | 148,695 | 37,444 | 噪声 + 长尾 |
| 半决赛 | 500 | 90,197 | 24,912 | 噪声 + 长尾 |

## 目录结构要求

下载解压后，把训练集放到 `data/train`，按类别文件夹组织：

```text
data/
  train/                 # 训练集（按类别文件夹）
    0001/xxx.jpg
    0002/yyy.jpg
    ...
  test.txt               # 测试图片列表（若官方提供）
```

> 若压缩包结构与上述不同，解压后调整成此结构即可。`data/` 已被 `.gitignore` 排除，不会误提交。

## 下载后校验

```bash
python scripts/check_data.py --config configs/v3.yaml
```

校验通过后，依次运行：

```bash
python scripts/run_stage0.py --config configs/v3.yaml   # 数据审计 + group split
python scripts/run_stage1.py --config configs/v3.yaml   # 冻结 CLIP 特征缓存
python scripts/run_stage2.py --config configs/v3.yaml   # q0 可靠度诊断
```

## 提交格式（评测用）

- 结果文件 `pred_results.csv`，两列：`图片文件名,类别编号`（编号 4 位补零）
- 压缩为 zip 提交
