# AIC 数据清洗脚本：终端使用指南

本指南说明如何在终端运行 `scripts/clean_dataset.py`。脚本遵循 CAPR-CLIP
技术路线 V3：保留可读图片和重复图片，对重复/近重复训练图分组，不使用
测试集统计训练模型，并且不修改原始 ZIP。

## 1. 路径在哪里填写

不需要修改 Python 文件，也不需要配置环境变量。每次运行时，在终端命令中
填写下面三个参数：

| 参数 | 含义 | 示例 |
|---|---|---|
| `--train-zip` | 本机训练集 ZIP 路径 | `E:\AIC\train.zip` |
| `--test-zip` | 本机测试集 ZIP 路径 | `E:\AIC\test.zip` |
| `--output-dir` | 清洗结果保存目录 | `E:\AIC\cleaned` |

路径中含空格或中文时，必须使用英文双引号包住完整路径。

## 2. 准备代码和依赖

数据集不会存放在 GitHub。请先通过比赛官方或团队授权渠道取得
`train.zip` 和 `test.zip`。

克隆代码并进入仓库根目录：

```powershell
git clone https://github.com/spinner-bot/AIC-2026.git
cd AIC-2026
```

只运行数据清洗所需的最小依赖：

```powershell
python -m pip install pillow ImageHash pandas PyYAML
```

也可以安装项目全部依赖：

```powershell
python -m pip install -r requirements.txt
```

## 3. Windows PowerShell 用法

在仓库根目录运行。PowerShell 多行续行符是反引号 `` ` ``：

```powershell
python scripts/clean_dataset.py `
  --train-zip "E:\AIC\train.zip" `
  --test-zip "E:\AIC\test.zip" `
  --output-dir "E:\AIC\cleaned" `
  --expected-classes 500 `
  --expected-train 103218 `
  --expected-test 24967
```

也可以写成一行：

```powershell
python scripts/clean_dataset.py --train-zip "E:\AIC\train.zip" --test-zip "E:\AIC\test.zip" --output-dir "E:\AIC\cleaned" --expected-classes 500 --expected-train 103218 --expected-test 24967
```

## 4. Windows CMD 用法

CMD 多行续行符是 `^`：

```bat
python scripts\clean_dataset.py ^
  --train-zip "E:\AIC\train.zip" ^
  --test-zip "E:\AIC\test.zip" ^
  --output-dir "E:\AIC\cleaned" ^
  --expected-classes 500 ^
  --expected-train 103218 ^
  --expected-test 24967
```

## 5. Linux/macOS 用法

Bash、Zsh 多行续行符是 `\`：

```bash
python scripts/clean_dataset.py \
  --train-zip "/data/AIC/train.zip" \
  --test-zip "/data/AIC/test.zip" \
  --output-dir "/data/AIC/cleaned" \
  --expected-classes 500 \
  --expected-train 103218 \
  --expected-test 24967
```

## 6. 不同比赛阶段的数量参数

三个 `--expected-*` 参数用于防止拿错数据，可以省略；建议按当前比赛阶段填写：

| 阶段 | `--expected-classes` | `--expected-train` | `--expected-test` |
|---|---:|---:|---:|
| 初赛 | 500 | 103218 | 24967 |
| 复赛 | 750 | 148695 | 37444 |
| 半决赛 | 500 | 90197 | 24912 |

这些参数只校验数量，不会改变图片或标签。

## 7. 清洗结果

假设 `--output-dir` 是 `E:\AIC\cleaned`，完成后得到：

```text
E:\AIC\cleaned\
├─ train\                         可用训练图片，保留原类别目录
├─ test\                          可用测试图片
├─ quarantine\                   不可解码原始字节（仅存在异常时生成）
└─ reports\
   ├─ train_manifest.csv          训练集路径、类别、尺寸、哈希、解码状态
   ├─ test_manifest.csv           测试集完整性审计结果
   ├─ split.csv                   group_id、train/val 折、头中尾划分
   ├─ duplicate_groups.csv        重复和近重复训练图分组
   ├─ anomaly.json                异常清单
   └─ summary.json                数量、数据指纹和参数汇总
```

原始 `train.zip` 和 `test.zip` 不会被删除、覆盖或改写。重复训练图也不会被
删除，只会记录为同一 `group_id`，防止跨训练/验证折泄漏。

## 8. 查看全部参数

```powershell
python scripts/clean_dataset.py --help
```

常用可选参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--near-dup-hamming` | 8 | 感知哈希近重复阈值，V3 默认值 |
| `--val-ratio` | 0.10 | 固定验证集比例 |
| `--oof-folds` | 3 | 小类别 OOF 折数 |
| `--seed` | 42 | 固定划分随机种子 |
| `--progress-every` | 1000 | 每处理多少张输出一次进度 |

如无消融实验需求，请保留这些默认值。

## 9. 安全重跑

脚本可以使用同一命令重跑：

- 已存在且内容相同的图片会被复用；
- 已存在但内容不同的图片不会被覆盖，会进入异常报告；
- 原始 ZIP 始终不变。

## 10. 常见问题

### 提示 `No module named ...`

重新安装最小依赖：

```powershell
python -m pip install pillow ImageHash pandas PyYAML
```

### 提示 ZIP 不存在

检查 `--train-zip` 和 `--test-zip` 后面的路径。路径必须指向 ZIP 文件本身，
而不是文件夹。

### 数据在其他盘符或目录

只修改命令中的三个路径参数，不要修改脚本源码。例如：

```powershell
python scripts/clean_dataset.py --train-zip "F:\比赛数据\train.zip" --test-zip "F:\比赛数据\test.zip" --output-dir "F:\比赛数据\cleaned"
```

### 是否要把数据上传到 GitHub

不要。GitHub 只保存清洗脚本；每位使用者在自己的电脑或服务器上填写本地
数据路径并运行。比赛数据、下载链接和清洗后的图片均不应提交到公开仓库。
