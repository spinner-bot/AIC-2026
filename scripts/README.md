# scripts/

存放一键脚本：数据下载、训练、评测等。

- `download_model.sh`：下载 CLIP ViT-B/32 权重到 `models/`（✅ 已实现）
- `clean_dataset.py`：按 V3 执行非破坏性 ZIP 审计、解压、重复分组和固定划分（✅ 已实现）
- `download_data.sh`：下载赛题数据（待补）
- `run_train.sh`：一键训练（待补）
- `run_eval.sh`：一键评测/生成提交（待补）

## V3 数据清洗

原始 ZIP 不会被修改。可读截断图保留；不可读项写入异常清单；训练集的
完全重复和近重复图保留并归入同一 group；测试集只做完整性审计，不参与
训练统计。

```powershell
python scripts/clean_dataset.py `
  --train-zip "D:\AIC数据集\train.zip" `
  --test-zip "D:\AIC数据集\test.zip" `
  --output-dir "D:\AIC数据集\cleaned" `
  --expected-classes 500 `
  --expected-train 103218 `
  --expected-test 24967
```

结果位于 `cleaned/train`、`cleaned/test` 和 `cleaned/reports`。脚本可安全重跑：
已有文件内容相同则复用，内容不同则拒绝覆盖。
