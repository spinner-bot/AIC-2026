# AIC 图片数据清洗

将 `train.zip` 和 `test.zip` 放在本目录，运行 `python clean_dataset.py`。Windows 用户也可以双击 `run_clean.bat`，它会在需要时安装 Pillow。

脚本保留原始压缩包，并在同一目录生成 `train_clean.zip`、`test_clean.zip`、`cleaning_report.json` 和 `removed_files.csv`。已有清洗结果时，脚本会停止；需要重新生成时运行 `python clean_dataset.py --overwrite`。

处理规则：完整解码每张图片并检查 ZIP 校验值；扩展名为 `.jpg`、但实际为 PNG、WEBP、GIF、MPO 或 BMP 的图片会转换为真正的 JPEG，非 RGB 的 JPEG 也会转为 RGB。透明部分填充白色；动图只保留第一帧。移除空文件、无法读取的图片、目录结构不符的文件和按原始字节完全重复的图片。若同一张训练图片出现在不同类别中，会移除这些冲突图片；若训练图片与测试图片完全相同，会从训练集中移除，以免评估时出现数据泄漏。测试集不参与计算训练参数或类别规则。

原本就是 RGB JPEG 的图片不会重新编码；其他格式转换为 JPEG 时可能有轻微画质变化。图片不会调整大小。模型训练时需要的缩放、归一化等处理应在训练流程中完成。这里的重复检查仅识别原始字节完全相同的文件。

脚本也接受 `--input-dir` 和 `--output-dir` 参数。GitHub 只需上传 `.py`、`.bat`、`requirements.txt`、`README.md` 和 `.gitignore`；压缩包已被 `.gitignore` 排除。

## 与本仓库现有训练流程的关系

本脚本是一种可选的清洗方案，会直接移除跨类别重复图片。本仓库现有技术路线则保留这些图片，将其标记后交给可信度模块处理。运行本仓库原有的训练流程时，请按项目文档选择数据版本，不要直接用本脚本生成的压缩包覆盖原始数据。
