#!/usr/bin/env bash
# 下载 CLIP ViT-B/32（openai/clip-vit-base-patch32）到本地 models/ 目录。
#
# 用法:
#   bash scripts/download_model.sh
#
# 说明:
#   - 模型是公开且字节固定的，队友各自跑一遍即得到与本机一致的权重，无需 git 同步大文件。
#   - 默认走 hf-mirror.com（国内更快），失败自动回退 huggingface.co。
#   - 只下载 PyTorch 所需文件（跳过 flax / tf 版本），大文件支持断点续传。
#   - 幂等：已完整下载的文件自动跳过，可重复运行。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_ID="openai/clip-vit-base-patch32"
DEST="$PROJECT_ROOT/models/clip-vit-base-patch32"
BIG_SIZE=605247071   # pytorch_model.bin 官方字节数

SMALL_FILES=(
  config.json
  preprocessor_config.json
  tokenizer_config.json
  tokenizer.json
  vocab.json
  merges.txt
  special_tokens_map.json
)
MIRRORS=("https://hf-mirror.com" "https://huggingface.co")

mkdir -p "$DEST"

download() {
  local file="$1" target="$DEST/$1" url base
  for base in "${MIRRORS[@]}"; do
    url="$base/$MODEL_ID/resolve/main/$file"
    echo "  下载: $file  <-  $base"
    if curl -L -C - --retry 3 -o "$target" "$url"; then
      return 0
    fi
    echo "  … 该镜像失败，切换下一个"
    rm -f "$target"
  done
  return 1
}

echo "==> 目标目录: $DEST"

# 小文件：非空即视为完整
for f in "${SMALL_FILES[@]}"; do
  if [ -s "$DEST/$f" ]; then
    echo "  跳过（已存在）: $f"
  else
    download "$f"
  fi
done

# 大权重：按官方字节数校验，未完整则续传
if [ -f "$DEST/pytorch_model.bin" ] && [ "$(wc -c < "$DEST/pytorch_model.bin")" = "$BIG_SIZE" ]; then
  echo "  跳过（已完整）: pytorch_model.bin"
else
  download "pytorch_model.bin"
  if [ "$(wc -c < "$DEST/pytorch_model.bin")" != "$BIG_SIZE" ]; then
    echo "❌ pytorch_model.bin 大小不匹配（预期 $BIG_SIZE），请重试"
    exit 1
  fi
fi

echo ""
echo "✅ 模型就绪: $DEST"
du -sh "$DEST" 2>/dev/null || true
