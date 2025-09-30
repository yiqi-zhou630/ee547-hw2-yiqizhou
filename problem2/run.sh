#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_papers.json> <output_dir> [epochs] [batch_size]"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"
EPOCHS="${3:-50}"
BATCH_SIZE="${4:-32}"

# 校验输入文件
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file $INPUT_FILE not found"
  exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 解析为绝对路径（POSIX 方式）
host_input="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"
host_output="$(cd "$OUTPUT_DIR" && pwd)"

# 在 Git Bash/ Cygwin 上把 /c/ 路径转成 C:/ 形式，避免卷挂载和冒号冲突
if command -v cygpath >/dev/null 2>&1; then
  host_input="$(cygpath -m "$host_input")"
  host_output="$(cygpath -m "$host_output")"
fi

echo "Training embeddings with the following settings:"
echo "  Input: $host_input"
echo "  Output: $host_output"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""


MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker run --rm \
  --name arxiv-embeddings \
  -v "$host_input":/data/input/papers.json:ro \
  -v "$host_output":/data/output \
  arxiv-embeddings:latest \
  /data/input/papers.json /data/output --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"


echo ""
echo "Training complete. Output files:"
ls -la "$host_output"
