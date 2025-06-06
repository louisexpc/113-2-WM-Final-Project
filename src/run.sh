#!/bin/bash

set -e
set -o pipefail

# 判斷 Python 執行指令
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "找不到 python 或 python3，請先安裝 Python！"
    exit 1
fi

echo "=== Step 1: 資料下載 ==="
cd dataset
if [ ! -x ./download.sh ]; then
    chmod +x ./download.sh
fi
./download.sh
cd ..

echo "=== Step 2: 執行主程式 ==="
if ! $PYTHON run.py; then
    echo "run.py 執行失敗，終止流程。"
    exit 1
fi

echo "=== Step 3: Session Split ==="
SESSION_INPUT="dataset/generated_real_name_4.pkl"
SESSION_OUTPUT_PREFIX="origin_5_4_mapping"
NUM_NEG=99
SEED=42

if ! $PYTHON run_session_split.py \
    --input_path "$SESSION_INPUT" \
    --output_prefix "$SESSION_OUTPUT_PREFIX" \
    --num_neg "$NUM_NEG" \
    --seed "$SEED"; then
    echo "run_session_split.py 執行失敗，終止流程。"
    exit 1
fi

echo "=== 全部流程完成 ==="
