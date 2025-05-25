#!/bin/bash

# 設定要使用的 GPU
GPU_ID=5

# 設定批次大小
BATCH_SIZE=10

# 設定輸入與輸出檔案
INPUT_PATH="data/llama3_enriched_sessions4_final.pkl"
OUTPUT_PATH="data/generated_product_name_4.pkl"

python main_product_name.py \
    --gpu $GPU_ID \
    --batch_size $BATCH_SIZE \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH