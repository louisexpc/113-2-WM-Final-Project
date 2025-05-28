#!/bin/bash

python main.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --session_path /home/guest/r13946008/HW/IR/final/dataset/sessions_5_4_30_mapping.pkl \
    --category_csv /home/guest/r13946008/HW/IR/final/dataset/product_types.csv \
    --prompt_path /home/guest/r13946008/HW/IR/final/get_product_type_from_history_prompt.txt \
    --max_users 1 \
    --save_path "testing.pkl" \
    --mapping_path dataset/articles_mapping.csv