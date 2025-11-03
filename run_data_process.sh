#!/bin/bash
# Data construction pipeline for Thinking in Character

set -e

# Usage: bash run_data_process.sh <data_dir> <model_dir>
# Example: bash run_data_process.sh data/ models/ 

DATA_DIR=${1:-data}
MODEL_DIR=${2:-models}

# 下载原始数据

# Step 1: Generate long reason examples
python scripts/long_reason_example.py \
    --data_path $DATA_DIR/rolebench_train_main_datas_best_specific.json \
    --model_path $MODEL_DIR/QwQ-32B \
    --output_path $DATA_DIR/rolebench_train_main_datas_best_specific_with_aware_reason.json

python scripts/long_reason_example.py \
    --data_path $DATA_DIR/rolebench_train_main_datas_best_general.json \
    --model_path $MODEL_DIR/QwQ-32B \
    --output_path $DATA_DIR/rolebench_train_main_datas_best_general_with_aware_reason.json

# Step 2: Generate general contrastive samples
python scripts/long_reason_sample_contrastive_general.py \
    --data_path $DATA_DIR/rolebench_train_main_datas_best_general.json \
    --model_path $MODEL_DIR/QwQ-32B \
    --output_path $DATA_DIR/rolebench_train_main_datas_best_general_with_aware_reason_contrastive_qwq.json

# Step 3: Generate specific contrastive samples
python scripts/long_reason_sample_contrastive_specific.py \
    --data_path $DATA_DIR/rolebench_train_main_datas_best_specific.json \
    --model_path $MODEL_DIR/QwQ-32B \
    --output_path $DATA_DIR/rolebench_train_main_datas_best_specific_with_aware_reason_contrastive_qwq.json

# Step 4
python scripts/data_dpo_convert_sft.py \
    --input_file_1 $DATA_DIR/rolebench_train_main_datas_best_specific_with_aware_reason_contrastive_qwq.json \
    --input_file_2 $DATA_DIR/rolebench_train_main_datas_best_general_with_aware_reason_contrastive_qwq.json \
    --output_file $DATA_DIR/rolebench_train_main_datas_best_all_with_aware_reason_contrastive_qwq_sft.json

echo "All data processing steps completed." 