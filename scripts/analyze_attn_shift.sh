#!/bin/bash

CKPT="llava-v1.5-7b"
FILE="gqa_only.jsonl"
CUDA_VISIBLE_DEVICES="2" python -m llava.eval.analyze_attn_shift \
    --model-path /data/models/${CKPT} \
    --question-file ./playground/data/analysis/${FILE} \
    --image-folder ./playground/data/train \
    --output-folder ./playground/data/analysis/attn_shift \
    --visual-token-num 576 \
    --temperature 0 \
    --conv-mode vicuna_v1
