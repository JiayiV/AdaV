#!/bin/bash

CKPT="llava-v1.5-7b"
FILE="llava_pope_test.jsonl"
CUDA_VISIBLE_DEVICES="2" python /data/user/hanjy/FasterVLM/llava/eval/analyze_attn_pope.py \
    --model-path /data/models/${CKPT} \
    --question-file ./playground/data/eval/pope/${FILE} \
    --image-folder ./playground/data/eval/pope/val2014 \
    --output-folder ./playground/data/analysis/pope/attn_shift \
    --visual-token-num 576 \
    --temperature 0 \
    --conv-mode vicuna_v1
