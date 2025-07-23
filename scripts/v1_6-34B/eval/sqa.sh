#!/bin/bash

CKPT="llava-v1.6-34b"
METHOD="fastervlm"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa_science \
    --model-path /data/models/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --single-pred-prompt \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode chatml_direct

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}/${METHOD}/${PARAM}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}/${METHOD}/${PARAM}_result.json \
