# AdaV: Adaptive Attention with Vision Tokens for Efficient Vision-Language Models

[![Paper](https://img.shields.io/badge/ACL%202025-Findings-blue)](https://aclanthology.org/2025.findings-acl.258.pdf)
[![Codebase](https://img.shields.io/badge/Based%20on-FasterVLM-green)](https://github.com/Theia-4869/FasterVLM)

This repository contains the implementation for the ACL 2025 Findings paper "ADAV: Adaptive Attention with Vision Tokens for Efficient Vision-Language Models". The code extends the [FasterVLM](https://github.com/Theia-4869/FasterVLM) framework with our novel adaptive token selection method.

## 🔧 Setup Instructions

1. **Clone and install base environment**:
```bash
git clone https://github.com/Theia-4869/FasterVLM.git
cd FasterVLM
conda env create -f environment.yml
conda activate fastervlm
```

2. **Merge AdaV codebase**:
```bash
git clone https://github.com/JiayiV/AdaV
cp -r AdaV/* FasterVLM/
```

## 🚀 Reproduction Guide

### Critical Configuration Steps
To reproduce our results, you **must** modify these configurations:

1. **Set method to 'adav'**  
   Edit `scripts/adav_configs.json`:
   ```json
   {
     "method": "adav",  // ⚠️ Ensure this is set to 'adav'
     ...
   }
   ```

2. **Adjust token count**  
   In the same file, modify `overall_tokens` to your desired value:
   ```json
   {
     "overall_tokens": 144,  // ✏️ reduction rate = 95% for LLaVA-Next Models
     ...
   }
   ```

### Running Experiments

For LLaVA-Next Models (7/13B), utilize the scripts from `scripts/v1_6`
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_6/eval/vqav2.sh 144
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/eval/textvqa.sh 58
```

For LLaVA-Next 34B, utilize the scripts from `scripts/v1_6_34B`, since LLaVA-Next-34B utilize Yi-34B LLM, from a different LLM series.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_6_34B/eval/vqav2.sh 144
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6_34B/eval/textvqa.sh 58
```

### 📊 Expected Performance
| Method              | GQA    | SQA-IMG | TextVQA | POPE   | MME      | MMB    | MM-Vet | Average  |
|---------------------|--------|---------|---------|--------|----------|--------|--------|----------|
| LLaVA-1.5-7B        | 61.94  | 69.51   | 58.21   | 85.88  | 1506.47  | 64.69  | 31.30  | 100.00%  |
| Reduction Rate ~75% |        |         |         |        |          |        |        |          |
| FastV               | 56.58  | 69.11   | 57.38   | 73.74  | 1463.39  | 64.00  | 28.60  | 94.67%   |
| FitPrune            | 59.38  | 69.01   | 56.49   | 80.75  | 1472.86  | 63.92  | 28.40  | 96.22%   |
| SparseVLM           | 55.11  | 69.36   | 55.99   | 77.57  | 1351.65  | 59.54  | 29.90  | 93.22%   |
| FaseterVLM          | 58.34  | 67.92   | 57.07   | 83.46  | 1433.76  | 62.54  | 34.20  | 98.32%   |
| **AdaV**                | 58.38  | 69.31   | 56.66   | 84.72  | 1432.68  | 62.28  | 32.40  | 97.83%   |
| Reduction Rate ~90% |        |         |         |        |          |        |        |          |
| FastV               | 51.20  | 69.81   | 54.75   | 57.30  | 1210.36  | 59.97  | 27.20  | 86.26%   |
| FitPrune            | 49.96  | 68.22   | 56.49   | 53.81  | 1147.46  | 56.27  | 21.80  | 81.62%   |
| SparseVLM           | 48.86  | 67.23   | 55.99   | 65.82  | 1030.61  | 49.05  | 18.60  | 78.87%   |
| FaseterVLM          | 54.91  | 68.91   | 55.28   | 75.85  | 1348.63  | 60.57  | 30.10  | 92.91%   |
| **AdaV**                | 55.30  | 68.82   | 54.53   | 82.33  | 1368.28  | 60.30  | 29.20  | 93.59%   |
| Reduction Rate ~95% |        |         |         |        |          |        |        |          |
| FastV               | 46.03  | 70.00  | 51.56  | 35.47  | 971.56   | 50.17  | 18.90  | 72.48%  |
| FitPrune            | 43.60  | 68.32  | 46.75  | 31.17  | 855.21   | 39.69  | 18.00  | 65.85%  |
| FaseterVLM | 51.51  | 69.56  | 53.09  | 67.24  | 1254.80  | 58.51  | 27.50  | 87.76%  |
| **AdaV**       | 52.96  | 68.42  | 51.89  | 78.04  | 1313.36  | 58.51  | 24.00  | 88.32%  |

## 📖 Citation
If you use our work, please cite:
```bibtex
@inproceedings{author2025adav,
  title={AadV: Adaptive Attention with Vision Tokens for Efficient Vision-Language Models},
  author={Jiayi Han, Liang Du, Yiwen Wu, Guanming Liang, Xiangguo Zhou, Weibo Zheng, Donghong Han, and Zixun Sun},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={4985–4997},
  year={2025}
}
```
