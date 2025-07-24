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
| LLaVA-NEXT-7B       | 62.93  | 69.66   | 59.59   | 86.32  | 1513.78  | 67.70  | 42.60  | 100.00%  |
| Reduction Rate ~75% |        |         |         |        |          |        |        |          |
| FastV               | 60.38  | 69.81   | 58.39   | 83.09  | 1477.31  | 65.64  | 41.10  | 97.35%   |
| SparseVLM           | 60.88  | 67.48   | 58.08   | 70.99  | 1446.10  | 63.83  | 38.00  | 93.19%   |
| FaseterVLM          | 61.31  | 68.82   | 59.33   | 85.50  | 1480.68  | 67.35  | 40.40  | 98.14%   |
| **AdaV**                | 62.04  | 69.31   | 58.37   | 87.20  | 1509.36  | 67.35  | 39.70  | 98.49%   |
| Reduction Rate ~90% |        |         |         |        |          |        |        |          |
| FastV               | 55.86  | 69.26   | 55.69   | 71.66  | 1282.86  | 61.60  | 22.70  | 84.81%   |
| SparseVLM           | 56.12  | 68.62   | 51.97   | 63.23  | 1332.22  | 54.47  | 24.70  | 82.08%   |
| FaseterVLM          | 58.12  | 68.12   | 57.57   | 80.00  | 1370.11  | 63.32  | 35.70  | 92.47%   |
| **AdaV**                 | 60.65  | 68.57   | 57.09   | 85.98  | 1503.25  | 66.32  | 36.00  | 96.00%   |
| Reduction Rate ~95% |        |         |         |        |          |        |        |          |
| FastV               | 49.83  | 68.52   | 51.85   | 51.66  | 1079.46  | 54.90  | 21.90  | 75.46%   |
| FaseterVLM | 54.73  | 68.86  | 55.97  | 72.89  | 1225.96  | 60.48  | 31.90  | 87.06%  |
| **AdaV**       | 58.53  | 68.91  | 55.11  | 85.25  | 1452.91  | 65.20  | 36.20  | 94.35%  |

## 📖 Citation
If you use our work, please cite:
```bibtex
@inproceedings{han-etal-2025-adav,
  title={AadV: Adaptive Attention with Vision Tokens for Efficient Vision-Language Models},
  author={Jiayi Han, Liang Du, Yiwen Wu, Guanming Liang, Xiangguo Zhou, Weibo Zheng, Donghong Han, and Zixun Sun},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={4985–4997},
  year={2025}
}
```
