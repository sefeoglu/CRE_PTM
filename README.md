# CRE_PTM

[![Paper](https://img.shields.io/badge/IEEEXplore-Access%20Paper-0078D4?style=for-the-badge&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/10.1109/ACCESS.2026.3682652)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-FFD21F?style=for-the-badge)](https://huggingface.co)
[![GitHub](https://img.shields.io/badge/GitHub-CRE__PTM-181717?style=for-the-badge&logo=github)](https://github.com/sefeoglu/CRE_PTM)

Large Language Models for Continual Relation Extraction.

This repository contains the source code and experimental pipeline for the paper "Large Language Models for Continual Relation Extraction". It includes data preparation, continual relation extraction training, metric computation, and result analysis for FewRel and TACRED using both encoder-decoder and decoder-only models.

![CRE](https://github.com/sefeoglu/CRE_PTM/blob/master/doc/cre.png)

> Trained models are publicly available on Hugging Face, as described in the journal article.

## Overview

The project studies continual relation extraction with large language models in task-incremental settings. It covers:

- FewRel and TACRED preprocessing and task splitting
- Prompt-based data construction for continual learning
- Training pipelines for Flan T5, Llama 2, and Mistral
- Evaluation metrics for average accuracy, whole accuracy, and backward transfer
- Result cleaning and visualization utilities

## Citation

```bibtex
@ARTICLE{efeoglu_2026,
  author={Efeoglu, Sefika and Paschke, Adrian and Schimmler, Sonja},
  journal={IEEE Access},
  title={Large Language Models for Continual Relation Extraction},
  year={2026},
  keywords={Semantic Web;Computer networks;Continual Relation Extraction;Schema-Level Errors;Large Language Models;Knowledge Graph Construction},
  doi={10.1109/ACCESS.2026.3682652}
}
```

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/sefeoglu/CRE_PTM.git
cd CRE_PTM
pip install -r requirements.txt
```

## Quick Start

The project reads configuration from `config.ini` and can be run with:

```bash
python main.py
```

Alternatively, follow the steps below to run the pipeline manually.

## Data Preparation

### TACRED

Convert raw TACRED samples into the required schema `(sentence, subject, object, object_type, subject_type)`:

```bash
python src/data_preparetation/data_preparation_tacred.py
```

Split the dataset according to the task setup from the paper:

```bash
python src/data_preparetation/instruction_ft_data_same_setting_tacred.py
```

### FewRel

Prepare the raw FewRel data:

```bash
python src/data_preparetation/data_preparation_fewrel.py
```

Generate task-level prompt data:

```bash
python src/data_preparetation/instruction_ft_data_same_setting_fewrel.py
```

## Training

### Decoder-only models

For Llama 2 and Mistral-style models:

```bash
python src/CRE/trainer_decoder.py
```

### Encoder-decoder model

For Flan T5:

```bash
python src/CRE/trainer_t5.py
```

## Evaluation and Post-processing

Clean decoder-only outputs that may contain explanations or extra text:

```bash
python src/clean/clean_decoder_results.py
```

Compute average and whole accuracy metrics:

```bash
python src/metrics/cl_metrics.py
```

Compute backward knowledge transfer:

```bash
python src/metrics/bwt.py
```

## Folder Structure

```text
.
├── LICENSE
├── README.md
├── config.ini
├── data/
│   ├── fewrel/
│   └── tacred/
├── doc/
├── logs/
├── main.py
├── requirements.txt
├── results/
├── src/
│   ├── CRE/
│   ├── analysis_viz/
│   ├── clean/
│   ├── data_preparetation/
│   ├── metrics/
│   ├── utils.py
│   └── zero_shot_prompting/
└── ...
```

## References

```bibtex
@inproceedings{cui-etal-2021-refining,
  title     = {{R}efining {S}ample {E}mbeddings with {R}elation {P}rototypes to {E}nhance {C}ontinual {R}elation {E}xtraction},
  author    = {Cui, Li and Yang, Deqing and Yu, Jiaxin and Hu, Chengwei and Cheng, Jiayang and Yi, Jingjie and Xiao, Yanghua},
  editor    = {Zong, Chengqing and Xia, Fei and Li, Wenjie and Navigli, Roberto},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  month     = {8},
  year      = {2021},
  address   = {Online},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2021.acl-long.20},
  doi       = {10.18653/v1/2021.acl-long.20},
  pages     = {232--243}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
