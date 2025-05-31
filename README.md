# satellite-coregistration

## 概要
SARと高額画像のこレジストレーション（位置合わせ）を行うプロジェクトです。

## ディレクトリ構造

```
.
├── CLAUDE.md
├── README.md
├── env
│   ├── Dockerfile
│   ├── README.md
│   ├── docker-compose.yml
│   └── requirements.txt
├── project
│   └── spacenet9
│       ├── 00_data
│       │   ├── README.md
│       │   ├── dataset
│       │   │   ├── opt_keypoint_detection
│       │   │   └── sar_keypoint_detection
│       │   ├── interim
│       │   │   ├── resample_verification
│       │   │   ├── resampled
│       │   │   ├── tiepoint_converted
│       │   │   └── tiepoint_resampled
│       │   ├── plan
│       │   ├── plan_completed
│       │   │   └── preprocess.md
│       │   └── raw
│       │       ├── publictest
│       │       └── train
│       ├── 01_eda
│       │   └── spacenet9_eda.ipynb
│       ├── configs
│       │   ├── data
│       │   ├── model
│       │   ├── preprocessing_config.yaml
│       │   └── training
│       ├── doc
│       │   └── spacenet9_paper.md
│       └── src
│           ├── preprocess
│           │   ├── 00_pre_alignment.py
│           │   ├── 01_pre_resample.py
│           │   ├── 02_verify_resampling.py
│           │   └── 03_generate_tiles.py
│           └── test
│               ├── __pycache__
│               └── test_preprocess_integration.py
└── pyproject.toml

29 directories, 18 files
```