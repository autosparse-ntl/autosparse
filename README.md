# AutoSparse

Anonymous code release accompanying a manuscript under review.

This repository contains the implementation of AutoSparse together with the parameter-efficient TargetFT attack baselines used in the experimental evaluation.

The study investigates whether Non-Transferable Learning defenses remain effective when an attacker applies modern fine-tuning methods under a limited trainable-parameter budget.

The datasets and protected model checkpoints are not included in this repository; the datasets are publicly available from their original sources, and the checkpoints can be reproduced using the corresponding methods and experimental settings provided in NTLBench.

## Included methods

- AutoSparse
- LoRA
- DoRA
- QLoRA
- BitFit
- Sparse fine-tuning
- Full fine-tuning

## Repository structure

```text
attacks/
├── autosparse.py
├── lora.py
├── dora.py
├── qlora.py
├── bitfit.py
├── sparse_ft.py
└── full_finetuning.py

requirements.txt
README.md
```

## Usage

To view the available AutoSparse options:

```bash
python attacks/autosparse.py --help
```

The remaining attack implementations are available in the corresponding files under the `attacks/` directory.

## Anonymous review

This repository has been prepared for anonymous peer review.
