# ML Module - LoRA Fine-tuning

This module handles fine-tuning models for resume parsing and interview question generation.

## Directory Structure

```
ml/
├── data/
│   ├── resumes/          # Raw resume datasets
│   ├── interview_qa/     # Interview Q&A datasets
│   └── processed/        # Processed training data
├── models/
│   ├── checkpoints/      # Training checkpoints
│   └── exported/         # Final exported models
├── training/
│   ├── train.py          # Main training script
│   ├── dataset.py        # Dataset processing
│   └── lora_config.py    # LoRA configuration
├── evaluation/
│   ├── evaluate.py       # Model evaluation
│   └── benchmark.py      # API vs Local comparison
└── configs/
    └── training_config.yaml
```

## Datasets to Explore

### Resume Datasets
- [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- [Resume Corpus](https://github.com/florex/resume_corpus)

### Interview Q&A Datasets
- [Interview Questions Dataset](https://www.kaggle.com/datasets/veer006/interview-question-answer)
- Custom collected behavioral questions

## TODO

- [ ] Collect and preprocess resume dataset
- [ ] Collect interview Q&A dataset
- [ ] Define LoRA configuration
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Export model for inference
