# 🧠 ML Module - LoRA Fine-tuning

This module handles fine-tuning language models for resume parsing and interview question generation.

## 📁 Directory Structure

```
ml/
├── data/
│   ├── resumes/              # Raw resume datasets
│   ├── interview_qa/         # Interview Q&A datasets
│   └── processed/            # Processed training data (JSONL format)
│
├── models/
│   ├── checkpoints/          # Training checkpoints
│   └── exported/             # Final models ready for inference
│
├── training/
│   ├── train.py              # Main training script
│   ├── dataset.py            # Dataset loading & preprocessing
│   └── lora_config.py        # LoRA hyperparameters
│
├── evaluation/
│   ├── evaluate.py           # Model evaluation metrics
│   └── benchmark.py          # Compare API vs Local model
│
└── configs/
    └── training_config.yaml  # Training configuration
```

## 🎯 Goals

We want to fine-tune a model that excels at:

1. **Resume Understanding**
   - Extract structured info from raw resume text
   - Identify key skills, experiences, achievements

2. **Interview Question Generation**
   - Generate relevant questions based on resume content
   - Create follow-up questions when answers are vague
   - Adapt tone for HR vs Technical interviews

## 🚀 Getting Started

### 1. Setup Environment

```bash
# From project root
cd resume-griller

# Install ML dependencies using uv
uv pip install -e ".[ml]"
```

### 2. Required Dependencies

The `[ml]` extras include:
- `torch` - PyTorch
- `transformers` - Hugging Face Transformers
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `datasets` - Dataset handling
- `accelerate` - Training acceleration

## 📊 Dataset Requirements

### Resume Dataset Format

Create JSONL files in `data/processed/` with this structure:

```jsonl
{"input": "Parse this resume:\n[raw resume text]", "output": "{\"name\": \"John Doe\", \"skills\": [...], ...}"}
{"input": "Parse this resume:\n[raw resume text]", "output": "{\"name\": \"Jane Smith\", \"skills\": [...], ...}"}
```

### Interview Q&A Dataset Format

```jsonl
{"input": "Resume: [resume summary]\nGenerate an HR interview question.", "output": "Tell me about a time when you had to lead a team through a difficult situation at [Company]."}
{"input": "Resume: [resume summary]\nPrevious answer: [vague answer]\nGenerate a follow-up question.", "output": "You mentioned 'improving performance'. Can you quantify that? What metrics did you use?"}
```

## 🔧 Suggested Base Models

| Model | Size | Pros | Cons |
|-------|------|------|------|
| **Llama 3 8B** | 8B | Good balance of quality/speed | Requires decent GPU |
| **Mistral 7B** | 7B | Fast, good instruction following | Slightly less capable |
| **Qwen 2 7B** | 7B | Good multilingual support | Less community resources |
| **Phi-3 Mini** | 3.8B | Very fast, runs on consumer GPU | Less capable for complex tasks |

**Recommendation**: Start with **Mistral 7B** or **Llama 3 8B** for best results.

## 📝 LoRA Configuration

Suggested starting point for `training/lora_config.py`:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Alpha scaling
    target_modules=[           # Modules to apply LoRA
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## 🏋️ Training Pipeline

### Basic Training Script Structure

```python
# training/train.py

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

def train():
    # 1. Load base model
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # 2. Apply LoRA
    lora_config = LoraConfig(...)
    model = get_peft_model(model, lora_config)
    
    # 3. Load dataset
    dataset = load_dataset("json", data_files="data/processed/train.jsonl")
    
    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        save_steps=100,
        logging_steps=10,
    )
    
    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    
    # 6. Save
    model.save_pretrained("models/exported/resume-griller-lora")
```

## 📈 Evaluation Metrics

### For Resume Parsing
- **Field Extraction Accuracy**: % of correctly extracted fields
- **JSON Validity**: % of outputs that are valid JSON

### For Question Generation
- **Relevance Score**: Does the question relate to the resume?
- **Specificity Score**: Does it reference specific resume details?
- **Follow-up Quality**: Does it address vagueness in previous answer?

### API vs Local Comparison

```python
# evaluation/benchmark.py

def compare_models(test_cases):
    """Compare API model vs fine-tuned local model."""
    results = {
        "api": {"latency": [], "quality": []},
        "local": {"latency": [], "quality": []},
    }
    
    for case in test_cases:
        # Test API
        api_result = call_api_model(case)
        results["api"]["latency"].append(api_result.latency)
        results["api"]["quality"].append(evaluate_quality(api_result))
        
        # Test Local
        local_result = call_local_model(case)
        results["local"]["latency"].append(local_result.latency)
        results["local"]["quality"].append(evaluate_quality(local_result))
    
    return results
```

## 🖥️ Hardware Requirements

### Minimum (for inference)
- GPU: 8GB VRAM (RTX 3070 or similar)
- RAM: 16GB
- Storage: 50GB

### Recommended (for training)
- GPU: 24GB VRAM (RTX 4090, A10, etc.)
- RAM: 32GB
- Storage: 100GB

### Cloud Options
- [RunPod](https://runpod.io/) - A100 GPUs on demand
- [Lambda Labs](https://lambdalabs.com/) - Good pricing
- [Google Colab Pro](https://colab.research.google.com/) - For smaller experiments

## 📚 Resources

### Papers
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)

### Tutorials
- [Hugging Face PEFT Guide](https://huggingface.co/docs/peft)
- [Fine-tuning LLMs with LoRA](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)

### Datasets
- [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- [Interview Questions Dataset](https://www.kaggle.com/datasets/veer006/interview-question-answer)

## ✅ TODO Checklist

- [ ] Collect and clean resume dataset
- [ ] Collect interview Q&A dataset
- [ ] Create data preprocessing pipeline
- [ ] Setup training environment
- [ ] Implement training script
- [ ] Run initial training experiments
- [ ] Evaluate model quality
- [ ] Benchmark against API
- [ ] Export model for backend integration
- [ ] Document inference API

## 🔗 Integration with Backend

Once your model is trained and exported, update the backend config:

```bash
# .env
LLM_MODE=local
LOCAL_MODEL_PATH=./ml/models/exported/resume-griller-lora
```

The `backend/app/services/llm_service.py` will automatically load your model.

---

Questions? Open an issue or reach out to the team!
