# Interview Coach - AI-Powered Mock Interview System

An LLM-based application that generates contextual interview questions based on uploaded resumes. Built with LoRA fine-tuning and RAG pipeline.

##  Architecture
```
PDF Upload → Resume Parser → Chunker → Embedder → Retriever → Fine-tuned LLM → Questions
```

## 📁 Project Structure
```
interview-coach/
├── rag/
│   ├── resume_parser.py    # PDF/text parsing
│   ├── chunker.py          # Semantic chunking
│   ├── embedder.py         # Vector embeddings (ChromaDB)
│   ├── retriever.py        # RAG retrieval
│   └── generator.py        # LLM question generation
├── data/
│   └── sample_resumes/     # Test resumes
├── tests/
│   └── test_parser.py
├── export_prompts.py       # Export prompts for Colab inference
├── requirements.txt
└── README.md
```

##  Quick Start

### Installation
```bash
conda create -n interview-coach python=3.11
conda activate interview-coach
pip install -r requirements.txt
```

### Run RAG Pipeline
```bash
# Test parser
python -m rag.resume_parser

# Test full pipeline
python -m rag.retriever

# Export prompts for Colab
python export_prompts.py
```

### Model Inference (Colab)

Model hosted on HuggingFace: [shubhampareek/interview-coach-lora](https://huggingface.co/shubhampareek/interview-coach-lora)

Use `Interview_Coach_Inference.ipynb` for GPU-accelerated inference.

##  Tech Stack

- **Fine-tuning:** LoRA on Mistral-7B
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB
- **Framework:** PyTorch, Transformers, PEFT


##  License

MIT