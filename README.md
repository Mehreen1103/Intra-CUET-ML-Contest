# Bengali Physics MCQ Solving with CoT-Driven Quantized LLM

**Competition:** [Intra-CUET Machine Learning Contest](https://www.kaggle.com/competitions/ml-contest-cuet/data)  
**Team:** Mehreen Rahman, Walisa Alam  
**Department:** CSE, Chittagong University of Engineering and Technology

---

## Overview

A Bengali physics MCQ answering system using **Chain-of-Thought (CoT) prompting** with
**4-bit quantized LLMs**. The system generates step-by-step reasoning before producing a
final answer, combining efficient inference with expert-style Bangla prompts.

---

## Repository Structure
```
├── README.md
├── enthusiasts-zeroshot-cot-using-qwen-14b-base.ipynb   # Main notebook (best model)
└── outputs/
    └── submission.csv
```

---

## Models Explored

| Model | Type | Prompt | Public Score | Private Score |
|-------|------|--------|-------------|---------------|
| **Qwen3-14B Base** ✅ | Zero-shot CoT | English | 0.6666 | **0.7666** |
| Qwen3-14B Conversational | Zero-shot CoT | English | 0.6600 | 0.6666 |
| Qwen3-14B Conversational | Zero-shot CoT | Bengali | — | 0.5260 |
| Phi-4 (14B) | Fine-tuned | — | 0.6000 | 0.5400 |
| Mistral-7B | Zero-shot | — | — | — |

> ✅ Best model: `unsloth/Qwen3-14B-Base-unsloth-bnb-4bit`  
> Also explored: `unsloth/mistral-7b-bnb-4bit`

---

## Methodology

### Pipeline
1. **Data Preprocessing** — custom `parse_options()` to normalize inconsistently formatted MCQ options using regex
2. **CoT Prompting** — English expert-style prompt elicits step-by-step physics reasoning
3. **Inference** — 4-bit quantized model via `BitsAndBytesConfig` for memory-efficient GPU inference
4. **Answer Extraction** — regex parses final answer from generated reasoning

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Max New Tokens | 180 |
| Temperature | 0.1 |
| Decoding | Greedy (sampling disabled) |
| Pad Token ID | `tokenizer.eos_token_id` |

---

## Key Findings

- **English CoT > Bengali CoT** — switching prompt language improved private score from `0.526` → `0.666` on the conversational model
- **Base > Conversational** — the base model with CoT outperformed the instruction-tuned variant
- **Scale > Fine-tuning** — larger zero-shot models consistently beat smaller fine-tuned ones (Phi-4 14B fine-tuned < Qwen3-14B zero-shot)

---

## How to Run

The notebook is designed for **Kaggle** with GPU (P100/T4) enabled.

1. Go to the [competition data page](https://www.kaggle.com/competitions/ml-contest-cuet/data) and add the dataset
2. Open `enthusiasts-zeroshot-cot-using-qwen-14b-base.ipynb` on Kaggle
3. Enable GPU accelerator in Settings
4. Click **Run All**

---

## Authors

**Mehreen Rahman** — u2004033@student.cuet.ac.bd  
**Walisa Alam** — u2004015@student.cuet.ac.bd  
Department of Computer Science and Engineering, CUET
