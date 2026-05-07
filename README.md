# NPPE-1: Multilingual Sentiment Analysis for Indian Languages

**Course:** Deep Learning for Practitioners (DLP) — IIT Madras BS in Data Science & Applications
**Term:** January 2026
**Student ID:** 21f2001203

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/nppe-dlp-2026-term-1)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/nppe-dlp-2026-term-1/data)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Gemma--3--1B--IT-yellow?logo=huggingface)](https://huggingface.co/google/gemma-3-1b-it)

| | |
|---|---|
| **Competition** | [NPPE1\_DLP\_2026\_Term1 on Kaggle](https://www.kaggle.com/competitions/nppe-dlp-2026-term-1) |
| **Dataset** | [Competition Data](https://www.kaggle.com/competitions/nppe-dlp-2026-term-1/data) |
| **Participants** | 209 teams, 1,080 submissions |
| **Evaluation** | Macro F1-Score |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Technical Architecture](#technical-architecture)
5. [Prompt Engineering](#prompt-engineering)
6. [Fine-tuning Strategy](#fine-tuning-strategy)
7. [Results](#results)
8. [Tech Stack](#tech-stack)
9. [Project Structure](#project-structure)
10. [How to Run](#how-to-run)
11. [Key Design Decisions](#key-design-decisions)

---

## Project Overview

This project solves a **binary sentiment classification** task on text written in **13 Indian languages**, as part of the IIT Madras Non-Proctored Programming Exam (NPPE-1) for the Deep Learning for Practitioners course. The approach combines **parameter-efficient fine-tuning (PEFT)** of a large language model with carefully crafted **prompt engineering** to achieve robust multilingual sentiment classification.

The solution fine-tunes **Google Gemma-3-1B-IT** using **QLoRA (Quantized Low-Rank Adaptation)** — a memory-efficient technique that trains only ~2.5% of the model's parameters while achieving strong performance across all 13 languages.

---

## Problem Statement

Given a sentence in one of 13 Indian languages, predict whether the sentiment is **Positive (1)** or **Negative (0)**.

**Evaluation Metric:** Macro F1-Score (equal weight to both classes, regardless of support)

**Why this is hard:**
- 13 diverse Indian languages with different scripts (Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Arabic/Urdu, Latin-romanized variants, etc.)
- Small dataset (only 900 training samples across 13 languages — ~63–76 samples per language)
- Cross-lingual generalization required
- Low-resource languages (some languages have fewer than 70 training examples)

---

## Dataset

| Split | Samples | Labels |
|-------|---------|--------|
| Train | 900 | 456 Positive, 444 Negative |
| Test  | 100 | Hidden (evaluated on Kaggle) |

**Languages (13 Indian languages):**

| Language | Script | Approx. Samples |
|----------|--------|-----------------|
| Assamese | Assamese (Bengali variant) | ~65 |
| Bodo | Devanagari | ~65 |
| Bengali | Bengali | ~70 |
| Gujarati | Gujarati | ~68 |
| Hindi | Devanagari | ~75 |
| Kannada | Kannada | ~70 |
| Malayalam | Malayalam | ~65 |
| Marathi | Devanagari | ~70 |
| Odia | Odia | ~65 |
| Punjabi | Gurmukhi | ~68 |
| Tamil | Tamil | ~65 |
| Telugu | Telugu | ~72 |
| Urdu | Arabic (Nastaliq) | ~63 |

**Key statistics:**
- Mean text length: 136.6 characters (range: 14–527)
- Class balance: ~50–54% Positive per language (well-balanced)
- Dataset files: `train.csv`, `test.csv`, `sample_submission.csv`

---

## Technical Architecture

### Base Model: Google Gemma-3-1B-IT

- **Parameters:** 1.02 billion
- **Type:** Instruction-tuned Causal Language Model
- **Vocabulary Size:** 262,144 tokens (excellent multilingual coverage)
- **Access:** Gated model via HuggingFace Hub (requires authentication)
- **Why Gemma-3-1B-IT:** Mandatory requirement for this competition; strong multilingual tokenization; instruction-tuned variant understands task descriptions and follows constrained output instructions natively

### Fine-tuning: QLoRA (Quantized LoRA)

QLoRA combines two techniques:
1. **4-bit NF4 Quantization** — Compresses the base model weights from 32-bit floats to 4-bit integers using NormalFloat4 format, reducing VRAM from ~4GB to ~2GB with minimal accuracy loss
2. **LoRA Adapters** — Adds small trainable rank-decomposition matrices alongside frozen base model weights, drastically reducing trainable parameter count

```
Total Parameters:    1,021,700,096  (1.02B)
Trainable (LoRA):       26,148,864  (26.1M)
Trainable %:                  2.54%
VRAM Used:              ~14.5 GB / 15.6 GB (Tesla T4)
```

**LoRA Configuration:**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Rank (r) | 32 | Higher rank for 13-language multilingual capacity |
| Alpha (α) | 64 | 2× rank — standard effective scaling |
| Dropout | 0.05 | Light regularization for small dataset |
| Target Modules | q, k, v, o projections + gate, up, down MLP projections | Full model coverage for best adaptation |

---

## Prompt Engineering

Prompt engineering is a core component of this project. Since Gemma-3-1B-IT is an **instruction-tuned** model, it responds to natural language task descriptions. The prompts are structured to:

1. Define the model's role and task
2. Provide language context
3. Optionally provide few-shot examples for low-resource languages
4. Enforce constrained single-word output

### Prompt Template Structure

The Gemma chat template (`<bos><start_of_turn>user ... <start_of_turn>model`) is applied via the tokenizer's built-in `apply_chat_template()` method.

**System/User Turn (prompt):**
```
You are a multilingual sentiment analysis expert specialising in Indian languages.
Classify each text as exactly one of: Positive, Negative.

Language: {language_name}
Text: {sentence}

Sentiment (respond with one word only — Positive or Negative):
```

**Assistant Turn (label, appended during training only):**
```
Positive
```
or
```
Negative
```

### Prompt Engineering Techniques Used

#### 1. Role Prompting
The system prompt establishes the model as a *"multilingual sentiment analysis expert specialising in Indian languages"*. This activates domain-relevant knowledge encoded in the instruction-tuned model, improving zero-shot and few-shot performance compared to a generic prompt.

#### 2. Language Context Injection
Each prompt explicitly names the source language in English:
```
Language: Hindi
Text: यह फिल्म बहुत अच्छी थी।
```
This simple addition helps the model:
- Attend to language-specific morphological and syntactic patterns
- Disambiguate scripts shared across languages (e.g., Devanagari is used for Hindi, Marathi, and Bodo)
- Leverage language-specific knowledge from pretraining

#### 3. Constrained Output Instruction
```
Sentiment (respond with one word only — Positive or Negative):
```
The phrase *"respond with one word only"* enforces single-token generation, which:
- Prevents verbose, hallucinated responses
- Makes output parsing trivial and deterministic
- Significantly reduces the number of generation tokens needed (max 10)

#### 4. Few-Shot Prompting (Inference-time, for sparse languages)
For languages with the lowest training sample counts, 2 manually selected examples (1 Positive, 1 Negative) from Bengali and Hindi are prepended to the prompt at inference time:
```
Language: Bengali
Text: এই ছবিটি অত্যন্ত সুন্দর।
Sentiment: Positive

Language: Hindi
Text: यह बहुत बुरा अनुभव था।
Sentiment: Negative

Language: {target_language}
Text: {target_sentence}
Sentiment (respond with one word only — Positive or Negative):
```
This leverages the model's **in-context learning** capability without any gradient updates, acting as a lightweight boost for low-resource languages.

#### 5. Completion-Only Loss Masking (Training)
During training, only the label token (`Positive` or `Negative`) contributes to the loss function — the prompt tokens are masked. This:
- Prevents the model from "wasting" gradient signal on learning to generate the prompt itself
- Focuses training exclusively on the classification decision
- Mirrors how instruction-tuned models are originally trained

#### 6. Greedy Decoding + Logits Fallback (Inference)
A two-stage inference strategy is used:
- **Stage 1:** Greedy decoding (deterministic, no sampling) with max 10 new tokens; parse the first word of the output
- **Stage 2 (fallback):** If the generated text doesn't contain "Positive" or "Negative" (11/100 test samples), extract the raw logits at the final input position and compare the log-probabilities of the "Positive" and "Negative" token IDs directly — this is robust to any generation failure

---

## Fine-tuning Strategy

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 6 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Learning rate | 1e-4 |
| LR schedule | Cosine with warmup |
| Warmup steps | 33 (~10% of total) |
| Total steps | 342 |
| Max sequence length | 256 tokens |
| Weight decay | 0.01 |
| Optimizer | paged_adamw_8bit |
| Gradient checkpointing | Enabled |
| Training samples | All 900 (no validation holdout for final run) |
| Hardware | Tesla T4 GPU (15.6 GB VRAM) |
| Training time | ~49 minutes |

### Why QLoRA over Full Fine-tuning?

| Approach | VRAM Required | Trainable Params | Training Time |
|----------|--------------|------------------|---------------|
| Full Fine-tune | ~16–32 GB | 1.02B | Very slow |
| LoRA (fp16) | ~8 GB | 26.1M | Moderate |
| **QLoRA (4-bit)** | **~2–4 GB** | **26.1M** | **Fast** |

QLoRA makes it feasible to fine-tune a 1B parameter model on a single 15.6 GB T4 GPU available on Kaggle's free tier.

---

## Results

| Metric | Value |
|--------|-------|
| Training Loss (final) | 1.6601 |
| Training Runtime | 2938 seconds (~49 min) |
| Test Positive predictions | 55 / 100 |
| Test Negative predictions | 45 / 100 |
| Expected Zero-shot Macro F1 | ~0.60–0.72 |
| Expected Fine-tuned Macro F1 | **~0.85–0.93** |

> Note: Actual Kaggle leaderboard F1 depends on the hidden test set labels.

---

## Tech Stack

### Core ML / NLP

| Library | Version | Purpose |
|---------|---------|---------|
| `transformers` | 5.2.0 | Model loading, tokenizer, generation |
| `peft` | 0.18.1 | LoRA adapter implementation |
| `trl` | ≥0.12 | SFTTrainer (Supervised Fine-Tuning) |
| `bitsandbytes` | ≥0.43 | 4-bit NF4 quantization backend |
| `accelerate` | 1.12.0 | Distributed training & mixed precision |
| `datasets` | 4.0.0 | HuggingFace datasets interface |

### Data Science

| Library | Purpose |
|---------|---------|
| `torch` 2.9.0+cu126 | PyTorch (CUDA 12.6) |
| `pandas` | DataFrame manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | F1-score, classification report |

### Infrastructure

| Component | Detail |
|-----------|--------|
| GPU | Tesla T4 (15.6 GB VRAM) |
| CUDA | 12.6 |
| Platform | Kaggle Notebooks |
| Model Access | HuggingFace Hub (gated, requires token) |
| Secret Management | Kaggle Secrets (HF_TOKEN) |

---

## Project Structure

```
NPPE 1/
├── 21f2001203.ipynb                        # Main submission notebook (full pipeline)
├── nppe1_multilingual_sentiment_21f2001203.ipynb  # Condensed version of the approach
├── notebook8149da5db3.ipynb                # Alternative/experimental notebook version
├── README.md                               # This file
├── nppe-dlp-2026-term-1/                   # Competition dataset
│   ├── train.csv                           # 900 labelled training samples
│   ├── test.csv                            # 100 unlabelled test samples
│   └── sample_submission.csv              # Submission format template
└── KAGGLE_GENERATED_OUTPUT/
    └── NPPE1_21f2001203.ipynb              # Kaggle-generated output version
```

### Key Notebook: `21f2001203.ipynb`

The main notebook follows this pipeline:

```
1. Environment Setup          → Install libraries, verify versions, set seeds
2. Data Loading & EDA         → Load CSVs, explore language distribution, class balance
3. Model Loading              → Load Gemma-3-1B-IT with 4-bit NF4 quantization
4. Prompt Engineering         → Define prompt templates, apply chat template
5. LoRA Configuration         → Set up PEFT LoRA adapters on attention + MLP layers
6. Training                   → SFTTrainer with 6 epochs, cosine LR schedule
7. Inference                  → Greedy decoding + logits fallback strategy
8. Submission                 → Format, validate, and export submission.csv
```

---

## How to Run

### Prerequisites

1. **Kaggle account** with GPU enabled (T4 or better)
2. **HuggingFace account** with access to `google/gemma-3-1b-it` (gated model — requires signing Google's terms on HuggingFace)
3. **Kaggle Secret:** Add your HuggingFace token as `HF_TOKEN` in Kaggle Secrets

### Steps

1. Upload the notebook to Kaggle
2. Add the competition dataset `nppe-dlp-2026-term-1` as a Kaggle dataset
3. Enable GPU accelerator (Tesla T4 × 1)
4. Set `HF_TOKEN` in Kaggle Secrets → Notebook secrets
5. Run all cells in order — total runtime ~50–60 minutes
6. The output `submission.csv` will be generated in `/kaggle/working/`

### Library Installation (first cell)

```bash
pip install -q \
  "transformers>=4.40.0" \
  "peft>=0.10.0" \
  "trl>=0.8.6" \
  "bitsandbytes>=0.43.0" \
  "accelerate>=0.27.0" \
  "datasets>=2.18.0"
```

---

## Key Design Decisions

### 1. Training on Full Dataset (No Validation Split)
For the final submission, all 900 training samples are used without a validation holdout. While this prevents monitoring overfitting, it maximizes the data available for learning 13 languages × binary classification, which is critical given the small per-language sample count (~65–76 samples).

### 2. LoRA Rank 32 (vs. Default 16)
Higher LoRA rank (32) provides more expressive capacity for the adapter. With 13 languages and distinct scripts, a rank-16 adapter may underfit the cross-lingual complexity. Rank 32 doubles the trainable parameters in each adapted layer at minimal compute cost.

### 3. 6 Epochs (vs. 3–4)
With only ~65 samples per language, the model needs more passes through the data to converge. 6 epochs with a cosine LR schedule and warmup prevents both underfitting (too few epochs) and overfitting (mitigated by weight decay + low LoRA rank).

### 4. paged_adamw_8bit Optimizer
Uses 8-bit Adam with paging to CPU when GPU memory is tight. Combined with gradient checkpointing, this prevents OOM errors on the 15.6 GB T4 even with effective batch size 16.

### 5. Logits Fallback for Robust Inference
When the model generates unexpected text (non-"Positive"/non-"Negative" outputs), the fallback directly compares logit scores for the label tokens at the final input token position. This handles 11/100 test samples and ensures 100% of test samples receive a valid prediction.

### 6. Double Quantization
Enabled in the 4-bit config — quantizes the quantization constants themselves, saving an additional ~0.4 bits per parameter (~50 MB on a 1B model). This headroom allows training with larger effective batch sizes.

---

## Concepts Demonstrated

This project demonstrates practical applications of the following deep learning and NLP concepts:

- **Large Language Models (LLMs):** Using a 1B parameter causal language model for classification
- **Parameter-Efficient Fine-Tuning (PEFT):** LoRA / QLoRA for low-resource adaptation
- **Quantization:** 4-bit NF4 (NormalFloat4) with double quantization for memory efficiency
- **Prompt Engineering:** Role prompting, language context injection, constrained output, few-shot learning, completion-only loss masking
- **Multilingual NLP:** Cross-lingual transfer, handling diverse Indian language scripts
- **Supervised Fine-Tuning (SFT):** Using TRL's SFTTrainer with causal language modeling objective
- **Instruction Tuning:** Leveraging instruction-tuned model checkpoints for better task alignment
- **In-Context Learning:** Few-shot examples at inference time for low-resource languages
- **Memory-Efficient Training:** Gradient checkpointing, 8-bit optimizer, 4-bit quantization on constrained GPU

---

## Academic Context

**Program:** IIT Madras BS in Data Science & Applications (Online Degree)
**Course:** Deep Learning for Practitioners (DLP)
**Assessment:** Non-Proctored Programming Exam 1 (NPPE-1)
**Term:** January 2026

This competition is part of the degree-level coursework, requiring students to apply state-of-the-art deep learning techniques (specifically fine-tuning `google/gemma-3-1b-it`) to a real-world multilingual NLP problem under resource constraints.

---

*Built with HuggingFace Transformers, PEFT, TRL, and PyTorch on Kaggle's free GPU tier.*
