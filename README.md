# TripAdvisor Review Analysis

This repository contains our NLP project for analyzing TripAdvisor hotel reviews. We tackle two core tasks: predicting review ratings and generating hotel review text, using both traditional and deep learning methods.

---

## 🔍 Project Overview

- **Tasks**:
  - Predict review ratings from text (1–5 stars)
  - Generate synthetic reviews conditioned on ratings and keywords
- **Techniques**:
  - Supervised learning (classification)
  - Conditional text generation
- **Dataset**: 200k+ English TripAdvisor hotel reviews
- **Goal**: Explore both discriminative and generative NLP pipelines

---
## Installation

```
uv venv python=3.10
source .venv/bin/activate
uv pip install -r pyproject.toml
```

## 🚀 Getting Started

Run the project using the repository CLI located at `src/main.py`. The CLI is configuration-driven and exposes three main commands: `train`, `eval` and `infer`.

Examples (run from repository root):

```bash
uv run src/main.py --config configurations/FeedForward/config.yaml train --help
# Show options for the train command

# Train a model using a specific configuration and seed (recommended runner)
uv run src/main.py --config configurations/FeedForward/config.yaml --seed 42 train

# Evaluate a trained model
uv run src/main.py --config configurations/FeedForward/config.yaml eval

# Run inference / generate text from a prompt (provide prompt and rating)
uv run src/main.py --config configurations/Transformer/config.yaml --seed 42 infer -p "clean, location, nice, pretty, comfortable" -r 5

```

Notes:
- The `--config` option is required and should point to one of the YAML files in the `configurations/` folder (for example `configurations/Transformer/config.yaml`).
- `--seed` sets the random seed for reproducibility (default: 42).
- Use `uv run src/main.py --config <config> <command> --help` or `python -m src.main --config <config> <command> --help` to see command-specific options.

Example configuration files are available in the `configurations` directory. All training, preprocessing, and model details are specified via YAML.

---

## 📚 Dataset

- **Name**: TripAdvisor Review Rating Dataset
- **Size**: 201,295 hotel reviews
- **Labels**: Integer ratings (1–5 stars)
- **Fields Used**: `review` (text), `overall` (label)
- **Source**: [Hugging Face](https://huggingface.co/datasets/jniimi/tripadvisor-review-rating)
- **Language**: English (filtered using fastText)

---

## 🛠️ Data Processing

### Preprocessing Steps

- Lowercasing  
- Punctuation removal  
- Byte-Pair Encoding (BPE) tokenization  
- Optional: Lemmatization, stopword removal (in notebooks)

### Imbalance Handling

The dataset is heavily skewed towards high ratings. We addressed this via:

- **Class weighting**
- **Macro-averaged metrics**
- **Data augmentation** (see below)

---

## 🔁 Data Augmentation

Implemented via a modular class using `nlpaug`, targeting minority class expansion:

- **Synonym replacement**
- **Sentence reordering**
- **Word deletion**

Augmentation improved classifier performance significantly, e.g., FNN accuracy increased from **0.65 → 0.80**.

---

## 🧠 Classification Models

| Model                        | Accuracy | F1 Score |
|-----------------------------|----------|----------|
| Naive Bayes (TF-IDF)        | 0.59     | 0.57     |
| Logistic Regression (BPE+TF-IDF) | 0.66 | 0.65     |
| Feed-Forward NN (TF-IDF)    | 0.80     | 0.80     |
| RNN (Word2Vec)              | 0.74     | 0.73     |
| BERT Mini (fine-tuned)      | 0.77     | 0.76     |

**Insight**: A simple feed-forward model with TF-IDF outperformed more complex RNNs and even a small Transformer.

---

## ✍️ Generative Models

### Input Format
- Rating (1–5) + Keywords → Full review text

### Models

| Model        | BLEU   | ROUGE-L | BERTScore F1 |
|--------------|--------|----------|----------------|
| N-gram       | —      | —        | —              |
| FNN          | 0.015  | 0.194    | -0.268         |
| RNN          | 0.057  | 0.296    | -0.024         |
| Transformer  | 0.028  | 0.177    | 0.073          |

**Limitations**: Training from scratch limited performance. Pre-trained models were not fine-tuned due to hardware constraints.

---

## 🧱 Architecture

Modular, configuration-driven system:

- Plug-and-play support for tokenizers, vectorizers, and models
- Easily switch between CPU, CUDA, and Apple MPS
- YAML-configurable training pipeline

---

## 📊 Vectorization Methods

- **TF–IDF**: Best performance for classification
- **Word2Vec**: Used in RNNs; inferior to learned embeddings in generation tasks

---

## 📦 Project structure

Repository layout (top-level):

```
pyproject.toml
README.md
configurations/        # Example YAML configs per pipeline/model
docs/                  # Documentation and figures
notebooks/             # Exploratory notebooks
src/                   # Source code (entrypoint: src/main.py)
  ├── main.py
  ├── data/             # preprocessing, augmentation, datasets
  ├── models/           # model implementations and modules
  ├── my_tokenizers/    # BPE / custom tokenizers
  ├── my_vectorizers/   # TF-IDF, word2vec helpers
  └── pipelines/        # training, evaluation, generation pipelines
```

Use `src/main.py` as the CLI entrypoint (via `uv run` or `python -m src.main`).

---

## 📈 Results Summary

- **Classification**: Feed-forward networks + TF-IDF offer strong baselines
- **Generation**: Transformer-based models show promise but require pre-training
- **Augmentation**: Addressed imbalance, significantly boosting performance

---

## 📄 Technical Report

For detailed methodology, architecture diagrams, results, and limitations, please refer to our full [Technical Report (PDF)](./docs/TechnicalReport.pdf).

---

## 👥 Authors

- Lucas Duport <lucas.duport@epita.fr>
- Eugenie Beauvillain <eugenie.beauvillain@epita.fr>
- Yanis Martin <yanis.martin@epita.fr>
- Arthur Courselle <arthur.courselle@epita.fr>
- Flavien Geoffray <flavien.geoffray@epita.fr>