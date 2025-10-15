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

## 🚀 Getting Started

Run the project using configuration-driven scripts:

```bash
python train.py -c config.yml      # Train classification/generative model
python evaluate.py -c config.yml   # Evaluate trained model
python generate.py -c config.yml   # Generate new reviews
```

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

## 📦 Project Structure

```
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── generate.py       # Text generation script
├── data/             # Preprocessing and augmentation
├── configurations/   # Example config files
├── notebooks/        # Exploratory notebooks
```

---

## 📈 Results Summary

- **Classification**: Feed-forward networks + TF-IDF offer strong baselines
- **Generation**: Transformer-based models show promise but require pre-training
- **Augmentation**: Addressed imbalance, significantly boosting performance

---

## 📄 Technical Report

For detailed methodology, architecture diagrams, results, and limitations, please refer to our full [Technical Report (PDF)](./TechnicalReport.pdf).

---

## 👥 Authors

- Lucas Duport  
- Eugenie Beauvillain  
- Yanis Martin  
- Arthur Courselle  
- Flavien Geoffray