# arXiv Scientific Domain Classification
A pet-project on article abstracts classification with NLP

This project explores large-scale text classification of scientific abstracts from arXiv into major research domains using both classical machine learning and modern transformer-based approaches.

The goal is to compare interpretability, performance, and robustness across methods, with a particular focus on class imbalance and interdisciplinary overlap.

---

## 📌 Problem Statement

Given the abstract of a scientific paper, predict its primary research domain among:

- cs (Computer Science)
- math (Mathematics)
- physics
- hep (High Energy Physics)
- nucl (Nuclear Physics)
- econ (Economics)
- q-bio (Quantitative Biology)
- q-fin (Quantitative Finance)

The dataset is highly imbalanced and contains substantial semantic overlap between domains, making accuracy alone an insufficient evaluation metric.

---

## 📚 Dataset

- Source: arXiv abstracts via Hugging Face datasets
- Size: ~1.5M abstracts
- Text field: paper abstract
- Labels: top-level arXiv subject categories
- Duplicate rate: <0.01%

Preprocessing steps:
- mapped fine-grained subjects to top-level domains to create adequate classification labels
- removed duplicates
- stratified train / validation / test splits

---

## 🧪 Approaches

### 1️⃣ TF-IDF + Logistic Regression (Baseline)

- TF-IDF vectorization (unigrams + bigrams)
- Logistic regression with L2 regularization
- Strong baseline accuracy but limited performance on minority classes

### 2️⃣ Transformer Fine-Tuning (DistilBERT)

- Model: `distilbert-base-uncased`
- Full fine-tuning on labeled abstracts
- Evaluated using accuracy and macro-F1
- Error analysis via normalized confusion matrices

---

## 📊 Results

| Model | Accuracy | Macro-F1 |
|------|----------|----------|
| TF-IDF + Logistic Regression | ~0.89 | ~0.70 |
| DistilBERT (fine-tuned) | ~0.93 | ~0.83 |

Key observations:
- Transformer significantly improves recall for underrepresented classes
- Most errors occur between semantically adjacent domains (e.g., econ ↔ math, physics ↔ hep)
- Confusions reflect genuine interdisciplinary overlap rather than random errors

---

## 🔍 Error Analysis

Normalized confusion matrices reveal structured misclassifications:
- econ papers often predicted as math or cs
- q-bio overlaps with cs and physics
- nuclear physics and high-energy physics are frequently confused

These patterns align with domain knowledge and motivate future extensions such as multi-label or hierarchical classification.

---

## 🛠️ Hardware Notes

- Baseline models trained on CPU
- Transformer fine-tuning performed on GPU (NVIDIA H200)
- Hardware significantly affected optimization behavior for linear models

---

## 📁 Repository Structure

notebooks/ # main analysis notebook
models/ # saved fine-tuned model and tokenizer
results/ # metrics and visualizations

---

## 🧠 Takeaways

This project demonstrates:
- principled baseline construction
- careful evaluation beyond accuracy
- transformer fine-tuning at scale
- domain-informed error interpretation

---

## 📎 Requirements

Key libraries:
- transformers
- datasets
- scikit-learn
- torch
- numpy
- matplotlib / seaborn

See `requirements.txt` for details.



