# 🧠 BERT Fine-Tuning for AG News Classification

This project fine-tunes a pre-trained BERT model on the [AG News dataset](https://huggingface.co/datasets/ag_news) to perform topic classification. It was built as part of a self-study plan to deepen understanding of transformer-based NLP pipelines, PyTorch workflows, and efficient dataset handling.

---

## 🚀 Overview

- Fine-tuned a pre-trained BERT (and DistilBERT) model for 4-class news topic classification.
- Implemented data preprocessing, model training, evaluation, and metric visualization from scratch.
- Optimized data loading by caching tokenized datasets and avoiding redundant computation.

---

## 🗂 Project Structure

BertFineTuneClassification/

├── data/ # CSV file and tokenized datasets

│ └── tokenized_*.pt

├── src/

│ ├── dataset.py # Dataset loading & tokenization (cached)

│ ├── evaluation.py # Test set evaluation logic

│ ├── train.py # Training loop with metric tracking

│ └── visualization.py # Metric plotting functions

├── training_plots/ # Output accuracy/loss plots

├── model.py # Model loading logic (DistilBERT)

├── run.py # Entry point: loads data, trains, evaluates

├── report.md  # Summary of work and lessons learned.

└── README.md

## ⚙️ Usage
Train the model:

python run.py

Tokenization:

Tokenized versions of train, val, and test sets are cached in the data/ directory.

If cached files exist, they will be loaded directly to skip tokenization.

Model Checkpoint:

The best model based on validation loss is saved in models/best_model.pt (not shown in repo structure until generated).

Plotting:

Accuracy and loss plots are saved in training_plots/.

Evaluation:

After training, the best model is evaluated on the test set.

Results (average loss and accuracy) are logged.
