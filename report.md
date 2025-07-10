# BERT Fine-Tuning for AG News Classification

This project fine-tunes a pretrained transformer (BERT/DistilBERT) on the AG News dataset for text classification. It is part of a broader self-study plan focused on mastering deep learning engineering practices and model deployment.

---

## 📌 Objective

Train a robust text classification model on the AG News dataset using transfer learning with a pretrained BERT model. Evaluate its performance, monitor training metrics, and improve training efficiency on a CPU-based system.

---

## 🧠 Model Overview

- **Backbone**: `distilbert-base-uncased` (switched from `bert-base-uncased` to reduce training time on CPU)
- **Architecture**: Pretrained encoder with classification head
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW with weight decay
- **Evaluation Metric**: Accuracy

---

## 📊 Results

| Metric          | Value     |
|-----------------|-----------|
| Test Accuracy   | ~83.6%    |
| Test Loss       | ~0.8484   |
| Best Val Acc    | ~68.6%    |
| Best Val Loss   | ~1.28     |

---

## 🔁 Training Setup

- **Epochs**: 5
- **Batch size**: 16
- **Token length**: 128
- **Hardware**: CPU-only
- **Training Time**: ~7 minutes per epoch on CPU

---

## 🧪 Evaluation

Evaluation was done on a held-out test set using the final saved model. Accuracy and average loss were computed using `evaluate_classifier()` in `evaluation.py`.

---

## 📈 Visualizations

Training and validation accuracy/loss per epoch were plotted to assess convergence and overfitting behavior.

Plots were generated from a `metrics` dictionary returned by the training loop.

---

## 🧼 Data Handling

- Used `datasets.load_dataset("ag_news")` from Hugging Face.
- Custom `AGNewsDataset` class handles tokenization and caching.
- Tokenized data is saved as `.pt` files and reused across runs to speed up training.

---

## ⚠️ Challenges and Lessons Learned

- **Tokenization performance**: Initially slow due to tokenizing inside `__getitem__`; resolved by preprocessing in bulk using `.map()` and caching.
- **Hardware constraints**: Training with `bert-base-uncased` on CPU was impractically slow. Switching to `distilbert-base-uncased` reduced runtime significantly.
- **Logging**: Careful logging helped identify training bottlenecks and debug formatting issues.
- **Model saving**: Only the best model based on validation loss is persisted to disk.

