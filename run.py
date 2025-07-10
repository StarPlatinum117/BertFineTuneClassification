import logging

import numpy as np
import pathlib
import torch.cuda
from torch.optim import SGD, AdamW
from transformers import DistilBertTokenizer

from model import load_model
from src.dataset import get_dataloaders
from src.evaluation import evaluate_classifier
from src.train import train_classifier
from src.visualization import plot_training_curves


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s'
)
# ============================ Settings. ===========================================
seed = 1
tokenizer_max_len = 128
n_epochs = 5
batch_size = 16
loss_fn = torch.nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilbert-base-uncased"
light_training = True  # faster training with smaller dataset
# ========================= Optimizer Config. ======================================
optimizer_config = {
    "adamw": {
        "class": AdamW,
        "params": {
            "lr": 2e-5,
            "weight_decay": 0.01,
            "eps": 1e-8
        }
    },
    "sgd": {
        "class": SGD,
        "params": {
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.01
        }
    }
}
chosen_optimizer = "adamw"
chosen_optimizer_config = optimizer_config[chosen_optimizer]
# ============================ Main. ===============================================
set_seed(seed)
# Load tokenizer and dataset.
logging.info("Starting the AG News dataset loading process...")
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
train_loader, val_loader, test_loader = get_dataloaders(
    tokenizer, batch_size=batch_size, max_len=tokenizer_max_len, light_training=light_training
)
logging.info("AG News dataset loaded successfully.")

# Load model.
logging.info(f"Loading {model_name} model")
model = load_model(model_name, num_labels=4, device=device)
logging.info("Model loaded successfully.")

# Commence training.
logging.info("Starting training process...")
model, metrics = train_classifier(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=n_epochs,
    optimizer_config=chosen_optimizer_config,
    loss_fn=loss_fn,
    device=device
)
logging.info("Training process completed successfully.")

# Visualize metrics.
output_path = pathlib.Path(__file__).parent / "training_curves.png"
plot_training_curves(metrics, save_path=output_path)
logging.info(f"Training curves saved to {output_path}")

# Evaluate the model on the test set.
logging.info("Starting evaluation on the test set...")
test_loss, test_acc = evaluate_classifier(model, test_loader, loss_fn, device)
