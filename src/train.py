from collections import defaultdict
from contextlib import nullcontext
import logging
from typing import Callable

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def train_classifier(
        model: Module,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        optimizer_config: dict,
        loss_fn: Callable,
        device: str,
) -> tuple[Module, dict[str, defaultdict[list]]]:
    """
    Train a text classifier model.

    Parameters:
        model: The model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        num_epochs: Number of epochs to train for.
        optimizer_config: Configuration dict for the optimizer with learning rate and other parameters.
        loss_fn: Loss function to use during training.
        device: Device to run the training on (CPU or GPU).

    Returns:
        The trained model.
        A dictionary containing training and validation loss and accuracy per epoch.
    """
    device = torch.device(device)

    # Initialize the optimizer with the model parameters that require gradients.
    optimizer = optimizer_config["class"](
        filter(lambda p: p.requires_grad, model.parameters()),
    )

    # Training loop
    metrics = {"training": defaultdict(list), "validation": defaultdict(list)}
    for epoch in range(num_epochs):
        # Run the training and validation epochs.
        train_loss, train_acc = run_epoch(model, loss_fn, device, data_loader=train_loader, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, loss_fn, device, data_loader=val_loader, optimizer=None)
        # Keep track of metrics.
        metrics["training"]["loss"].append(train_loss)
        metrics["training"]["accuracy"].append(train_acc)
        metrics["validation"]["loss"].append(val_loss)
        metrics["validation"]["accuracy"].append(val_acc)
        # Print statistics.
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}: \n" 
            f"Train Loss: {train_loss: .4f}, Validation Loss: {val_loss: .4f} \n"
            f"Train Accuracy: {train_acc: .4f}, Validation Accuracy: {val_acc: .4f} \n"
            "=" * 70
        )

    return model, metrics


def run_epoch(
        model: Module,
        loss_fn: Callable,
        device: torch.device,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """
    Run a single epoch of training or validation.
    Parameters:
        model: The model to train or validate.
        loss_fn: Loss function to compute the loss.
        device: Device to run the computations on (CPU or GPU).
        data_loader: DataLoader for the dataset.
        optimizer: Optimizer for updating model parameters (None for validation).
    Returns:
        The average loss for the epoch.
    """
    # Set functionality depending on train or validation step.
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    grad_context = nullcontext() if is_training else torch.no_grad()

    # Initialize epoch loss and accuracy.
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch in data_loader:
        # Move data to device.
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Zero the gradients if training.
        if is_training:
            optimizer.zero_grad()

        # Forward pass with gradient context (skips grads during validation).
        with grad_context:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        # Backward pass and optimization step if training.
        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        # Compute accuracy.
        preds = torch.argmax(logits, dim=-1)
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    # Compute epoch loss and accuracy.
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy


