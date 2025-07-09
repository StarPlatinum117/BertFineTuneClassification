import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


class AGNewsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # Squeeze the batch dimension.
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(label)
        return input_ids, attention_mask, label


def get_dataloaders(
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 16,
        max_len: int = 128,
        seed: int = 1) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and splits the AG News dataset into training, validation, and test sets.

    Parameters:
        tokenizer: The tokenizer to use for encoding the text.
        batch_size: The batch size for the DataLoader.
        max_len: The maximum length of the input sequences.
        seed: Random seed for reproducibility.
    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
    """
    # Set the random seed for reproducibility.
    np.random.seed(seed)
    # Load the AG News dataset.
    dataset = datasets.load_dataset("ag_news")
    raw_train = dataset["train"]
    test_dataset = dataset["test"]
    # Split the training set into training and validation sets. Prepare train, val, test.
    train_val_dataset = raw_train.train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")
    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]
    # Create the datasets.
    train_dataset = AGNewsDataset(train_dataset, tokenizer)
    val_dataset = AGNewsDataset(val_dataset, tokenizer)
    test_dataset = AGNewsDataset(test_dataset, tokenizer)
    # Create the DataLoaders.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

