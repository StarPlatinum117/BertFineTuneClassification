import logging
import datasets
import pathlib
import time
import torch
from datasets import DatasetDict
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


class AGNewsDataset(TorchDataset):
    def __init__(self, split_name: str, dataset: HFDataset, tokenizer: PreTrainedTokenizerBase, max_length: int = 128):
        cache_path = DATA_DIR / f"tokenized_{split_name}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the dataset if cache exists.
        if cache_path.exists():
            logging.info(f"Loading cached tokenized dataset from {cache_path}")
            cached_data = torch.load(cache_path)
            input_ids = cached_data["input_ids"]
            attention_mask = cached_data["attention_mask"]
            labels = cached_data["labels"]

        # Tokenize and save dataset if cache does not exist.
        else:
            logging.info(f"Tokenizing dataset {split_name}...")
            start_time = time.time()
            input_ids, attention_mask, labels = self.tokenize_dataset(dataset, tokenizer, max_length)
            end_time = time.time()
            duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            logging.info(f"Tokenization of {split_name} dataset completed in {duration}.")

            torch.save({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }, cache_path)
            logging.info(f"File saved to {cache_path}")

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    @staticmethod
    def tokenize_dataset(
            dataset: HFDataset,
            tokenizer: PreTrainedTokenizerBase,
            max_length: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes a dataset using the provided tokenizer.
        This function processes the dataset to convert text into input IDs and attention masks.

        Parameters:
            dataset: The AG News train/val/test dataset to tokenize.
            tokenizer: The tokenizer to use for encoding the text.
            max_length: The maximum length of the input sequences.
        Returns:
            A tuple containing input IDs, attention masks and labels.
        """
        tokenized_dataset = dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            ),
            batched=True,
            remove_columns=["text"],
        )
        input_ids = torch.tensor(tokenized_dataset["input_ids"])
        attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
        labels = torch.tensor(tokenized_dataset["label"])
        return input_ids, attention_mask, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def get_dataloaders(
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 16,
        max_len: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and splits the AG News dataset into training, validation, and test sets.

    Parameters:
        tokenizer: The tokenizer to use for encoding the text.
        batch_size: The batch size for the DataLoader.
        max_len: The maximum length of the input sequences.
    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
    """
    # Load the AG News dataset.
    dataset = datasets.load_dataset("ag_news")
    raw_train = dataset["train"]
    test_dataset = dataset["test"]
    # Split the training set into training and validation sets. Prepare train, val, test.
    train_val_dataset = raw_train.train_test_split(test_size=0.1, stratify_by_column="label")
    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]
    # Create the datasets.
    train_dataset = AGNewsDataset("train", train_dataset, tokenizer, max_length=max_len)
    val_dataset = AGNewsDataset("validation", val_dataset, tokenizer, max_length=max_len)
    test_dataset = AGNewsDataset("test", test_dataset, tokenizer, max_length=max_len)
    # Create the DataLoaders.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

