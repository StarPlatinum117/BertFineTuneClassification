import torch
from torch.nn import Module
from transformers import BertForSequenceClassification


def load_model(num_labels: int, device: str = "cpu") -> Module:
    device = torch.device(device)
    # Load vanilla model (google-bert/bert-base-uncased is too big).
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    # Freeze encoder parameters.x
    for name, param in model.bert.named_parameters():
        param.requires_grad = False
    model.to(device)
    return model
