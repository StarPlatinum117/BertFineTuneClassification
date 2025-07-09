import torch
from torch.nn import Module
from transformers import DistilBertForSequenceClassification


def load_model(model_name: str, num_labels: int, device: str = "cpu") -> Module:
    device = torch.device(device)
    # Load vanilla model (google-bert/bert-base-uncased is too big).
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Freeze encoder parameters.x
    for name, param in model.distilbert.named_parameters():
        param.requires_grad = False
    model.to(device)
    return model
