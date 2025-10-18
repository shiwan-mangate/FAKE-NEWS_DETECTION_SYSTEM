# model_helper.py
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Fake_News(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base", return_dict=True)

        # Freeze base RoBERTa layers (optional)
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(cls_token)


@torch.no_grad()
def load_model(model_path="robert_fake_news_model.pth", device=None):
    """Safely load model trained with newer PyTorch versions"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = Fake_News()

    # ✅ Load weights to CPU first to avoid meta-tensor issue
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # ✅ Move to target device AFTER weights are fully loaded
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    return model, tokenizer, device


@torch.no_grad()
def predict(model, tokenizer, text, device, max_length=300):
    """Generate prediction and confidence"""
    encoding = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    output = model(input_ids, attention_mask).squeeze()
    confidence = float(output.item())

    prediction = "Fake News" if confidence > 0.5 else "Real News"
    return prediction, confidence
