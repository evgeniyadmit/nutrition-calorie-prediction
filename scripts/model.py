import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet18, ResNet18_Weights


class NutritionModel(nn.Module):
    def __init__(self, text_model="distilbert-base-uncased"):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size

        self.image_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.mass_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden = text_out.last_hidden_state

        # Masked mean pooling: учитываем все токены ингредиентов,
        # а padding не влияет на представление.
        mask = attention_mask.unsqueeze(-1).float()
        text_features = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        text_features = self.text_proj(text_features)

        image_features = self.image_proj(self.image_encoder(image))

        # Масштабирование делает числовой признак устойчивее для MLP.
        mass_features = self.mass_proj((mass.unsqueeze(1) / 500.0))

        fused = torch.cat(
            [text_features, image_features, mass_features], dim=1
        )
        return self.head(fused).squeeze(1)
