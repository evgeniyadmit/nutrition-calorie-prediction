import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer


def build_transforms(image_size=224, train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class NutritionDataset(Dataset):
    def __init__(self, df, data_dir, text_model, max_length=128,
                 image_size=224, train=False):
        self.df = df.reset_index(drop=True).copy()
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.max_length = max_length
        self.transform = build_transforms(image_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.data_dir, "images", str(row["dish_id"]), "rgb.png"
        )
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        encoded = self.tokenizer(
            str(row["ingredients_text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "mass": torch.tensor(float(row["total_mass"]), dtype=torch.float32),
            "target": torch.tensor(float(row["total_calories"]), dtype=torch.float32),
            "dish_id": str(row["dish_id"]),
        }


def make_loaders(train_df, val_df, test_df, cfg):
    train_ds = NutritionDataset(
        train_df, cfg.DATA_DIR, cfg.TEXT_MODEL, cfg.MAX_LENGTH,
        cfg.IMAGE_SIZE, train=True
    )
    val_ds = NutritionDataset(
        val_df, cfg.DATA_DIR, cfg.TEXT_MODEL, cfg.MAX_LENGTH,
        cfg.IMAGE_SIZE, train=False
    )
    test_ds = NutritionDataset(
        test_df, cfg.DATA_DIR, cfg.TEXT_MODEL, cfg.MAX_LENGTH,
        cfg.IMAGE_SIZE, train=False
    )

    common = dict(
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )
