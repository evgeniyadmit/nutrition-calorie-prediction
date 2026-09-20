"""Воспроизводимый запуск обучения из конфигурационного файла.

Пример:
    python train.py --config scripts/config.py
"""
import argparse
import importlib.util
import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from scripts.dataset import make_loaders
from scripts.model import NutritionModel
from scripts.utils import seed_everything, train as fit_model


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("project_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Не удалось загрузить конфигурацию: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config()


def decode_ingredient_column(dish, ingredients):
    id_to_name = dict(zip(ingredients["id"].astype(str), ingredients["ingr"].astype(str)))

    def decode(value):
        names = []
        for item in str(value).split(";"):
            item = item.strip()
            if not item:
                continue
            if item in id_to_name:
                names.append(id_to_name[item])
            else:
                numeric = item.replace("ingr_", "").lstrip("0") or "0"
                names.append(id_to_name.get(numeric, item))
        return ", ".join(names)

    dish = dish.copy()
    dish["ingredients_text"] = dish["ingredients"].apply(decode)
    return dish


def train(config_path):
    cfg = load_config(config_path)
    seed_everything(cfg.SEED)

    dish = pd.read_csv(os.path.join(cfg.DATA_DIR, "dish.csv"))
    ingredients = pd.read_csv(os.path.join(cfg.DATA_DIR, "ingredients.csv"))
    dish = decode_ingredient_column(dish, ingredients)

    train_full = dish[dish["split"].astype(str).str.lower() == "train"].copy()
    test_df = dish[dish["split"].astype(str).str.lower() == "test"].copy()
    train_df, val_df = train_test_split(
        train_full,
        test_size=0.15,
        random_state=cfg.SEED,
    )

    train_loader, val_loader, _ = make_loaders(train_df, val_df, test_df, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NutritionModel(cfg.TEXT_MODEL)

    history, best_val_mae = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )
    print(f"Лучший validation MAE: {best_val_mae:.2f}")
    print(f"Checkpoint: {cfg.MODEL_PATH}")
    return history, best_val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config.py")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
