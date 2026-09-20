import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, return_predictions=False):
    model.eval()
    criterion = nn.L1Loss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    predictions, targets, dish_ids = [], [], []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        pred = model(input_ids, attention_mask, image, mass)
        total_loss += criterion(pred, target).item()
        total_n += target.size(0)

        if return_predictions:
            predictions.extend(pred.detach().cpu().numpy().tolist())
            targets.extend(target.detach().cpu().numpy().tolist())
            dish_ids.extend(batch["dish_id"])

    mae = total_loss / max(total_n, 1)
    if return_predictions:
        return mae, predictions, targets, dish_ids
    return mae


def train(model, train_loader, val_loader, cfg, device):
    seed_everything(cfg.SEED)
    model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": list(model.text_proj.parameters()) +
                   list(model.image_proj.parameters()) +
                   list(model.mass_proj.parameters()) +
                   list(model.head.parameters()),
         "lr": cfg.CLASSIFIER_LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.L1Loss()

    best_val_mae = float("inf")
    history = {"train_mae": [], "val_mae": []}

    model_dir = os.path.dirname(cfg.MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_abs_error = 0.0
        train_n = 0

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            mass = batch["mass"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(input_ids, attention_mask, image, mass)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_abs_error += torch.abs(pred.detach() - target).sum().item()
            train_n += target.size(0)

        train_mae = train_abs_error / max(train_n, 1)
        val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mae)

        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        marker = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), cfg.MODEL_PATH)
            marker = " | сохранена лучшая модель"

        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
            f"train MAE {train_mae:.2f} | val MAE {val_mae:.2f}{marker}"
        )

    return history, best_val_mae
