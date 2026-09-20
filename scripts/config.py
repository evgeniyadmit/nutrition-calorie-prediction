from dataclasses import dataclass

@dataclass
class Config:
    DATA_DIR: str = "/content/data"
    MODEL_PATH: str = "/content/drive/MyDrive/nutrition_best_model.pt"

    TEXT_MODEL: str = "distilbert-base-uncased"
    IMAGE_SIZE: int = 224
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 0
    EPOCHS: int = 25

    TEXT_LR: float = 5e-6
    IMAGE_LR: float = 5e-6
    CLASSIFIER_LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    SEED: int = 42
