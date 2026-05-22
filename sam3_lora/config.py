from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


SplitName = Literal["train", "val", "test"]


@dataclass
class DatasetConfig:
    coco_json: Path
    img_root: Path
    split: SplitName


@dataclass
class LoraConfig:
    rank: int
    alpha: float
    dropout: float
    target_modules: list[str]
    trainable_fraction: float


@dataclass
class TrainConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    output_dir: Path
    log_every_steps: int
    val_split_ratio: float = 0.1
    use_class_weights: bool = True
    loss_type: str = "combined"  # "dice", "focal", or "combined"
    enable_validation: bool = True
    validation_frequency: int = 1  # Validate every N batches
    save_visualizations: bool = True
    visualization_frequency: int = 10  # Save viz every N validations


@dataclass
class ModelConfig:
    sam3_checkpoint: str
    device: str


@dataclass
class RunConfig:
    dataset: DatasetConfig
    lora: LoraConfig
    train: TrainConfig
    model: ModelConfig



def load_config(path: str | Path) -> RunConfig:
    payload = yaml.safe_load(Path(path).read_text())
    dataset = DatasetConfig(
        coco_json=Path(payload["dataset"]["coco_json"]),
        img_root=Path(payload["dataset"]["img_root"]),
        split=payload["dataset"]["split"],
    )
    lora = LoraConfig(
        rank=int(payload["lora"]["rank"]),
        alpha=float(payload["lora"]["alpha"]),
        dropout=float(payload["lora"]["dropout"]),
        target_modules=list(payload["lora"]["target_modules"]),
        trainable_fraction=float(payload["lora"]["trainable_fraction"]),
    )
    train = TrainConfig(
        seed=int(payload["train"]["seed"]),
        epochs=int(payload["train"]["epochs"]),
        batch_size=int(payload["train"]["batch_size"]),
        learning_rate=float(payload["train"]["learning_rate"]),
        output_dir=Path(payload["train"]["output_dir"]),
        log_every_steps=int(payload["train"]["log_every_steps"]),
        val_split_ratio=float(payload["train"].get("val_split_ratio", 0.1)),
        use_class_weights=bool(payload["train"].get("use_class_weights", True)),
        loss_type=str(payload["train"].get("loss_type", "combined")),
        enable_validation=bool(payload["train"].get("enable_validation", True)),
        validation_frequency=int(payload["train"].get("validation_frequency", 1)),
        save_visualizations=bool(payload["train"].get("save_visualizations", True)),
        visualization_frequency=int(payload["train"].get("visualization_frequency", 10)),
    )
    model = ModelConfig(
        sam3_checkpoint=str(payload["model"]["sam3_checkpoint"]),
        device=str(payload["model"]["device"]),
    )
    return RunConfig(dataset=dataset, lora=lora, train=train, model=model)
