# SAM3-LoRA (COCO split-aware)

This repository provides a structured, Poetry-managed fine-tuning scaffold for **SAM3 + LoRA** with:

- multiple LoRA fine-tuning levels (1%, 5%, 10% trainable parameters)
- COCO JSON ingestion with per-image split support (`train` / `val` / `test`)
- YAML-driven runs with configurable training improvements
- Bash scripts for reproducible execution
- logging + run summaries for traceability
- **NEW**: Inverse class weighting for imbalanced datasets
- **NEW**: Validation loop with metrics tracking
- **NEW**: Segmentation visualization with class labels
- **NEW**: Improved training logging (batch aggregation instead of per-image)

## Project layout

- `sam3_lora/`
  - `config.py` - typed YAML config loader with new training parameters
  - `dataset.py` - split-aware COCO dataset parser with class statistics and train/val split
  - `lora.py` - LoRA injection + trainable parameter fraction control
  - `modeling.py` - SAM3 loader integration
  - `train.py` - training orchestrator with real image loading, validation, and logging
  - `loss.py` - weighted loss functions (Dice, Focal, Combined) for imbalanced datasets
  - `visualization.py` - segmentation visualization with class labels and open-set labeling
  - `cli.py` - command line interface
  - `logging_utils.py` - logging configuration
- `configs/`
  - `lora_1pct.yaml`
  - `lora_5pct.yaml`
  - `lora_10pct.yaml`
- `scripts/`
  - `train.sh`
  - `train_all_levels.sh`
- `tests/`
  - unit tests for split handling and trainable-fraction selection
- `TRAINING_IMPROVEMENTS.md` - detailed documentation of new training features

## Dataset format

The loader expects a COCO JSON where split can come from either:

1. `images[].split` (preferred)
2. `annotations[].attributes.split` (fallback)

This matches schemas like:

```json
{
  "images": [{"id": 1, "file_name": "a.jpg", "split": "train"}],
  "annotations": [{"image_id": 1, "attributes": {"split": "train"}}]
}
```

`img_root` should point to the root directory for `file_name` paths.

## Setup (Poetry)

```bash
poetry install
```

## Run training

Single config:

```bash
./scripts/train.sh configs/lora_1pct.yaml
```

All predefined levels:

```bash
./scripts/train_all_levels.sh
```

Direct CLI:

```bash
poetry run sam3-lora train --config configs/lora_1pct.yaml
```

## SAM3 version/checkpoint note

The trainer is designed to use the **latest SAM3 package/checkpoint** you install in your environment.

In YAML:

- `model.sam3_checkpoint`: pass current SAM3 checkpoint/model ID (for example `sam3_hiera_large`) or a local checkpoint path.

If your installed SAM3 API differs, keep SAM3 updated and adjust checkpoint/model name accordingly.

## Logging and traceability

Each run writes:

- `outputs/<run_name>/train.log`
- `outputs/<run_name>/run_summary.json`

Logged fields include:
- Training metrics: epoch, step, average loss
- Validation metrics: validation loss (when enabled)
- Class statistics: class distribution and computed weights
- Segmentation visualizations: saved at configured frequency

Run summaries include:
- Split information
- Number of training samples
- Trainable parameter statistics
- SAM3 checkpoint used
- Class statistics and weights
- Total training steps
- Number of validation runs

## Training Improvements

For detailed information about the recent training improvements, see [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md):

- **Fixed zero-loss issue**: Real images are now loaded from disk instead of using random tensors
- **Inverse class weighting**: Automatically handles imbalanced datasets using inverse class frequency weights
- **Validation loop**: Validates model performance on held-out validation set
- **Segmentation visualization**: Generates visualizations of model predictions with class labels
- **Improved logging**: Changed from per-image logging to batch-aggregated logging for cleaner output

### New Configuration Parameters

Key training parameters in config YAML:

```yaml
train:
  # Existing parameters
  seed: 42
  epochs: 2
  batch_size: 1
  learning_rate: 0.0001
  output_dir: outputs/lora_1pct
  log_every_steps: 10
  
  # New parameters for improved training
  val_split_ratio: 0.1              # Use 10% of training data for validation
  use_class_weights: true           # Use inverse class weights for imbalanced datasets
  loss_type: "combined"             # Options: "dice", "focal", or "combined"
  enable_validation: true           # Run validation during training
  validation_frequency: 1           # Validate every N training batches
  save_visualizations: true         # Save segmentation visualizations
  visualization_frequency: 10       # Save visualization every N validations
```

## Tests

```bash
poetry run pytest -q
```
