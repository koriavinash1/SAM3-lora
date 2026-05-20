# SAM3-LoRA (COCO split-aware)

This repository provides a structured, Poetry-managed fine-tuning scaffold for **SAM3 + LoRA** with:

- multiple LoRA fine-tuning levels (1%, 5%, 10% trainable parameters)
- COCO JSON ingestion with per-image split support (`train` / `val` / `test`)
- YAML-driven runs
- Bash scripts for reproducible execution
- logging + run summaries for traceability

## Project layout

- `sam3_lora/`
  - `config.py` - typed YAML config loader
  - `dataset.py` - split-aware COCO dataset parser
  - `lora.py` - LoRA injection + trainable parameter fraction control
  - `modeling.py` - SAM3 loader integration
  - `train.py` - training orchestrator + logging
  - `cli.py` - command line interface
- `configs/`
  - `lora_1pct.yaml`
  - `lora_5pct.yaml`
  - `lora_10pct.yaml`
- `scripts/`
  - `train.sh`
  - `train_all_levels.sh`
- `tests/`
  - unit tests for split handling and trainable-fraction selection

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

Logged fields include split, step, image id, annotation count, and trainable parameter ratio.

## Tests

```bash
poetry run pytest -q
```
