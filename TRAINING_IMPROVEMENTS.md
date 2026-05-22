# Training Improvements Documentation

This document describes the recent improvements made to the SAM3-LoRA training pipeline to address the zero-loss issue, handle class imbalance, add validation, and improve logging.

## Key Improvements

### 1. Fixed Zero Loss Issue

**Problem:** The original training loop was using random tensors instead of actual images, resulting in zero loss values.

**Solution:**
- Implemented actual image loading from disk using PIL
- Images are now properly preprocessed and resized to model input size
- The model receives real image data instead of random tensors

**Code Changes:**
- Added `_load_image()` function in `train.py` that loads images from disk and handles preprocessing
- Image loading includes fallback to random tensors if PIL is unavailable or file doesn't exist

### 2. Inverse Weighted Loss for Imbalanced Classes

**Problem:** The dataset is heavily imbalanced in terms of class distribution, leading to poor model convergence.

**Solution:**
- Implemented class weight computation based on inverse class frequency
- Weights are normalized to have a mean of 1 for numerical stability
- Three loss functions available: Dice loss, Focal loss, and combined Dice+Focal

**Features:**
- `loss.py`: Contains three loss functions:
  - `dice_loss()`: Weighted Dice coefficient loss
  - `focal_loss()`: Focal loss with class weighting for handling hard negatives
  - `combined_loss()`: Combines Dice and Focal with configurable weights
  
- `compute_class_weights()`: Analyzes dataset and computes inverse class weights
  - Counts annotations per class
  - Computes inverse weights: w_i = 1 / count_i
  - Normalizes weights to improve numerical stability
  - Logs class distribution statistics

**Configuration:**
```yaml
train:
  use_class_weights: true      # Enable class weight computation
  loss_type: "combined"         # Choose loss function: "dice", "focal", or "combined"
```

### 3. Validation Loop

**Problem:** No validation metrics were computed during training, making it difficult to assess model performance.

**Solution:**
- Implemented validation loop that runs after specified number of training batches
- Validation metrics include loss computation on held-out validation set
- Validation frequency is configurable to balance training speed and monitoring

**Features:**
- `_validate()` function computes validation loss on subset of validation data
- Validation loss is logged separately from training loss
- Supports same loss functions as training (Dice, Focal, or combined)

**Configuration:**
```yaml
train:
  enable_validation: true           # Enable validation
  validation_frequency: 1           # Run validation every N training batches
  val_split_ratio: 0.1             # Use 10% of training data for validation
```

### 4. Segmentation Visualization

**Problem:** No way to visualize model predictions during training.

**Solution:**
- Implemented `visualization.py` module for creating segmentation visualizations
- Visualizations show input image, predictions, ground truth, and per-class masks
- Open-set class labels are automatically assigned

**Features:**
- `plot_segmentation()`: Creates matplotlib figures showing:
  - Input image
  - Ground truth masks (if available)
  - Combined prediction masks
  - Individual class segmentations (up to 3 classes shown)
  - Class names and IDs

- `assign_open_set_labels()`: Assigns open-set labels to predictions
  - Maps class IDs to human-readable names
  - Computes confidence scores from logits
  - Handles unknown classes gracefully

**Configuration:**
```yaml
train:
  save_visualizations: true        # Enable visualization saving
  visualization_frequency: 10      # Save visualization every N validations
```

### 5. Improved Training Logging

**Problem:** The original code logged every single image, creating excessive log output.

**Solution:**
- Changed logging to aggregate metrics over batches
- Log messages now show average loss over N steps instead of per-image
- Cleaner, more manageable log output

**Changes:**
- Logging now respects `log_every_steps` parameter correctly
- Aggregates metrics (average loss) instead of individual values
- Removed per-image logging of annotation count
- Added validation metrics to logs
- Improved log message clarity

**Sample Log Output:**
```
epoch=0 step=10 avg_loss=0.123456 record_count=10
epoch=0 step=10 val_loss=0.145678
```

## New Configuration Parameters

The training config now supports the following new parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `val_split_ratio` | float | 0.1 | Fraction of training data to use for validation |
| `use_class_weights` | bool | true | Whether to use inverse class weights |
| `loss_type` | str | "combined" | Loss function to use: "dice", "focal", or "combined" |
| `enable_validation` | bool | true | Whether to run validation during training |
| `validation_frequency` | int | 1 | Run validation every N training batches |
| `save_visualizations` | bool | true | Whether to save segmentation visualizations |
| `visualization_frequency` | int | 10 | Save visualization every N validation runs |

## Updated Config Files

All config files (`lora_1pct.yaml`, `lora_5pct.yaml`, `lora_10pct.yaml`) have been updated with the new parameters. Users can customize these based on their needs:

```yaml
train:
  seed: 42
  epochs: 2
  batch_size: 1
  learning_rate: 0.0001
  output_dir: outputs/lora_1pct
  log_every_steps: 10
  
  # New training improvements
  val_split_ratio: 0.1
  use_class_weights: true
  loss_type: "combined"
  enable_validation: true
  validation_frequency: 1
  save_visualizations: true
  visualization_frequency: 10
```

## Dataset Enhancements

The `dataset.py` module now includes:

- `get_class_statistics()`: Computes class distribution and inverse weights
- `split_train_val()`: Splits dataset into train and validation subsets
- `get_class_ids()`: Get class IDs for a specific image record

**Example Usage:**
```python
dataset = CocoSplitDataset(...)

# Get class statistics
stats = dataset.get_class_statistics()
print(stats['class_weights'])  # {1: 0.8, 2: 1.2, ...}

# Split into train/val
train_dataset, val_dataset = dataset.split_train_val()
```

## Module Dependencies

New modules use the following optional dependencies:
- `PIL` (Pillow): For image loading
- `matplotlib`: For visualization
- `numpy`: For numerical operations (used by visualization)

All dependencies are handled gracefully with fallbacks if not available.

## Performance Considerations

1. **Image Loading**: Loading real images instead of random tensors adds minimal overhead
2. **Class Weight Computation**: Done once at dataset initialization
3. **Validation**: Can be expensive; adjust `validation_frequency` to reduce frequency
4. **Visualizations**: Generated less frequently; adjust `visualization_frequency` to control disk I/O

## Usage Example

```bash
# Train with all new improvements enabled
poetry run sam3-lora train --config configs/lora_1pct.yaml

# Monitor training with logs
tail -f outputs/lora_1pct/train.log
```

## Future Enhancements

Potential future improvements:
1. Segmentation mask loading from COCO annotations
2. Data augmentation during training
3. Learning rate scheduling
4. Model checkpointing and early stopping
5. Tensorboard integration for metric tracking
6. Multi-GPU training support

## Troubleshooting

### Loss is still zero
- Check that images exist at the specified `img_root` path
- Verify COCO JSON has valid image file references
- Check model output tensors are not all zeros

### Validation not running
- Ensure `enable_validation: true` in config
- Check that dataset has enough samples (min 2 needed for train/val split)
- Verify `validation_frequency` is set appropriately

### Memory issues
- Reduce `validation_frequency` to validate less often
- Disable `save_visualizations` if visualization generation is too slow
- Reduce batch size in config

### Missing dependencies
- Install PIL: `pip install Pillow`
- Install matplotlib: `pip install matplotlib`
- Install numpy: `pip install numpy`
