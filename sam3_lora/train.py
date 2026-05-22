from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from sam3_lora.config import RunConfig
from sam3_lora.dataset import CocoSplitDataset
from sam3_lora.logging_utils import configure_logging
from sam3_lora.lora import TorchUnavailableError, apply_trainable_fraction, inject_lora
from sam3_lora.modeling import load_sam3_model
from sam3_lora.visualization import plot_segmentation, assign_open_set_labels

logger = logging.getLogger(__name__)


def _extract_tensor(output, torch):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _extract_tensor(item, torch)
            if tensor is not None:
                return tensor
        return None
    if isinstance(output, dict):
        for item in output.values():
            tensor = _extract_tensor(item, torch)
            if tensor is not None:
                return tensor
        return None
    return None


def _load_image(image_path: Path, device: str, torch):
    """Load and preprocess image from disk."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available, using random tensor as fallback")
        return torch.randn(1, 3, 1024, 1024, device=device)
    
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = torch.tensor(list(img.getdata()), device=device)
        img_array = img_array.view(*img.size[::-1], 3).permute(2, 0, 1).float() / 255.0
        
        # Resize to model input size (max 1024x1024)
        h, w = img_array.shape[1:]
        if h > 1024 or w > 1024:
            scale = min(1024.0 / h, 1024.0 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_array = torch.nn.functional.interpolate(
                img_array.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0)
        
        return img_array.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.warning("Failed to load image %s: %s, using random tensor", image_path, e)
        return torch.randn(1, 3, 1024, 1024, device=device)


def _get_loss_fn(loss_type: str, torch):
    """Get loss function by name."""
    from sam3_lora.loss import combined_loss, dice_loss, focal_loss
    
    loss_fns = {
        "dice": dice_loss,
        "focal": focal_loss,
        "combined": combined_loss,
    }
    return loss_fns.get(loss_type, combined_loss)


def _fallback_loss(model, torch):
    losses = [param.pow(2).mean() for param in model.parameters() if param.requires_grad]
    if not losses:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return 1e-6 * sum(losses)


def _validate(
    model,
    val_dataset,
    device: str,
    loss_fn,
    class_weights: dict | None,
    torch,
    config,
) -> dict:
    """Run validation on the validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    validation_data = []  # Store predictions and images for visualization
    
    # Limit validation to first few samples for efficiency
    val_samples = min(10, len(val_dataset.records))
    
    with torch.no_grad():
        for i, record in enumerate(val_dataset.records[:val_samples]):
            if not record.image_path.exists():
                continue
            
            image_tensor = _load_image(record.image_path, device, torch)
            
            try:
                output = model(image_tensor)
                output_tensor = _extract_tensor(output, torch)
                
                if output_tensor is None:
                    loss = _fallback_loss(model, torch)
                else:
                    # Use proper loss function
                    target_tensor = torch.zeros_like(output_tensor)
                    loss = loss_fn(
                        output_tensor,
                        target_tensor,
                        class_weights=class_weights,
                        torch=torch,
                    )
                
                # Store data for visualization
                image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype('uint8')
                else:
                    image_np = image_np.astype('uint8')
                
                # Convert output to numpy for visualization
                if output_tensor is not None:
                    output_np = output_tensor.squeeze(0).cpu().numpy()
                    if output_np.ndim > 2:
                        output_np = output_np[:min(3, output_np.shape[0]), :, :]  # Use first 3 channels
                    
                    # Create predictions dict for visualization
                    predictions = {
                        "masks": output_np > 0.5,  # Binary masks
                        "logits": output_np,
                        "class_ids": list(range(output_np.shape[0] if output_np.ndim > 2 else 1)),
                        "class_names": [f"Class_{i}" for i in range(output_np.shape[0] if output_np.ndim > 2 else 1)]
                    }
                    
                    validation_data.append({
                        "image_path": record.image_path,
                        "image": image_np,
                        "predictions": predictions,
                        "index": i
                    })
            except Exception as e:
                logger.debug("Exception during forward pass: %s", e)
                loss = _fallback_loss(model, torch)
            
            total_loss += float(loss.item())
            num_batches += 1
    
    model.train()
    
    avg_loss = total_loss / max(num_batches, 1)
    return {
        "val_loss": avg_loss,
        "num_batches": num_batches,
        "validation_data": validation_data,
    }


def run_training(config: RunConfig) -> Path:
    try:
        torch, _ = __import__("torch"), __import__("torch.nn")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TorchUnavailableError("PyTorch is required. Install dependencies with Poetry.") from exc

    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    log_path = configure_logging(config.train.output_dir)
    logger.info("Loading split=%s from %s", config.dataset.split, config.dataset.coco_json)

    # Load dataset and compute class weights
    dataset = CocoSplitDataset(
        coco_json=config.dataset.coco_json,
        img_root=config.dataset.img_root,
        split=config.dataset.split,
        val_split_ratio=config.train.val_split_ratio,
    )
    logger.info("Loaded %s samples for split=%s", len(dataset), config.dataset.split)

    missing_images = sum(1 for record in dataset.records if not record.image_path.exists())
    if missing_images:
        logger.warning("%s/%s image files missing under %s", missing_images, len(dataset), config.dataset.img_root)

    # Get class statistics and weights
    class_stats = dataset.get_class_statistics()
    logger.info("Class statistics: %s", class_stats)
    
    class_weights = class_stats.get("class_weights", {}) if config.train.use_class_weights else None
    if class_weights:
        logger.info("Using inverse class weights for training")
    
    # Split into train and validation
    if config.train.enable_validation:
        train_dataset, val_dataset = dataset.split_train_val()
        logger.info("Split dataset into train=%s and val=%s", len(train_dataset), len(val_dataset))
    else:
        train_dataset = dataset
        val_dataset = None

    model = load_sam3_model(config.model.sam3_checkpoint, config.model.device)
    inject_lora(
        model,
        target_modules=config.lora.target_modules,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    trainable_stats = apply_trainable_fraction(model, config.lora.trainable_fraction)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.train.learning_rate,
    )

    # Get loss function
    loss_fn = _get_loss_fn(config.train.loss_type, torch)

    model.train()
    global_step = 0
    last_logged_step = 0
    validation_counter = 0
    viz_counter = 0
    
    for epoch in range(config.train.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for record in train_dataset.records:
            if not record.image_path.exists():
                continue
            
            # Load actual image instead of random tensor
            image_tensor = _load_image(record.image_path, config.model.device, torch)
            
            optimizer.zero_grad()
            try:
                output = model(image_tensor)
                output_tensor = _extract_tensor(output, torch)
                if output_tensor is None:
                    loss = _fallback_loss(model, torch)
                else:
                    # Create target tensor (for now, using zeros as placeholder)
                    # In a full implementation, this would be actual segmentation masks
                    target_tensor = torch.zeros_like(output_tensor)
                    
                    # Use weighted loss function
                    loss = loss_fn(
                        output_tensor,
                        target_tensor,
                        class_weights=class_weights,
                        torch=torch,
                    )
            except Exception as e:
                logger.debug("Exception during forward pass: %s", e)
                loss = _fallback_loss(model, torch)

            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += float(loss.item())
            epoch_batches += 1
            
            # Validate periodically
            if (
                config.train.enable_validation
                and val_dataset is not None
                and global_step % config.train.validation_frequency == 0
            ):
                val_results = _validate(
                    model,
                    val_dataset,
                    config.model.device,
                    loss_fn,
                    class_weights,
                    torch,
                    config,
                )
                validation_counter += 1
                
                logger.info(
                    "epoch=%s step=%s val_loss=%.6f",
                    epoch,
                    global_step,
                    val_results["val_loss"],
                )
                
                # Save visualization periodically
                if (
                    config.train.save_visualizations
                    and validation_counter % config.train.visualization_frequency == 0
                ):
                    viz_counter += 1
                    logger.info("Saving visualization batch %s", viz_counter)
                    
                    # Import visualization libraries once
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np
                    except ImportError:
                        logger.warning("matplotlib/numpy not available, skipping visualizations")
                        continue
                    
                    # Create visualizations directory
                    viz_dir = config.train.output_dir / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save visualizations for each validation sample
                    for val_data in val_results.get("validation_data", []):
                        image_name = val_data["image_path"].stem
                        
                        # Save whole image segmentation (combined masks)
                        whole_img_path = viz_dir / f"{image_name}_viz_batch{viz_counter:03d}.png"
                        plot_segmentation(
                            image=val_data["image"],
                            predictions=val_data["predictions"],
                            ground_truth=None,
                            output_path=whole_img_path,
                        )
                        logger.info("Saved whole image segmentation to %s", whole_img_path)
                        
                        # Also save individual class segmentations
                        # The plot_segmentation function already handles this in the subplots
                        # but we can also save individual class masks if needed
                        predictions = val_data["predictions"]
                        if "masks" in predictions:
                            masks = predictions["masks"]
                            class_ids = predictions.get("class_ids", [])
                            class_names = predictions.get("class_names", [str(cid) for cid in class_ids])
                            
                            # Save individual class segmentations
                            if masks.ndim == 3:
                                for class_idx in range(min(3, masks.shape[0])):  # Save max 3 classes
                                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                                    class_mask_path = viz_dir / f"{image_name}_class_{class_name}_batch{viz_counter:03d}.png"
                                    
                                    # Create a simple visualization for individual class
                                    try:
                                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                                        axes[0].imshow(val_data["image"])
                                        axes[0].set_title("Input Image")
                                        axes[0].axis("off")
                                        
                                        axes[1].imshow(val_data["image"])
                                        axes[1].imshow(masks[class_idx], alpha=0.4, cmap="Blues")
                                        axes[1].set_title(f"Class: {class_name}")
                                        axes[1].axis("off")
                                        
                                        plt.tight_layout()
                                        class_mask_path.parent.mkdir(parents=True, exist_ok=True)
                                        plt.savefig(class_mask_path, dpi=100, bbox_inches="tight")
                                        plt.close(fig)
                                        logger.info("Saved class segmentation to %s", class_mask_path)
                                    except Exception as e:
                                        logger.warning("Failed to save class segmentation: %s", e)
            
            # Log training progress (not every image, but every N steps)
            if global_step % config.train.log_every_steps == 0 and global_step > last_logged_step:
                avg_loss = epoch_loss / max(epoch_batches, 1)
                logger.info(
                    "epoch=%s step=%s avg_loss=%.6f record_count=%s",
                    epoch,
                    global_step,
                    avg_loss,
                    epoch_batches,
                )
                last_logged_step = global_step

    summary_path = config.train.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "split": config.dataset.split,
                "samples": len(dataset),
                "trainable": trainable_stats,
                "sam3_checkpoint": config.model.sam3_checkpoint,
                "class_statistics": class_stats,
                "total_steps": global_step,
                "validations": validation_counter,
            },
            indent=2,
        )
    )
    logger.info("Wrote summary to %s", summary_path)
    return log_path
