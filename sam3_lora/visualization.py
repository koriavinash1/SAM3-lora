"""Visualization utilities for segmentation results."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_segmentation(
    image: dict,  # Changed type hint to allow optional import
    predictions: dict,
    ground_truth: dict | None = None,
    output_path: Path | None = None,
) -> dict | None:
    """
    Plot segmentation results with class labels and predictions.
    
    Args:
        image: Input image (H, W, 3) normalized to [0, 1] or [0, 255]
        predictions: Dict with keys:
            - "masks": (H, W) or (N, H, W) binary masks
            - "logits": (H, W) or (N, H, W) prediction logits/scores
            - "class_ids": List of class IDs
            - "class_names": List of class names (optional)
        ground_truth: Dict with same structure as predictions (optional)
        output_path: Path to save figure
        
    Returns:
        Visualization image as numpy array if output_path is None, else None
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib or numpy not installed, skipping visualization")
        return None
    
    # Ensure image is in correct range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Determine number of subplots needed
    num_plots = 1  # Input image
    if ground_truth is not None:
        num_plots += 1
    num_plots += 1  # Predictions
    
    # Additional plots for per-class segmentations
    num_class_ids = len(predictions.get("class_ids", []))
    num_plots += min(num_class_ids, 3)  # Show up to 3 class segmentations
    
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.tolist()
    
    plot_idx = 0
    
    # Plot input image
    axes[plot_idx].imshow(image)
    axes[plot_idx].set_title("Input Image")
    axes[plot_idx].axis("off")
    plot_idx += 1
    
    # Plot ground truth if provided
    if ground_truth is not None and "masks" in ground_truth:
        gt_masks = ground_truth["masks"]
        if gt_masks.ndim == 3:
            gt_combined = gt_masks.max(axis=0)  # Combine multiple masks
        else:
            gt_combined = gt_masks
        
        axes[plot_idx].imshow(image)
        axes[plot_idx].imshow(gt_combined, alpha=0.4, cmap="jet")
        axes[plot_idx].set_title("Ground Truth")
        axes[plot_idx].axis("off")
        plot_idx += 1
    
    # Plot predictions
    if "masks" in predictions:
        pred_masks = predictions["masks"]
        if pred_masks.ndim == 3:
            pred_combined = pred_masks.max(axis=0)
        else:
            pred_combined = pred_masks
        
        axes[plot_idx].imshow(image)
        axes[plot_idx].imshow(pred_combined, alpha=0.4, cmap="jet")
        
        # Add class info to title
        class_ids = predictions.get("class_ids", [])
        class_names = predictions.get("class_names", [str(cid) for cid in class_ids])
        title_str = "Predictions\n"
        title_str += ", ".join(f"{name}(id={cid})" for name, cid in zip(class_names, class_ids))
        axes[plot_idx].set_title(title_str, fontsize=8)
        axes[plot_idx].axis("off")
        plot_idx += 1
    
    # Plot individual class segmentations
    class_ids = predictions.get("class_ids", [])
    for i, class_id in enumerate(class_ids[:3]):  # Show max 3 classes
        if plot_idx >= len(axes):
            break
        
        if "masks" in predictions:
            masks = predictions["masks"]
            if masks.ndim == 3 and i < masks.shape[0]:
                mask = masks[i]
            elif masks.ndim == 2:
                mask = masks
            else:
                continue
            
            class_name = predictions.get("class_names", [str(cid) for cid in class_ids])
            if i < len(class_name):
                class_name = class_name[i]
            else:
                class_name = str(class_id)
            
            axes[plot_idx].imshow(image)
            axes[plot_idx].imshow(mask, alpha=0.4, cmap="Blues")
            axes[plot_idx].set_title(f"Class: {class_name}\nID: {class_id}")
            axes[plot_idx].axis("off")
            plot_idx += 1
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        logger.info("Saved visualization to %s", output_path)
        plt.close(fig)
        return None
    else:
        # Return as numpy array
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image_array


def assign_open_set_labels(
    predictions: dict,
    class_name_map: dict[int, str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Assign open-set class labels to predictions.
    
    Args:
        predictions: Dict with prediction masks and logits
        class_name_map: Mapping from class_id to class name
        threshold: Confidence threshold for assigning labels
        
    Returns:
        Updated predictions dict with assigned labels
    """
    if class_name_map is None:
        class_name_map = {}
    
    class_ids = predictions.get("class_ids", [])
    class_names = []
    
    for class_id in class_ids:
        if class_id in class_name_map:
            class_names.append(class_name_map[class_id])
        else:
            # Use generic open-set label
            class_names.append(f"Object_{class_id}")
    
    predictions["class_names"] = class_names
    
    # Add confidence scores if available
    if "logits" in predictions:
        logits = predictions["logits"]
        if logits.ndim >= 2:
            scores = logits.max(axis=tuple(range(1, logits.ndim)))
            predictions["confidence_scores"] = scores
    
    return predictions
