"""Loss functions for SAM3 LoRA training with class weighting."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_class_weights(dataset, torch) -> dict:
    """
    Compute inverse class weights from dataset annotations.
    
    Args:
        dataset: CocoSplitDataset instance
        torch: PyTorch module
        
    Returns:
        Dictionary with class statistics and computed weights
    """
    class_counts = {}
    total_annotations = 0
    
    # Count annotations per class
    for record in dataset.records:
        for ann in record.annotations:
            class_id = ann.get("category_id")
            if class_id is not None:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_annotations += 1
    
    if not class_counts:
        logger.warning("No annotations found in dataset")
        return {"class_counts": {}, "class_weights": {}, "total": 0}
    
    # Compute inverse weights: w_i = 1 / count_i
    class_weights = {}
    for class_id, count in class_counts.items():
        # Inverse weight with smoothing to avoid division by zero
        weight = 1.0 / (count + 1e-8)
        class_weights[class_id] = weight
    
    # Normalize weights to have mean 1
    if class_weights:
        mean_weight = sum(class_weights.values()) / len(class_weights)
        class_weights = {k: v / mean_weight for k, v in class_weights.items()}
    
    logger.info("Class distribution (before normalization):")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        logger.info("  Class %s: %s annotations (%.2f%%)", class_id, count, percentage)
    
    logger.info("Computed class weights (after normalization):")
    for class_id in sorted(class_weights.keys()):
        logger.info("  Class %s: %.4f", class_id, class_weights[class_id])
    
    return {
        "class_counts": class_counts,
        "class_weights": class_weights,
        "total_annotations": total_annotations,
    }


def dice_loss(predictions, targets, class_weights=None, torch=None, smooth=1e-5):
    """
    Compute weighted Dice loss for segmentation.
    
    Args:
        predictions: Model predictions, shape (B, C, H, W) or (B, H, W)
        targets: Ground truth masks, shape (B, H, W) or (B, C, H, W)
        class_weights: Dictionary of class_id -> weight or tensor
        torch: PyTorch module
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Scalar loss value
    """
    if torch is None:
        import torch as _torch
        torch = _torch
    
    # Handle different input shapes
    if predictions.dim() == 3:
        predictions = predictions.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    
    # Flatten spatial dimensions
    predictions = predictions.view(predictions.size(0), predictions.size(1), -1)  # (B, C, H*W)
    targets = targets.view(targets.size(0), targets.size(1), -1)  # (B, C, H*W)
    
    # Compute Dice coefficient per channel
    intersection = (predictions * targets).sum(dim=2)  # (B, C)
    union = predictions.sum(dim=2) + targets.sum(dim=2)  # (B, C)
    
    dice = (2 * intersection + smooth) / (union + smooth)  # (B, C)
    
    # Apply class weights if provided
    if class_weights is not None:
        if isinstance(class_weights, dict):
            weights = torch.tensor(
                [class_weights.get(i, 1.0) for i in range(dice.shape[1])],
                device=dice.device,
                dtype=dice.dtype
            )
        else:
            weights = class_weights
        dice = dice * weights.unsqueeze(0)
    
    # Return mean loss
    loss = 1 - dice.mean()
    return loss


def focal_loss(predictions, targets, class_weights=None, alpha=0.25, gamma=2.0, torch=None):
    """
    Compute weighted Focal loss for handling class imbalance.
    
    Args:
        predictions: Model predictions (logits or probabilities), shape (B, C, H, W) or (B, H, W)
        targets: Ground truth masks, shape (B, H, W) or (B, C, H, W)
        class_weights: Dictionary of class_id -> weight or tensor
        alpha: Balancing parameter
        gamma: Focusing parameter
        torch: PyTorch module
        
    Returns:
        Scalar loss value
    """
    if torch is None:
        import torch as _torch
        torch = _torch
    
    # Convert to probabilities if needed
    if predictions.dim() == 3:
        predictions = predictions.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    
    # Apply sigmoid to get probabilities
    p = torch.sigmoid(predictions)
    
    # Compute focal loss
    ce_loss = -(targets * torch.log(p + 1e-7) + (1 - targets) * torch.log(1 - p + 1e-7))
    focal_weight = alpha * (1 - p) ** gamma
    focal = focal_weight * ce_loss
    
    # Apply class weights if provided
    if class_weights is not None:
        if isinstance(class_weights, dict):
            weights = torch.tensor(
                [class_weights.get(i, 1.0) for i in range(focal.shape[1])],
                device=focal.device,
                dtype=focal.dtype
            )
        else:
            weights = class_weights
        focal = focal * weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    return focal.mean()


def combined_loss(predictions, targets, class_weights=None, torch=None, dice_weight=0.5):
    """
    Compute combined Dice + Focal loss.
    
    Args:
        predictions: Model predictions
        targets: Ground truth masks
        class_weights: Class weight dictionary
        torch: PyTorch module
        dice_weight: Weight for Dice loss (focal_weight = 1 - dice_weight)
        
    Returns:
        Scalar loss value
    """
    if torch is None:
        import torch as _torch
        torch = _torch
    
    dice = dice_loss(predictions, targets, class_weights, torch)
    focal = focal_loss(predictions, targets, class_weights, torch=torch)
    
    combined = dice_weight * dice + (1 - dice_weight) * focal
    return combined
