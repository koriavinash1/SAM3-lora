from __future__ import annotations

import importlib
import logging
from pathlib import Path

from sam3_lora.lora import TorchUnavailableError, _require_torch

logger = logging.getLogger(__name__)



def _load_from_sam3_package(checkpoint: str):
    candidates = [
        ("sam3.build_sam", "build_sam3"),
        ("sam3.build_sam", "build_sam"),
        ("sam3", "build_sam3_image_model"),
        ("sam3", "build_sam3"),
        ("sam3", "build_sam"),
    ]

    module = importlib.import_module("sam3")
    fn = getattr(module, "build_sam3_image_model")
    logger.info("Function loaded")
    print(module, fn, checkpoint)

    model = fn() #checkpoint_path = checkpoint)
    logger.info("Loaded SAM3 model via %s", checkpoint)
    return model


def load_sam3_model(checkpoint: str, device: str):
    torch, nn = _require_torch()

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        model = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(model, nn.Module):
            logger.info("Loaded model checkpoint from %s", checkpoint_path)
            return model.to(device)

    model = _load_from_sam3_package(checkpoint)
    if not isinstance(model, nn.Module):
        raise RuntimeError("SAM3 loader did not return a torch.nn.Module")
    return model.to(device)


__all__ = ["load_sam3_model", "TorchUnavailableError"]
