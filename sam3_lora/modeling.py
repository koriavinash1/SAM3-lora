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

    fn = None
    last_error = None
    
    for module_name, fn_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, fn_name):
                fn = getattr(module, fn_name)
                logger.info("Found SAM3 builder: %s.%s", module_name, fn_name)
                break
        except (ImportError, AttributeError) as e:
            last_error = e
            continue

    if fn is None:
        error_msg = f"Could not find a valid SAM3 builder function. Tried: {candidates}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise RuntimeError(error_msg)

    logger.info("Function loaded")
    model = fn(checkpoint)
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
