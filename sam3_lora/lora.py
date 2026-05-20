from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)



def choose_trainable_parameter_names(
    named_parameter_sizes: list[tuple[str, int]],
    trainable_fraction: float,
    always_trainable_patterns: Iterable[str] = (),
) -> set[str]:
    if not 0 < trainable_fraction <= 1:
        raise ValueError("trainable_fraction must be in (0, 1]")

    total_params = sum(size for _, size in named_parameter_sizes)
    target = max(1, math.ceil(total_params * trainable_fraction))

    selected: set[str] = set()
    selected_count = 0

    compiled = [re.compile(pattern) for pattern in always_trainable_patterns]
    for name, size in named_parameter_sizes:
        if any(pattern.search(name) for pattern in compiled):
            selected.add(name)
            selected_count += size

    for name, size in sorted(named_parameter_sizes, key=lambda item: item[1], reverse=True):
        if selected_count >= target:
            break
        if name in selected:
            continue
        selected.add(name)
        selected_count += size

    logger.info(
        "Selected %s/%s params (%.4f) across %s tensors",
        selected_count,
        total_params,
        selected_count / max(total_params, 1),
        len(selected),
    )
    return selected


class TorchUnavailableError(RuntimeError):
    pass



def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TorchUnavailableError(
            "PyTorch is required. Install dependencies via `poetry install`."
        ) from exc
    return torch, nn


class _LoRALinearBase:
    pass



def make_lora_linear(linear, rank: int, alpha: float, dropout: float):
    torch, nn = _require_torch()

    class LoRALinear(nn.Module, _LoRALinearBase):
        def __init__(self, base_linear):
            super().__init__()
            self.base_linear = base_linear
            self.rank = rank
            self.alpha = alpha
            self.scale = alpha / max(rank, 1)
            self.dropout = nn.Dropout(dropout)

            in_features = base_linear.in_features
            out_features = base_linear.out_features

            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            self.base_linear.weight.requires_grad_(False)
            if self.base_linear.bias is not None:
                self.base_linear.bias.requires_grad_(False)

        def forward(self, x):
            base = self.base_linear(x)
            lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
            return base + (self.scale * lora_out)

    return LoRALinear(linear)



def _get_module_and_name(root, path: str):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]



def inject_lora(model, target_modules: Iterable[str], rank: int, alpha: float, dropout: float) -> int:
    _, nn = _require_torch()

    target_patterns = [re.compile(pattern) for pattern in target_modules]
    replaced = 0

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(pattern.search(module_name) for pattern in target_patterns):
            continue

        parent, child_name = _get_module_and_name(model, module_name)
        setattr(parent, child_name, make_lora_linear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1

    logger.info("Injected LoRA into %s linear modules", replaced)
    return replaced



def apply_trainable_fraction(model, trainable_fraction: float) -> dict[str, float]:
    named_parameters = [(name, int(param.numel())) for name, param in model.named_parameters()]
    keep = choose_trainable_parameter_names(
        named_parameter_sizes=named_parameters,
        trainable_fraction=trainable_fraction,
        always_trainable_patterns=[r"lora_A", r"lora_B"],
    )

    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        should_train = name in keep
        parameter.requires_grad_(should_train)
        pcount = int(parameter.numel())
        total += pcount
        if should_train:
            trainable += pcount

    ratio = trainable / max(total, 1)
    logger.info("Trainable parameters: %s/%s (%.4f)", trainable, total, ratio)
    return {"trainable": trainable, "total": total, "ratio": ratio}
