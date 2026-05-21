#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/lora_1pct.yaml}"

# poetry install
poetry run sam3-lora train --config "${CONFIG_PATH}"
