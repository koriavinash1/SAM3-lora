#!/usr/bin/env bash
set -euo pipefail

for cfg in configs/lora_1pct.yaml configs/lora_5pct.yaml configs/lora_10pct.yaml; do
  poetry run sam3-lora train --config "${cfg}"
done
