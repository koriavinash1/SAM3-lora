from __future__ import annotations

import argparse

from sam3_lora.config import load_config
from sam3_lora.train import run_training



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM3 LoRA fine-tuning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run LoRA fine-tuning")
    train_parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = load_config(args.config)
        run_training(config)


if __name__ == "__main__":
    main()
