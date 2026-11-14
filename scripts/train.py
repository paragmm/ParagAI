"""Command-line script to train ParagAI models."""

import os
import sys

# Ensure repository root and src are in Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "src"))

import argparse

import yaml

from paragai.config import ModelConfig, TrainConfig
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.bpe_tokenizer import BPETokenizer
from paragai.tokenizer.char_tokenizer import CharTokenizer
from paragai.training.dataset import TextDataset
from paragai.training.trainer import Trainer
from paragai.utils.logger import get_logger
from paragai.utils.seed import set_seed

logger = get_logger("TrainScript")


def main():
    parser = argparse.ArgumentParser(description="Train a ParagAI language model")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml", help="Path to train config YAML"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_small.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--data", type=str, default="data/raw/input.txt", help="Path to training text file"
    )
    parser.add_argument(
        "--tokenizer_type", type=str, choices=["char", "bpe"], default="char", help="Tokenizer type"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configurations
    with open(args.config, "r", encoding="utf-8") as f:
        train_yaml = yaml.safe_load(f)["training"]
    train_config = TrainConfig(**train_yaml)

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_yaml = yaml.safe_load(f)["model"]

    # If data path is a directory, automatically merge all text files
    if os.path.isdir(args.data):
        try:
            from scripts.prepare_data import merge_categorized_files
        except ImportError:
            from prepare_data import merge_categorized_files
        processed_file = "data/processed/combined_corpus.txt"
        logger.info(
            f"Directory detected at '{args.data}'. Merging all files into '{processed_file}'..."
        )
        merge_categorized_files(raw_dir=args.data, output_file=processed_file)
        args.data = processed_file

    # Load or generate sample text if not found
    if not os.path.exists(args.data):
        logger.info(f"Training data not found at {args.data}. Creating default sample corpus...")
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        sample_text = (
            "ParagAI is an intelligent autoregressive language model designed and built from scratch.\n"
            "First-principles engineering enables full mastery of neural attention and gradient descent.\n"
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune,\n"
            "Or to take arms against a sea of troubles and by opposing end them.\n"
        ) * 50
        with open(args.data, "w", encoding="utf-8") as f:
            f.write(sample_text)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    # Build tokenizer
    if args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=model_yaml.get("vocab_size", 512))
        tokenizer.train(text)
    else:
        tokenizer = CharTokenizer(list(text))

    # Update vocab_size in model config to match tokenizer
    model_yaml["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(**model_yaml)

    # Save tokenizer
    tok_dir = os.path.join(train_config.checkpoint_dir)
    os.makedirs(tok_dir, exist_ok=True)
    tokenizer.save(os.path.join(tok_dir, "tokenizer.json"))

    dataset = TextDataset.from_text(text, tokenizer, val_ratio=0.1)
    model = ParagAIModel(model_config)

    trainer = Trainer(model, dataset, train_config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
