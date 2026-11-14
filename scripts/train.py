"""Command-line script to train ParagAI models."""

import os
import sys

# Ensure repository root and src are in Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "src"))

import argparse
import json
import shutil

import numpy as np
import yaml

from paragai.config import ModelConfig, TrainConfig
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.bpe_tokenizer import BPETokenizer
from paragai.tokenizer.char_tokenizer import CharTokenizer
from paragai.tokenizer.indic_bpe import IndicBPETokenizer
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
        "--tokenizer_type",
        type=str,
        choices=["char", "bpe", "indic_bpe"],
        default="bpe",
        help="Tokenizer type",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to pre-trained tokenizer JSON (e.g. checkpoints/tokenizer_indic.json)",
    )
    parser.add_argument(
        "--max_iters", type=int, default=None, help="Override max training iterations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override training batch size"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override training device (e.g. cuda, cpu)"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configurations
    with open(args.config, "r", encoding="utf-8") as f:
        train_yaml = yaml.safe_load(f)["training"]

    # Command line overrides for train config
    if args.max_iters is not None:
        train_yaml["max_iters"] = args.max_iters
    if args.batch_size is not None:
        train_yaml["batch_size"] = args.batch_size
    if args.device is not None:
        train_yaml["device"] = args.device

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

    # Resolve tokenizer
    tok_dir = os.path.join(train_config.checkpoint_dir)
    os.makedirs(tok_dir, exist_ok=True)
    target_tok_path = os.path.join(tok_dir, "tokenizer.json")

    # If user provided a specific path, or if tokenizer_indic.json exists in checkpoint_dir
    tok_candidate = args.tokenizer_path
    if not tok_candidate:
        indic_default = os.path.join(tok_dir, "tokenizer_indic.json")
        if os.path.exists(indic_default) and args.tokenizer_type in ["bpe", "indic_bpe"]:
            tok_candidate = indic_default

    tokenizer = None
    if tok_candidate and os.path.exists(tok_candidate):
        logger.info(f"Loading pre-trained tokenizer from {tok_candidate}...")
        with open(tok_candidate, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("tokenizer_class") == "indic_bpe" or args.tokenizer_type == "indic_bpe":
            tokenizer = IndicBPETokenizer.load(tok_candidate)
        elif data.get("type") == "bpe":
            tokenizer = BPETokenizer.load(tok_candidate)
        else:
            tokenizer = CharTokenizer.load(tok_candidate)

        # Ensure tokenizer.json also contains a copy for downstream tools
        if os.path.abspath(tok_candidate) != os.path.abspath(target_tok_path):
            shutil.copyfile(tok_candidate, target_tok_path)
            logger.info(f"Synced tokenizer state to {target_tok_path}")
    elif args.tokenizer_type == "indic_bpe":
        tokenizer = IndicBPETokenizer(vocab_size=model_yaml.get("vocab_size", 4096))
        tokenizer.train(text)
        tokenizer.save(target_tok_path)
    elif args.tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=model_yaml.get("vocab_size", 512))
        tokenizer.train(text)
        tokenizer.save(target_tok_path)
    else:
        tokenizer = CharTokenizer(list(text))
        tokenizer.save(target_tok_path)

    # Update vocab_size in model config to match tokenizer
    model_yaml["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(**model_yaml)

    logger.info("Tokenizing dataset line-by-line (ultra-fast mode)...")
    lines = text.splitlines(keepends=True)
    token_ids: list[int] = []
    total_lines = len(lines)
    for idx, line in enumerate(lines):
        if line:
            token_ids.extend(tokenizer.encode(line))
        if (idx + 1) % 2500 == 0 or (idx + 1) == total_lines:
            logger.info(f"Tokenized {idx + 1:,}/{total_lines:,} lines ({len(token_ids):,} tokens)...")

    dataset = TextDataset(np.array(token_ids, dtype=np.int32), val_ratio=0.1)
    model = ParagAIModel(model_config)

    trainer = Trainer(model, dataset, train_config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
