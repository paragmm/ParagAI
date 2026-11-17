"""Command-line script to train ParagAI models with Two-Stage Pipeline (Pre-training + SFT)."""

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
from paragai.training.dataset import SFTDataset, TextDataset
from paragai.training.trainer import Trainer
from paragai.utils.logger import get_logger
from paragai.utils.seed import set_seed

logger = get_logger("TrainScript")


def main():
    parser = argparse.ArgumentParser(description="Train a ParagAI language model")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "sft"],
        default="pretrain",
        help="Training stage: 'pretrain' (TinyStories, LR ~4e-4) or 'sft' (Chat/Tools, LR ~3e-5)",
    )
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
        "--data", type=str, default=None, help="Path to training text file (auto-detected by stage if omitted)"
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="checkpoints/pretrained_model.pt",
        help="Base checkpoint to load for Stage 2 SFT fine-tuning",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["char", "bpe", "indic_bpe"],
        default="indic_bpe",
        help="Tokenizer type",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to pre-trained tokenizer JSON (e.g. checkpoints/tokenizer_indic.json)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override learning rate for this stage"
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

    # Stage-aware default learning rates
    if args.learning_rate is not None:
        train_yaml["learning_rate"] = args.learning_rate
    elif args.stage == "sft":
        train_yaml["learning_rate"] = 3e-5
        logger.info("Stage 2 SFT mode: defaulted learning rate to 3e-5")
    else:
        # Default pretrain lr: 4e-4 unless specified in yaml
        train_yaml["learning_rate"] = train_yaml.get("learning_rate", 4e-4)

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

    # Resolve training data file by stage if not specified
    if args.data is None:
        if args.stage == "pretrain":
            args.data = "data/processed/pretrain_corpus.txt"
        else:
            args.data = "data/processed/sft_corpus.txt"

    # Auto-generate dataset file if missing
    if not os.path.exists(args.data):
        from scripts.prepare_data import prepare_pretrain_corpus, prepare_sft_corpus

        if args.stage == "pretrain":
            logger.info(f"Pre-train data not found. Compiling {args.data}...")
            prepare_pretrain_corpus(output_file=args.data)
        else:
            logger.info(f"SFT data not found. Compiling {args.data}...")
            prepare_sft_corpus(output_file=args.data)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    # Resolve tokenizer
    tok_dir = os.path.join(train_config.checkpoint_dir)
    os.makedirs(tok_dir, exist_ok=True)
    target_tok_path = os.path.join(tok_dir, "tokenizer.json")

    tok_candidate = args.tokenizer_path
    if not tok_candidate:
        indic_default = os.path.join(tok_dir, "tokenizer_indic.json")
        if os.path.exists(indic_default):
            tok_candidate = indic_default
        elif os.path.exists(target_tok_path):
            tok_candidate = target_tok_path

    tokenizer = None
    if tok_candidate and os.path.exists(tok_candidate):
        logger.info(f"Loading pre-trained tokenizer from {tok_candidate}...")
        with open(tok_candidate, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if (
            data.get("tokenizer_class") == "fast_indic_bpe"
            or "version" in data
            or "model" in data
        ):
            from paragai.tokenizer.fast_indic_bpe import FastIndicBPETokenizer
            tokenizer = FastIndicBPETokenizer.load(tok_candidate)
        elif data.get("tokenizer_class") == "indic_bpe" or args.tokenizer_type == "indic_bpe":
            tokenizer = IndicBPETokenizer.load(tok_candidate)
        elif data.get("type") == "bpe":
            tokenizer = BPETokenizer.load(tok_candidate)
        else:
            tokenizer = CharTokenizer.load(tok_candidate)

        if (
            os.path.abspath(tok_candidate) != os.path.abspath(target_tok_path)
            and data.get("tokenizer_class") != "fast_indic_bpe"
        ):
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

    # Ensure special tokens are registered
    special_needed = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    if hasattr(tokenizer, "add_special_tokens"):
        tokenizer.add_special_tokens(special_needed)

    # Update vocab_size in model config to match tokenizer
    model_yaml["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(**model_yaml)
    model = ParagAIModel(model_config)

    # Prepare dataset according to stage
    if args.stage == "sft":
        logger.info("Initializing SFTDataset with assistant loss masking (-100)...")
        dataset = SFTDataset.from_text(text, tokenizer=tokenizer, val_ratio=0.1)
    else:
        logger.info("Tokenizing Stage 1 pre-training corpus line-by-line...")
        lines = text.splitlines(keepends=True)
        token_ids: list[int] = []
        total_lines = len(lines)
        for idx, line in enumerate(lines):
            if line:
                token_ids.extend(tokenizer.encode(line))
            if (idx + 1) % 5000 == 0 or (idx + 1) == total_lines:
                logger.info(f"Tokenized {idx + 1:,}/{total_lines:,} lines ({len(token_ids):,} tokens)...")
        dataset = TextDataset(np.array(token_ids, dtype=np.int32), val_ratio=0.1)

    trainer = Trainer(model, dataset, train_config, tokenizer=tokenizer)

    # Load weights for SFT or resume
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume, load_optimizer=True)
    elif args.stage == "sft":
        # In SFT stage, attempt to load the pre-trained base model
        base_ckpt = args.pretrained_checkpoint
        if not os.path.exists(base_ckpt):
            # Fall back to best_model.pt if available
            alt_ckpt = os.path.join(train_config.checkpoint_dir, "best_model.pt")
            if os.path.exists(alt_ckpt):
                base_ckpt = alt_ckpt

        if os.path.exists(base_ckpt):
            logger.info(f"Stage 2 SFT: Loading pre-trained base checkpoint from {base_ckpt}...")
            trainer.load_checkpoint(base_ckpt, load_optimizer=False)
            # Ensure model embeddings match current tokenizer vocab size
            if model.config.vocab_size != tokenizer.vocab_size:
                logger.info(f"Resizing model embeddings to match tokenizer vocab ({tokenizer.vocab_size})...")
                model.resize_token_embeddings(tokenizer.vocab_size)
        else:
            logger.warning(
                f"No pre-trained checkpoint found at {base_ckpt}. Training SFT from scratch."
            )

    logger.info(f"Launching ParagAI Training [Stage: {args.stage.upper()}]...")
    trainer.train()

    # Save final stage checkpoints
    stage_ckpt_name = "sft_model.pt" if args.stage == "sft" else "pretrained_model.pt"
    trainer.save_checkpoint(os.path.join(train_config.checkpoint_dir, stage_ckpt_name))
    trainer.save_checkpoint(os.path.join(train_config.checkpoint_dir, "best_model.pt"))
    logger.info(f"Stage {args.stage.upper()} completed. Checkpoints saved to {train_config.checkpoint_dir}")


if __name__ == "__main__":
    main()
