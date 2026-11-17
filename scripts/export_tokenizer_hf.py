"""Export ParagAI Tokenizer to Hugging Face and vLLM-compatible format.

Usage:
    python scripts/export_tokenizer_hf.py \\
        --input checkpoints/tokenizer_indic.json \\
        --output_dir checkpoints/export_hf/
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paragai.tokenizer.fast_indic_bpe import FastIndicBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ParagAI tokenizer to Hugging Face tokenizer.json format for vLLM & Ollama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="checkpoints/tokenizer_indic.json",
        help="Path to ParagAI tokenizer JSON checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/export_hf",
        help="Destination directory for Hugging Face tokenizer artifacts",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input tokenizer checkpoint not found: {args.input}")
        sys.exit(1)

    print(f"[export_tokenizer_hf] Loading tokenizer: {args.input}")
    tokenizer = FastIndicBPETokenizer.load(args.input)
    print(f"[export_tokenizer_hf] Loaded vocab size: {tokenizer.vocab_size:,}")

    os.makedirs(args.output_dir, exist_ok=True)
    hf_path = tokenizer.export_huggingface(args.output_dir)
    print(f"[export_tokenizer_hf] Exported tokenizer.json → {hf_path}")
    print(f"[export_tokenizer_hf] Exported tokenizer_config.json → {os.path.join(args.output_dir, 'tokenizer_config.json')}")
    print("\n✓ Ready for vLLM serving:")
    print(f"  vllm serve {args.output_dir} --tokenizer {args.output_dir}")


if __name__ == "__main__":
    main()
