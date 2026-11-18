"""Data preparation pipeline: Strictly separates pre-training corpora from SFT corpora.

Modes:
- Pre-train: Compiles continuous English stories (TinyStories, technology, prose) without tags.
- SFT: Compiles 100% formatted chat dialogues (<|system|>...<|user|>...<|assistant|>...<|end|>).
- All: Compiles pre-train, SFT, and a combined corpus for full training pipelines.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rich.console import Console
from rich.panel import Panel

console = Console()


def prepare_pretrain_corpus(
    raw_dir: str = "data/raw",
    output_file: str = "data/processed/pretrain_corpus.txt",
    max_mb: float = 15.0,
) -> str:
    """Extract clean English text for Stage 1 pre-training without prompt tags."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Automatically find all TinyStories files (colab, local, or chunked)
    story_files = sorted(glob.glob(os.path.join(raw_dir, "stories", "tinystories*.txt")))
    other_files = [
        os.path.join(raw_dir, "stories", "english_story.txt"),
        os.path.join(raw_dir, "technology", "ai_transformer.txt"),
        os.path.join(raw_dir, "technology", "python_intro.txt"),
    ]
    target_files = story_files + [f for f in other_files if f not in story_files]

    total_bytes = 0
    max_bytes = int(max_mb * 1024 * 1024)
    file_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for fpath in target_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8", errors="replace") as in_f:
                content = in_f.read().strip()
                if not content:
                    continue
                # Clean out any old bracketed categories or legacy Prompt: tags
                lines = [
                    line
                    for line in content.splitlines()
                    if not line.startswith(("[CATEGORY:", "Prompt:", "User:", "ParagAI:"))
                ]
                cleaned = "\n".join(lines).strip()
                out_f.write(cleaned + "\n\n")
                total_bytes += len(cleaned.encode("utf-8"))
                file_count += 1
                if total_bytes >= max_bytes:
                    break

    console.print(
        Panel(
            f"[bold green]Stage 1 Pre-training Corpus Ready[/bold green]\n"
            f"Sources: {file_count} files merged\n"
            f"Size: {total_bytes:,} bytes\n"
            f"Output: [cyan]{output_file}[/cyan]",
            title="Pre-train Data",
            border_style="green",
        )
    )
    return output_file


def prepare_sft_corpus(output_file: str = "data/processed/sft_corpus.txt") -> str:
    """Generate Stage 2 SFT dialogues in standard chat template."""
    from scripts.build_sft_dataset import generate_sft_dataset

    generate_sft_dataset(output_file=output_file)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="ParagAI Two-Stage Data Preparation")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "pretrain", "sft"],
        default="all",
        help="Which data stage to prepare",
    )
    args = parser.parse_args()

    pretrain_path = "data/processed/pretrain_corpus.txt"
    sft_path = "data/processed/sft_corpus.txt"

    if args.stage in ("all", "pretrain"):
        pretrain_path = prepare_pretrain_corpus()
    if args.stage in ("all", "sft"):
        sft_path = prepare_sft_corpus()

    # Generate combined_corpus.txt for backwards compatibility with legacy train configs
    if args.stage == "all":
        combined_path = "data/processed/combined_corpus.txt"
        with open(combined_path, "w", encoding="utf-8") as out:
            if os.path.exists(pretrain_path):
                with open(pretrain_path, "r", encoding="utf-8") as f:
                    out.write(f.read().strip() + "\n\n")
            if os.path.exists(sft_path):
                with open(sft_path, "r", encoding="utf-8") as f:
                    out.write(f.read().strip() + "\n\n")
        console.print(f"[bold green]Created combined corpus for full training:[/bold green] [cyan]{combined_path}[/cyan]")

    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
