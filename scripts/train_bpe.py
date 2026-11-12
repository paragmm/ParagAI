"""Train BPETokenizer on processed corpora and save to checkpoints/."""

import os
import argparse
import time
from rich.console import Console
from rich.panel import Panel
from paragai.tokenizer.bpe_tokenizer import BPETokenizer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train BPETokenizer on text corpus")
    parser.add_argument("--corpus", type=str, default="data/processed/combined_corpus.txt", help="Path to text corpus")
    parser.add_argument("--vocab_size", type=int, default=1024, help="Target vocabulary size (e.g. 1024, 2048, 4096)")
    parser.add_argument("--output", type=str, default="checkpoints/tokenizer_bpe.json", help="Output path for tokenizer.json")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        console.print(f"[bold red]Corpus not found at {args.corpus}[/bold red]")
        console.print("Run: [cyan]python scripts/prepare_data.py --no_tags[/cyan] first.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    console.print(Panel(
        f"[bold cyan]Training BPETokenizer (Step 04)[/bold cyan]\n"
        f"Corpus: [yellow]{args.corpus}[/yellow]\n"
        f"Target Vocab Size: [green]{args.vocab_size:,}[/green]\n"
        f"Output Destination: [magenta]{args.output}[/magenta]",
        title="ParagAI Tokenizer Builder",
        border_style="cyan"
    ))

    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()

    console.print(f"Loaded corpus: [bold green]{len(text):,}[/bold green] characters.")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    t0 = time.time()
    tokenizer.train(text, verbose=False)
    elapsed = time.time() - t0

    tokenizer.save(args.output)
    console.print(f"[bold green]Training complete in {elapsed:.2f} seconds![/bold green]")
    console.print(f"Final Vocab Size: [bold cyan]{tokenizer.vocab_size:,}[/bold cyan]")

    # Run quick sanity check
    sample_text = "<|im_start|>user\nHello! How are you?<|im_end|>\n<|im_start|>assistant\nআমি ভালো আছি।<|im_end|>"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    exact = sample_text == decoded

    console.print(f"Sanity Check: {len(sample_text)} characters -> {len(encoded)} subword tokens.")
    console.print(f"Lossless Roundtrip: [{'bold green' if exact else 'bold red'}]{exact}[/{'bold green' if exact else 'bold red'}]")


if __name__ == "__main__":
    main()

