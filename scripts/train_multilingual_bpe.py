"""Train the IndicBPETokenizer on multilingual text corpora.

Usage
-----
    python scripts/train_multilingual_bpe.py \\
        --input "data/raw/**/*.txt" \\
        --vocab_size 4096 \\
        --output checkpoints/tokenizer_indic.json \\
        --sample_mb 20

The script:
1. Globs all matching .txt files.
2. Samples up to ``--sample_mb`` MB of text (round-robin across files).
3. Reports total characters read and unique Unicode scripts detected.
4. Trains an ``IndicBPETokenizer`` on the sampled text.
5. Runs ``get_coverage_stats`` on a 1000-char held-out sample per script.
6. Saves the tokenizer JSON to ``--output``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import unicodedata
from typing import Dict, List, Tuple

# Allow running from project root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paragai.tokenizer.indic_bpe import (
    VALID_VOCAB_SIZES,
    IndicBPETokenizer,
    _SCRIPT_RANGES,
    _detect_scripts,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _rprint(msg: str) -> None:
    if HAS_RICH:
        console.print(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Text sampling
# ---------------------------------------------------------------------------

def _collect_files(pattern: str) -> List[str]:
    """Expand *pattern* (glob) and return a sorted list of .txt file paths."""
    if os.path.isfile(pattern):
        return [pattern]
    matches = sorted(glob.glob(pattern, recursive=True))
    return [p for p in matches if os.path.isfile(p)]


def _sample_text(files: List[str], max_bytes: int) -> str:
    """Round-robin sample up to *max_bytes* bytes from *files*."""
    if not files:
        return ""

    chunks: List[str] = []
    total = 0
    # Interleave files for language balance
    file_handles = [open(f, "r", encoding="utf-8", errors="replace") for f in files]
    try:
        while total < max_bytes:
            exhausted = 0
            for fh in file_handles:
                if total >= max_bytes:
                    break
                chunk = fh.read(65_536)  # 64 KB at a time
                if not chunk:
                    exhausted += 1
                    continue
                chunks.append(chunk)
                total += len(chunk.encode("utf-8"))
            if exhausted == len(file_handles):
                break
    finally:
        for fh in file_handles:
            fh.close()

    return "".join(chunks)


# ---------------------------------------------------------------------------
# Script detection helpers
# ---------------------------------------------------------------------------

def _build_script_samples(text: str, sample_len: int = 1000) -> Dict[str, str]:
    """Return a mapping of script -> held-out sample of *sample_len* chars."""
    buckets: Dict[str, List[str]] = {s: [] for s in _SCRIPT_RANGES}
    for ch in text:
        for script, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= ord(ch) <= hi:
                if len(buckets[script]) < sample_len:
                    buckets[script].append(ch)
    return {
        script: "".join(chars)
        for script, chars in buckets.items()
        if chars
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IndicBPETokenizer on multilingual corpora",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/**/*.txt",
        help="Glob pattern or single file path for training text files",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=4096,
        choices=list(VALID_VOCAB_SIZES),
        help="Target BPE vocabulary size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/tokenizer_indic.json",
        help="Destination path for the saved tokenizer JSON",
    )
    parser.add_argument(
        "--sample_mb",
        type=float,
        default=20.0,
        help="Approximate megabytes of text to sample for training",
    )
    args = parser.parse_args()

    max_bytes = int(args.sample_mb * 1024 * 1024)

    # ---- File discovery ---------------------------------------------------
    files = _collect_files(args.input)
    if not files:
        _rprint(
            f"[bold red]No files found matching:[/bold red] {args.input}"
            if HAS_RICH
            else f"ERROR: No files found matching: {args.input}"
        )
        sys.exit(1)

    _rprint(
        f"[bold cyan]Found {len(files)} file(s)[/bold cyan]" if HAS_RICH
        else f"Found {len(files)} file(s)"
    )

    # ---- Sampling ---------------------------------------------------------
    _rprint(f"Sampling up to {args.sample_mb:.1f} MB of text…")
    t_sample = time.time()
    text = _sample_text(files, max_bytes)
    sample_elapsed = time.time() - t_sample

    total_chars = len(text)
    detected_scripts = _detect_scripts(text)

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold cyan]IndicBPETokenizer Training[/bold cyan]\n"
                f"Input pattern : [yellow]{args.input}[/yellow]\n"
                f"Files matched : [green]{len(files)}[/green]\n"
                f"Total chars   : [green]{total_chars:,}[/green]\n"
                f"Scripts found : [magenta]{detected_scripts or ['(none recognised)']}[/magenta]\n"
                f"Vocab size    : [cyan]{args.vocab_size:,}[/cyan]\n"
                f"Output path   : [yellow]{args.output}[/yellow]",
                title="ParagAI Multilingual Tokenizer",
                border_style="cyan",
            )
        )
    else:
        print("=" * 60)
        print("IndicBPETokenizer Training")
        print(f"  Input pattern : {args.input}")
        print(f"  Files matched : {len(files)}")
        print(f"  Total chars   : {total_chars:,}")
        print(f"  Scripts found : {detected_scripts or ['(none recognised)']}")
        print(f"  Vocab size    : {args.vocab_size:,}")
        print(f"  Output path   : {args.output}")
        print("=" * 60)

    if not text.strip():
        _rprint("[bold red]Sampled text is empty — aborting.[/bold red]" if HAS_RICH else "ERROR: Sampled text is empty.")
        sys.exit(1)

    # Build held-out samples (before training so they are not affected)
    script_samples = _build_script_samples(text)

    # ---- Training ---------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    tokenizer = IndicBPETokenizer(vocab_size=args.vocab_size)
    _rprint(
        f"\n[bold green]Starting BPE training…[/bold green]"
        if HAS_RICH else "\nStarting BPE training…"
    )
    t_train = time.time()
    tokenizer.train(text, verbose=True)
    train_elapsed = time.time() - t_train

    _rprint(
        f"\n[bold green]Training complete in {train_elapsed:.2f}s.[/bold green] "
        f"Final vocab size: [cyan]{tokenizer.vocab_size:,}[/cyan]"
        if HAS_RICH
        else f"\nTraining complete in {train_elapsed:.2f}s. Final vocab size: {tokenizer.vocab_size:,}"
    )

    # ---- Coverage stats per script ----------------------------------------
    if HAS_RICH:
        table = Table(title="Script Coverage Stats (held-out 1000-char samples)")
        table.add_column("Script", style="cyan")
        table.add_column("Chars", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Unique Tokens", justify="right")
        table.add_column("Avg tok/word", justify="right")
        table.add_column("Compression", justify="right")

    else:
        print("\nScript Coverage Stats (held-out 1000-char samples)")
        print("-" * 70)
        print(f"{'Script':<14} {'Chars':>6} {'Tokens':>7} {'Unique':>7} {'Avg t/w':>8} {'Ratio':>7}")
        print("-" * 70)

    for script, sample in script_samples.items():
        stats = tokenizer.get_coverage_stats(sample)
        if HAS_RICH:
            table.add_row(
                script,
                str(stats["total_chars"]),
                str(stats["total_tokens"]),
                str(stats["unique_tokens"]),
                f"{stats['avg_tokens_per_word']:.3f}",
                f"{stats['compression_ratio']:.3f}",
            )
        else:
            print(
                f"{script:<14} {stats['total_chars']:>6} {stats['total_tokens']:>7} "
                f"{stats['unique_tokens']:>7} {stats['avg_tokens_per_word']:>8.3f} "
                f"{stats['compression_ratio']:>7.3f}"
            )

    if HAS_RICH:
        console.print(table)
    else:
        print("-" * 70)

    # ---- Quick round-trip sanity check ------------------------------------
    test_texts = {
        "English": "Hello, how are you today?",
        "Bengali": "হ্যালো বাংলা কেমন আছেন",
        "Hindi": "नमस्ते हिंदी में आपका स्वागत है",
        "Tamil": "வணக்கம் தமிழ் நடுவர்",
        "Telugu": "నమస్కారం తెలుగు పరీక్ష",
    }
    _rprint("\n[bold]Round-trip sanity checks:[/bold]" if HAS_RICH else "\nRound-trip sanity checks:")
    all_pass = True
    for lang, sample in test_texts.items():
        enc = tokenizer.encode(sample)
        dec = tokenizer.decode(enc)
        ok = dec == sample
        all_pass = all_pass and ok
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        _rprint(
            f"  {status} {lang}: {len(enc)} tokens"
            if HAS_RICH else f"  {'✓' if ok else '✗'} {lang}: {len(enc)} tokens"
        )

    # ---- Save -------------------------------------------------------------
    tokenizer.save(args.output)
    _rprint(
        f"\n[bold green]Tokenizer saved to:[/bold green] [yellow]{args.output}[/yellow]"
        if HAS_RICH else f"\nTokenizer saved to: {args.output}"
    )

    if not all_pass:
        _rprint(
            "[bold yellow]Warning:[/bold yellow] Some round-trip checks failed — "
            "check for text with unseen byte sequences."
            if HAS_RICH
            else "Warning: Some round-trip checks failed."
        )


if __name__ == "__main__":
    main()
