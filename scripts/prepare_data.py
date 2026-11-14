"""Data preparation script: Ingests multiple files, handles categories, and prepares training corpora."""

import argparse
import os

from rich.console import Console
from rich.table import Table

console = Console()


def scan_dataset_directory(raw_dir: str = "data/raw") -> dict:
    """Scan raw directory and group .txt files by category folder."""
    categories = {}
    for root, dirs, files in os.walk(raw_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        if not txt_files:
            continue
        # Use subfolder name as category name
        rel_path = os.path.relpath(root, raw_dir)
        category = "general" if rel_path == "." else rel_path.replace("\\", "/")
        categories[category] = [os.path.join(root, f) for f in txt_files]
    return categories


def merge_categorized_files(
    raw_dir: str = "data/raw",
    output_file: str = "data/processed/combined_corpus.txt",
    selected_category: str | None = None,
    include_tags: bool = True,
) -> None:
    """Merge multiple text files with category headers and document separators."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    categories = scan_dataset_directory(raw_dir)

    if not categories:
        console.print(f"[bold red]No .txt files found in {raw_dir}[/bold red]")
        return

    table = Table(title="Discovered Categories & Files", border_style="cyan")
    table.add_column("Category", style="bold magenta")
    table.add_column("File Count", justify="right", style="green")
    table.add_column("Files", style="dim")

    total_files = 0
    for cat, files in sorted(categories.items()):
        total_files += len(files)
        file_names = ", ".join([os.path.basename(f) for f in files[:3]])
        if len(files) > 3:
            file_names += f" (+{len(files) - 3} more)"
        table.add_row(cat, str(len(files)), file_names)

    console.print(table)

    total_characters = 0
    documents_merged = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for cat, file_paths in categories.items():
            if selected_category and cat != selected_category:
                continue

            for fpath in file_paths:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as in_f:
                        content = in_f.read().strip()

                    if not content:
                        continue

                    # Write category tag header if requested
                    if include_tags:
                        header = f"<|startoftext|>\n[CATEGORY: {cat}]\n"
                    else:
                        header = "<|startoftext|>\n"

                    footer = "\n<|endoftext|>\n\n"

                    out_f.write(header + content + footer)
                    total_characters += len(content)
                    documents_merged += 1
                except (OSError, UnicodeDecodeError) as e:
                    console.print(f"[red]Error reading {fpath}: {e}[/red]")

    console.print(
        Panel_msg(
            f"[bold green]Successfully merged {documents_merged} files![/bold green]\n"
            f"Total Characters: {total_characters:,}\n"
            f"Output file: [cyan]{output_file}[/cyan]"
        )
    )


def Panel_msg(text: str):
    from rich.panel import Panel

    return Panel(text, title="Data Pipeline", border_style="green")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and categorize multiple text datasets for ParagAI"
    )
    parser.add_argument(
        "--raw_dir", type=str, default="data/raw", help="Path to raw data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/combined_corpus.txt",
        help="Combined output file",
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Specific category to merge (e.g. 'stories')"
    )
    parser.add_argument(
        "--no_tags", action="store_true", help="Do not inject [CATEGORY: ...] headers"
    )
    args = parser.parse_args()

    merge_categorized_files(
        raw_dir=args.raw_dir,
        output_file=args.output,
        selected_category=args.category,
        include_tags=not args.no_tags,
    )


if __name__ == "__main__":
    main()
