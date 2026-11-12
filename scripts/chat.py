"""Interactive terminal chat interface for ParagAI models."""

import os
import sys

# Ensure src is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import argparse
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from paragai.config import ModelConfig
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.bpe_tokenizer import BPETokenizer
from paragai.tokenizer.char_tokenizer import CharTokenizer
from paragai.inference.generate import Generator

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Chat interactively with a trained ParagAI model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--max_tokens", type=int, default=150, help="Maximum new tokens to generate")
    parser.add_argument("--chatml", action="store_true", default=True, help="Use ChatML formatting for prompt")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        console.print(f"[bold red]Error:[/bold red] Checkpoint not found at {args.checkpoint}")
        console.print("Train a model first using: [cyan]python scripts/train.py[/cyan]")
        return

    # Load tokenizer
    if os.path.exists(args.tokenizer):
        try:
            tokenizer = BPETokenizer.load(args.tokenizer)
        except Exception:
            tokenizer = CharTokenizer.load(args.tokenizer)
    else:
        console.print("[yellow]Tokenizer not found, falling back to basic ASCII tokenizer...[/yellow]")
        tokenizer = CharTokenizer([chr(i) for i in range(32, 127)])

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = ParagAIModel(model_config)
    
    # Filter out legacy buffers (e.g. attn.mask from pre-SDPA checkpoints)
    state_dict = {
        k: v for k, v in checkpoint["model_state_dict"].items()
        if not k.endswith(".attn.mask")
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    generator = Generator(model, tokenizer, device=device)

    console.print(Panel(
        f"[bold green]ParagAI Interactive Terminal[/bold green]\n"
        f"Model: {model_config.name} | Parameters: {model.count_parameters():,} | Device: {device}\n"
        f"Type [yellow]exit[/yellow] or [yellow]quit[/yellow] to end session.",
        title="ParagAI",
        border_style="green"
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Prompt[/bold cyan]")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue

            # Format with ChatML if requested
            if args.chatml:
                prompt_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt_text = user_input

            console.print("[bold green]ParagAI:[/bold green] ", end="")
            accumulated = ""
            for token in generator.stream_generate(
                prompt=prompt_text,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                accumulated += token
                # Check for stop markers
                if "<|im_end|>" in accumulated:
                    # Print any text before the marker and stop
                    before = token.split("<|im_end|>")[0]
                    if before:
                        console.print(before, end="", highlight=False)
                    break
                if "\nUser:" in accumulated or "\n<|im_start|>" in accumulated:
                    break
                console.print(token, end="", highlight=False)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted.[/yellow]")
            break


if __name__ == "__main__":
    main()

