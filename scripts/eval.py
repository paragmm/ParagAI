"""Comprehensive evaluation suite for ParagAI.

Tasks
-----
perplexity  — Cross-entropy perplexity on a held-out text file.
multiling   — 10 prompts each in English / Bengali / Hindi; % non-empty responses.
tools       — 5 tool-use prompts; % where output contains valid {"action":…,"command":…} JSON.
story       — 5 story prompts; % where the response starts with a narrative word.

Usage
-----
python scripts/eval.py \\
    --checkpoint checkpoints/best_model.pt \\
    --tokenizer  checkpoints/tokenizer.json \\
    --tasks      perplexity,multiling,tools,story \\
    --val_data   data/processed/val.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Lazy imports (model / tokenizer may not be available in every environment)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str,
) -> Tuple[Any, Any, Dict]:
    """Load ParagAI model and tokenizer from checkpoint paths.

    Returns:
        Tuple of ``(model, tokenizer, model_config)``.
    """
    # ---- tokenizer -----------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    try:
        with open(tokenizer_path, "r", encoding="utf-8") as fh:
            tok_data = json.load(fh)
        if tok_data.get("tokenizer_class") == "indic_bpe":
            from paragai.tokenizer.indic_bpe import IndicBPETokenizer  # type: ignore[import]
            tok = IndicBPETokenizer.load(tokenizer_path)
        elif tok_data.get("type") == "bpe":
            from paragai.tokenizer.bpe_tokenizer import BPETokenizer  # type: ignore[import]
            tok = BPETokenizer.load(tokenizer_path)
        else:
            from paragai.tokenizer.char_tokenizer import CharTokenizer  # type: ignore[import]
            tok = CharTokenizer.load(tokenizer_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {exc}") from exc

    # ---- checkpoint ----------------------------------------------------------
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {exc}") from exc

    model_config: Dict = ckpt["model_config"]
    state_dict: Dict = ckpt["model_state_dict"]

    # ---- model ---------------------------------------------------------------
    try:
        from paragai.model.transformer import ParagAIModel  # type: ignore[import]
        from paragai.config import ModelConfig  # type: ignore[import]

        cfg = ModelConfig(**model_config)
        model = ParagAIModel(cfg)
    except Exception as exc:
        raise RuntimeError(f"Failed to build model: {exc}") from exc

    # strip attn mask (causal buffer) so strict=False doesn't complain
    state_dict = {k: v for k, v in state_dict.items() if "attn.mask" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, tok, model_config


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _generate(
    model: Any,
    tok: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_k: int = 40,
    device: str = "cpu",
) -> str:
    """Run greedy / top-k generation and decode the output."""
    ids: List[int] = tok.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    max_seq = getattr(getattr(model, "config", None), "max_seq_len", getattr(model, "max_seq_len", 256))

    for _ in range(max_new_tokens):
        # truncate to context window
        idx_cond = input_ids[:, -max_seq:]
        out = model(idx_cond)
        logits = out[0] if isinstance(out, tuple) else out
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < vals[:, [-1]]] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

        # stop on eos
        eos_id = getattr(tok, "eos_id", None) or getattr(tok, "_eos_id", None)
        if eos_id is not None and next_id.item() == eos_id:
            break

    generated_ids = input_ids[0, len(ids):].tolist()
    return tok.decode(generated_ids)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def task_perplexity(
    model: Any,
    tok: Any,
    val_data_path: Optional[str],
    device: str,
) -> Dict[str, Any]:
    """Compute perplexity on the validation text.

    Args:
        model:         Loaded ParagAI Transformer.
        tok:           Loaded tokenizer.
        val_data_path: Path to plain-text validation file.
        device:        Torch device string.

    Returns:
        Result dict with keys ``ppl``, ``nll``, ``n_tokens``.
    """
    if not val_data_path or not Path(val_data_path).exists():
        return {"error": f"val_data not found: {val_data_path}", "ppl": None}

    text = Path(val_data_path).read_text(encoding="utf-8")
    ids = tok.encode(text)
    if len(ids) < 2:
        return {"error": "val_data too short after tokenisation", "ppl": None}

    max_seq = getattr(getattr(model, "config", None), "max_seq_len", getattr(model, "max_seq_len", 256))
    stride = max_seq // 2
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)

    total_nll = 0.0
    n_tokens = 0
    model.eval()

    with torch.inference_mode():
        for i in range(0, len(ids_t) - 1, stride):
            chunk = ids_t[i : i + max_seq + 1]
            if len(chunk) < 2:
                break
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:]
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits_flat = logits[0].float()  # (T, V)
            loss = torch.nn.functional.cross_entropy(
                logits_flat, y, reduction="sum"
            )
            total_nll += loss.item()
            n_tokens += len(y)
            if n_tokens >= 50_000:  # cap for speed
                break

    nll = total_nll / n_tokens if n_tokens > 0 else float("nan")
    ppl = math.exp(min(nll, 20))  # clip to avoid inf display
    return {"ppl": round(ppl, 4), "nll": round(nll, 6), "n_tokens": n_tokens}


def task_multiling(
    model: Any,
    tok: Any,
    device: str,
) -> Dict[str, Any]:
    """Probe model with 10 prompts per language and report % non-empty responses.

    Languages: English, Bengali, Hindi.

    Returns:
        Result dict with per-language and overall non-empty percentages.
    """
    prompts: Dict[str, List[str]] = {
        "english": [
            "Hello, how are you?",
            "What is the capital of France?",
            "Tell me about artificial intelligence.",
            "Translate 'good morning' to French.",
            "What is 2 + 2?",
            "Who invented the telephone?",
            "Describe the water cycle.",
            "Write a sentence about the sun.",
            "What is machine learning?",
            "Explain gravity in simple terms.",
        ],
        "bengali": [
            "আপনি কেমন আছেন?",
            "বাংলাদেশের রাজধানী কী?",
            "কৃত্রিম বুদ্ধিমত্তা সম্পর্কে বলুন।",
            "সূর্য কী?",
            "দুই যোগ দুই কত?",
            "মহাকর্ষ কী?",
            "বাংলা ভাষার ইতিহাস বলুন।",
            "একটি গল্প বলুন।",
            "পানি চক্র কী?",
            "ভালো সকাল।",
        ],
        "hindi": [
            "आप कैसे हैं?",
            "भारत की राजधानी क्या है?",
            "कृत्रिम बुद्धिमत्ता के बारे में बताएं।",
            "सूरज क्या है?",
            "दो जमा दो कितना होता है?",
            "गुरुत्वाकर्षण क्या है?",
            "हिंदी भाषा का इतिहास बताएं।",
            "एक कहानी सुनाएं।",
            "जल चक्र क्या है?",
            "शुभ प्रभात।",
        ],
    }

    lang_results: Dict[str, Dict[str, Any]] = {}
    total_non_empty = 0
    total_prompts = 0

    for lang, lang_prompts in prompts.items():
        non_empty = 0
        for prompt in lang_prompts:
            try:
                chatml = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                response = _generate(model, tok, chatml, max_new_tokens=64, device=device)
                if response.strip():
                    non_empty += 1
            except Exception:
                pass  # crashed = empty
        pct = round(100.0 * non_empty / len(lang_prompts), 1)
        lang_results[lang] = {"non_empty": non_empty, "total": len(lang_prompts), "pct": pct}
        total_non_empty += non_empty
        total_prompts += len(lang_prompts)

    overall_pct = round(100.0 * total_non_empty / total_prompts, 1) if total_prompts else 0.0
    return {"per_language": lang_results, "overall_pct": overall_pct}


def task_tools(
    model: Any,
    tok: Any,
    device: str,
) -> Dict[str, Any]:
    """Test tool-adherence: check if output contains valid JSON with action+command.

    Returns:
        Result dict with counts and percentage of valid JSON tool responses.
    """
    tool_prompts = [
        "Count lines in src/",
        "List all Python files in the project.",
        "Show me the directory structure.",
        "Find TODO comments in the codebase.",
        "Run the test suite and report results.",
    ]

    valid_json_count = 0
    details: List[Dict[str, Any]] = []

    for prompt in tool_prompts:
        chatml = (
            "<|im_start|>system\nYou are a tool-use assistant. "
            'Respond with JSON: {"action": "...", "command": "..."}<|im_end|>\n'
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        try:
            response = _generate(model, tok, chatml, max_new_tokens=128, device=device)
        except Exception as exc:
            details.append({"prompt": prompt, "valid": False, "error": str(exc)})
            continue

        is_valid = _has_tool_json(response)
        if is_valid:
            valid_json_count += 1
        details.append({"prompt": prompt, "valid": is_valid, "response_preview": response[:80]})

    pct = round(100.0 * valid_json_count / len(tool_prompts), 1)
    return {"valid_json": valid_json_count, "total": len(tool_prompts), "pct": pct, "details": details}


def _has_tool_json(text: str) -> bool:
    """Return True if *text* contains a JSON object with 'action' and 'command'."""
    # Try to find {...} blocks
    for match in re.finditer(r"\{[^}]+\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if "action" in obj and "command" in obj:
                return True
        except json.JSONDecodeError:
            pass
    return False


def task_story(
    model: Any,
    tok: Any,
    device: str,
) -> Dict[str, Any]:
    """Probe story generation: check if response begins with a narrative word.

    Narrative start words (English + Bengali):
    ``Once, There, In, A, The, এক, একদা``

    Returns:
        Result dict with counts and percentage of narrative starts.
    """
    story_prompts = [
        "Write a short story for me.",
        "Tell me a story.",
        "Create a bedtime story.",
        "একটি ছোট গল্প লিখুন।",
        "मुझे एक कहानी सुनाओ।",
    ]

    narrative_words = {
        "once", "there", "in", "a", "the",
        "এক", "একদা",
    }

    narrative_count = 0
    details: List[Dict[str, Any]] = []

    for prompt in story_prompts:
        chatml = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        try:
            response = _generate(model, tok, chatml, max_new_tokens=128, device=device)
        except Exception as exc:
            details.append({"prompt": prompt, "narrative_start": False, "error": str(exc)})
            continue

        first_word = response.strip().split()[0].lower().rstrip(".,!?") if response.strip() else ""
        is_narrative = first_word in narrative_words
        if is_narrative:
            narrative_count += 1
        details.append({
            "prompt": prompt,
            "narrative_start": is_narrative,
            "first_word": first_word,
            "response_preview": response[:80],
        })

    pct = round(100.0 * narrative_count / len(story_prompts), 1)
    return {"narrative_starts": narrative_count, "total": len(story_prompts), "pct": pct, "details": details}


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_markdown_table(results: Dict[str, Any]) -> str:
    """Render evaluation results as a GitHub-flavoured Markdown table."""
    rows: List[Tuple[str, str, str]] = []

    if "perplexity" in results:
        r = results["perplexity"]
        if r.get("ppl") is not None:
            rows.append(("Perplexity", f"{r['ppl']:.2f}", f"n_tokens={r.get('n_tokens', '?')}"))
        else:
            rows.append(("Perplexity", "N/A", r.get("error", "")))

    if "multiling" in results:
        r = results["multiling"]
        overall = r.get("overall_pct", "N/A")
        per_lang = r.get("per_language", {})
        detail_parts = ", ".join(
            f"{lang}: {info['pct']}%" for lang, info in per_lang.items()
        )
        rows.append(("Multilingual", f"{overall}% non-empty", detail_parts))

    if "tools" in results:
        r = results["tools"]
        rows.append(("Tool Adherence", f"{r.get('pct', 0)}% valid JSON",
                      f"{r.get('valid_json', 0)}/{r.get('total', 0)} prompts"))

    if "story" in results:
        r = results["story"]
        rows.append(("Story Generation", f"{r.get('pct', 0)}% narrative start",
                      f"{r.get('narrative_starts', 0)}/{r.get('total', 0)} prompts"))

    col_widths = (
        max(len(r[0]) for r in rows) if rows else 20,
        max(len(r[1]) for r in rows) if rows else 20,
        max(len(r[2]) for r in rows) if rows else 30,
    )

    def _row(c0: str, c1: str, c2: str) -> str:
        return f"| {c0:<{col_widths[0]}} | {c1:<{col_widths[1]}} | {c2:<{col_widths[2]}} |"

    separator = "|-" + "-" * col_widths[0] + "-|-" + "-" * col_widths[1] + "-|-" + "-" * col_widths[2] + "-|"
    lines = [
        _row("Task", "Score", "Detail"),
        separator,
    ] + [_row(*r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="ParagAI Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to ParagAI .pt checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Path to tokenizer JSON file.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="perplexity,multiling,tools,story",
        help="Comma-separated list of tasks to run.",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation text file (required for perplexity task).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto, cpu, cuda, mps.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/eval_results.json",
        help="Where to save JSON results.",
    )
    args = parser.parse_args()

    # ---- device --------------------------------------------------------------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[eval] Device: {device}")

    # ---- load model ----------------------------------------------------------
    print(f"[eval] Loading model from {args.checkpoint} …")
    try:
        model, tok, model_config = _load_model_and_tokenizer(
            args.checkpoint, args.tokenizer, device
        )
    except RuntimeError as exc:
        print(f"[eval] ERROR: {exc}")
        sys.exit(1)

    tasks_requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_results: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
        "device": device,
        "tasks_run": tasks_requested,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ---- run tasks -----------------------------------------------------------
    if "perplexity" in tasks_requested:
        print("[eval] Running: perplexity …")
        all_results["perplexity"] = task_perplexity(model, tok, args.val_data, device)
        print(f"       PPL = {all_results['perplexity'].get('ppl', 'N/A')}")

    if "multiling" in tasks_requested:
        print("[eval] Running: multiling …")
        all_results["multiling"] = task_multiling(model, tok, device)
        print(f"       Overall non-empty: {all_results['multiling'].get('overall_pct')}%")

    if "tools" in tasks_requested:
        print("[eval] Running: tools …")
        all_results["tools"] = task_tools(model, tok, device)
        print(f"       Valid JSON tool responses: {all_results['tools'].get('pct')}%")

    if "story" in tasks_requested:
        print("[eval] Running: story …")
        all_results["story"] = task_story(model, tok, device)
        print(f"       Narrative starts: {all_results['story'].get('pct')}%")

    # ---- print table ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60 + "\n")
    print(_format_markdown_table(all_results))
    print()

    # ---- save JSON -----------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"[eval] Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
