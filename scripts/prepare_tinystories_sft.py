#!/usr/bin/env python3
"""
prepare_tinystories_sft.py
--------------------------
Phase 5 – TinyStories SFT Data Pipeline.

Converts raw TinyStories text into ChatML-formatted SFT examples and
optionally interleaves replay examples from other SFT splits.

Usage
-----
python scripts/prepare_tinystories_sft.py \\
    --input  data/raw/stories/tinystories_chunk_01.txt \\
    --output data/processed/stories_sft.txt \\
    --replay_mix True \\
    --replay_ratio 0.15
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# ChatML template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are ParagAI, a creative and imaginative AI storytelling assistant."
)

CHATML_TEMPLATE = (
    "<|im_start|>system\n"
    "{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{assistant}<|im_end|>"
)

# ---------------------------------------------------------------------------
# Prompt bank  (≥20 English, ≥10 Bengali, ≥5 Hindi)
# ---------------------------------------------------------------------------

ENGLISH_PROMPTS: List[str] = [
    "Write a story for me.",
    "Tell me a creative story.",
    "Write a short bedtime story.",
    "Can you tell me an imaginative story?",
    "Please write a short story.",
    "Create a fun little story.",
    "Tell me a story with a happy ending.",
    "Write a story suitable for children.",
    "Give me a quick short story.",
    "Make up a creative tale.",
    "Write an engaging short story.",
    "Tell me a story about friendship.",
    "Write a story with an unexpected twist.",
    "Compose a short fantasy story.",
    "Share a brief adventure story.",
    "Tell me a simple and heartwarming story.",
    "Write a story that teaches a lesson.",
    "Create a short story for a young child.",
    "Please tell me a fun story.",
    "Write a short original story.",
    "Tell me a story about an animal.",
    "Write a story set in a magical world.",
    "Give me a cheerful bedtime story.",
    "Write a story about a brave child.",
    "Tell me a story with talking animals.",
]

BENGALI_PROMPTS: List[str] = [
    "আমাকে একটি গল্প বলো।",
    "একটি সুন্দর ছোট গল্প লেখো।",
    "আমাকে একটি মজার গল্প শোনাও।",
    "একটি শিশুদের গল্প বলো।",
    "একটি কল্পনাপ্রবণ গল্প লেখো।",
    "আমাকে একটি রূপকথার গল্প বলো।",
    "একটি অ্যাডভেঞ্চার গল্প লেখো।",
    "একটি ছোট্ট প্রেমের গল্প বলো।",
    "আমাকে একটি নৈতিক গল্প শোনাও।",
    "একটি পশু-পাখির গল্প বলো।",
    "একটি বন্ধুত্বের গল্প লেখো।",
]

HINDI_PROMPTS: List[str] = [
    "एक छोटी कहानी सुनाओ।",
    "मुझे एक रोचक कहानी बताओ।",
    "एक बच्चों की कहानी लिखो।",
    "कोई मजेदार कहानी सुनाओ।",
    "एक छोटी सी कल्पना की कहानी लिखो।",
    "मुझे एक नैतिक कहानी बताओ।",
    "एक जानवरों की कहानी सुनाओ।",
]

ALL_BASE_PROMPTS: List[str] = ENGLISH_PROMPTS + BENGALI_PROMPTS + HINDI_PROMPTS


# ---------------------------------------------------------------------------
# Name extraction helpers
# ---------------------------------------------------------------------------

# Heuristic: capitalised word after sentence-starters that often introduce names.
_NAME_INTRO_RE = re.compile(
    r"\b(?:Once upon a time,?\s+(?:there was\s+)?(?:a|an|the)\s+\w+\s+(?:named|called)\s+)"
    r"([A-Z][a-z]{2,})"
    r"|"
    r"\b([A-Z][a-z]{2,})\s+(?:was|lived|had|loved|liked|wanted|went|saw|found)",
    re.MULTILINE,
)


def _extract_first_name(story: str) -> Optional[str]:
    """Return the first character name found in *story*, or None."""
    m = _NAME_INTRO_RE.search(story[:500])
    if m:
        return m.group(1) or m.group(2)
    # Fallback: first capitalised word that isn't a common story opener.
    _SKIP = {
        "Once", "There", "The", "A", "An", "In", "On", "One", "It",
        "He", "She", "They", "We", "His", "Her", "Their",
    }
    for word in story.split():
        clean = re.sub(r"[^A-Za-z]", "", word)
        if clean and clean[0].isupper() and clean not in _SKIP and len(clean) > 2:
            return clean
    return None


def _pick_prompt(story: str) -> str:
    """Choose a prompt, possibly personalised with a character name."""
    name = _extract_first_name(story)
    candidates = list(ALL_BASE_PROMPTS)
    if name:
        candidates += [
            f"Write a story about {name}.",
            f"Tell me a story about {name}.",
            f"Can you write a story featuring {name}?",
        ]
    return random.choice(candidates)


# ---------------------------------------------------------------------------
# Story parsing
# ---------------------------------------------------------------------------

_ENDOFTEXT_RE = re.compile(r"<\|endoftext\|>")


def parse_stories(raw: str) -> List[str]:
    """Split *raw* text into individual story strings."""
    # First split on <|endoftext|>
    parts = _ENDOFTEXT_RE.split(raw)
    stories: List[str] = []
    for part in parts:
        # Within each part, split further on double newlines.
        candidates = re.split(r"\n{2,}", part)
        for c in candidates:
            s = c.strip()
            if s:
                stories.append(s)
    return stories


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

def format_chatml(prompt: str, story: str) -> str:
    """Wrap a (prompt, story) pair in ChatML."""
    return CHATML_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        user=prompt,
        assistant=story,
    )


# ---------------------------------------------------------------------------
# Replay file loading
# ---------------------------------------------------------------------------

REPLAY_FILES = [
    "data/raw/chat/conversations_sft.txt",
    "data/raw/translation/parallel_sft.txt",
    "data/raw/automation/tools_sft.txt",
]


def load_replay_examples(workspace_root: Path) -> List[str]:
    """Load lines from known replay SFT files; skip files that don't exist."""
    lines: List[str] = []
    for rel_path in REPLAY_FILES:
        p = workspace_root / rel_path
        if not p.exists():
            print(f"[replay] {rel_path} not found — skipping.", file=sys.stderr)
            continue
        with p.open("r", encoding="utf-8") as fh:
            file_lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        print(f"[replay] loaded {len(file_lines):,} lines from {rel_path}")
        lines.extend(file_lines)
    return lines


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_sft_examples(
    stories: List[str],
    replay_examples: List[str],
    replay_mix: bool,
    replay_ratio: float,
    seed: int = 42,
) -> List[str]:
    """
    Combine ChatML-wrapped story examples with optional replay examples.

    Parameters
    ----------
    stories:
        Parsed story strings.
    replay_examples:
        Pre-formatted lines from replay files.
    replay_mix:
        If True, interleave replay examples.
    replay_ratio:
        Fraction of the final dataset that should come from replay.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Shuffled list of training example strings.
    """
    rng = random.Random(seed)

    # Format story examples as ChatML
    story_examples: List[str] = []
    for story in stories:
        prompt = _pick_prompt(story)
        story_examples.append(format_chatml(prompt, story))

    replay_injected = 0
    all_examples: List[str] = list(story_examples)

    if replay_mix and replay_examples:
        n_story = len(story_examples)
        # target: replay_ratio fraction of total
        # total = n_story + n_replay  =>  n_replay = n_story * ratio / (1 - ratio)
        n_replay_target = int(n_story * replay_ratio / max(1 - replay_ratio, 1e-9))
        n_replay_target = min(n_replay_target, len(replay_examples))

        sampled_replay = rng.sample(replay_examples, n_replay_target)
        all_examples.extend(sampled_replay)
        replay_injected = len(sampled_replay)

    rng.shuffle(all_examples)
    return all_examples, replay_injected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw TinyStories to ChatML-formatted SFT examples."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw TinyStories .txt file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/stories_sft.txt",
        help="Output path for the SFT dataset (default: data/processed/stories_sft.txt).",
    )
    parser.add_argument(
        "--replay_mix",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="BOOL",
        help="Interleave replay examples (default: True).",
    )
    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=0.15,
        help="Fraction of dataset from replay sources (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Safety guard: never touch combined_corpus.txt
    combined_corpus = Path("data/processed/combined_corpus.txt").resolve()
    if output_path.resolve() == combined_corpus:
        print(
            "[ERROR] Output path points to data/processed/combined_corpus.txt — "
            "this file must not be modified. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read input
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    raw_text = input_path.read_text(encoding="utf-8", errors="replace")

    if not raw_text.strip():
        warnings.warn(
            f"Input file '{input_path}' is empty. Writing empty output.",
            stacklevel=1,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        print("Written: 0 examples (empty input).")
        return

    print(f"[info] Reading stories from: {input_path}")
    stories = parse_stories(raw_text)
    print(f"[info] Parsed {len(stories):,} stories.")

    # Determine workspace root (project root, two levels up from scripts/)
    workspace_root = Path(__file__).resolve().parent.parent

    # Load replay examples
    replay_examples: List[str] = []
    if args.replay_mix:
        replay_examples = load_replay_examples(workspace_root)
    else:
        print("[info] Replay mix disabled.")

    # Build examples
    all_examples, replay_injected = build_sft_examples(
        stories=stories,
        replay_examples=replay_examples,
        replay_mix=args.replay_mix,
        replay_ratio=args.replay_ratio,
        seed=args.seed,
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n\n".join(all_examples) + "\n"
    output_path.write_text(output_text, encoding="utf-8")

    total_chars = len(output_text)
    print("\n=== SFT Pipeline Report ===")
    print(f"  Total examples written : {len(all_examples):,}")
    print(f"  Story examples         : {len(all_examples) - replay_injected:,}")
    print(f"  Replay examples inject : {replay_injected:,}")
    print(f"  Total characters       : {total_chars:,}")
    print(f"  Output file            : {output_path}")
    print("===========================")


if __name__ == "__main__":
    main()
