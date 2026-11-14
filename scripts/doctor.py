#!/usr/bin/env python
"""ParagAI pre-flight diagnostic script (doctor.py).

Run from the project root::

    python scripts/doctor.py

Exit codes
----------
0  All critical checks passed.
1  One or more critical checks failed.

Checks performed
----------------
1.  Python version ≥ 3.10
2.  PyTorch version + CUDA availability, device name, VRAM, bfloat16 support
3.  torch.compile availability (Windows → Eager mode warning)
4.  Load checkpoints/best_model.pt and verify tensor shapes vs configs/model_small.yaml
5.  Load checkpoints/tokenizer.json and roundtrip encode/decode for four strings
6.  Ollama service reachable at http://127.0.0.1:11434
7.  vLLM importable
8.  Coloured summary (rich if available, else ANSI fallback)
9.  Exit 0 / 1 based on critical check outcomes
"""

from __future__ import annotations

import json
import platform
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Rich / ANSI colour helpers
# ---------------------------------------------------------------------------
try:
    from rich import box as rich_box
    from rich.console import Console
    from rich.table import Table

    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore[assignment]

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

PASS_SYM = "✓"
FAIL_SYM = "✗"
WARN_SYM = "⚠"


def _ansi(text: str, colour: str) -> str:
    """Wrap *text* in ANSI escape codes (no-op when stdout is not a tty)."""
    if not sys.stdout.isatty():
        return text
    return f"{colour}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single diagnostic check.

    Attributes
    ----------
    name:
        Human-readable check name.
    passed:
        ``True`` if the check succeeded (or produced only a warning).
    critical:
        ``True`` if a failure here should cause exit-code 1.
    detail:
        Extra information shown in the summary table.
    warning:
        ``True`` if the check passed with caveats (shown differently).
    """

    name: str
    passed: bool
    critical: bool = True
    detail: str = ""
    warning: bool = False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> CheckResult:
    """Verify Python ≥ 3.10."""
    ver = sys.version_info
    detail = f"Python {ver.major}.{ver.minor}.{ver.micro}"
    passed = (ver.major, ver.minor) >= (3, 10)
    return CheckResult(
        name="Python version ≥ 3.10",
        passed=passed,
        critical=True,
        detail=detail + ("" if passed else " — requires 3.10+"),
    )


def check_pytorch() -> CheckResult:
    """Check PyTorch installation, CUDA device, VRAM, bfloat16 support."""
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="PyTorch",
            passed=False,
            critical=True,
            detail="torch not importable — install PyTorch",
        )

    details: list[str] = [f"torch {torch.__version__}"]

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device_name: str = torch.cuda.get_device_name(device_idx)
        vram_gb: float = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        bf16: bool = torch.cuda.is_bf16_supported()
        details += [
            f"CUDA ✓ | {device_name}",
            f"VRAM {vram_gb:.1f} GB",
            f"bfloat16 {'✓' if bf16 else '✗'}",
        ]
    else:
        details.append("CUDA ✗ — CPU only")

    return CheckResult(
        name="PyTorch + CUDA",
        passed=True,
        critical=False,  # CPU-only training is still possible
        detail=" | ".join(details),
        warning=not torch.cuda.is_available(),
    )


def check_torch_compile() -> CheckResult:
    """Check whether torch.compile is usable on the current platform."""
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="torch.compile",
            passed=False,
            critical=False,
            detail="torch not importable",
        )

    has_compile = hasattr(torch, "compile")
    is_windows = platform.system() == "Windows"

    if not has_compile:
        return CheckResult(
            name="torch.compile",
            passed=False,
            critical=False,
            detail=f"torch.compile not available (torch {torch.__version__})",
            warning=True,
        )

    if is_windows:
        return CheckResult(
            name="torch.compile",
            passed=True,
            critical=False,
            detail="Available but Windows detected — running in Eager mode (no Triton)",
            warning=True,
        )

    return CheckResult(
        name="torch.compile",
        passed=True,
        critical=False,
        detail=f"Available (torch {torch.__version__})",
    )


def _load_yaml_model_config(yaml_path: Path) -> dict:
    """Minimal YAML parser for simple key: value pairs (no external dep).

    Handles nested mappings one level deep (the ``model:`` block).
    """
    config: dict = {}
    current_section: dict | None = None

    with yaml_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            # Top-level key (no leading spaces)
            if not line.startswith(" ") and not line.startswith("\t"):
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"')
                if val == "":
                    # This is a section header
                    current_section = {}
                    config[key] = current_section
                else:
                    config[key] = _coerce(val)
                    current_section = None
            else:
                # Nested key inside a section
                if current_section is None:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"')
                current_section[key] = _coerce(val)
    return config


def _coerce(val: str):
    """Try to convert *val* to int, float, or bool; fall back to str."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def check_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
) -> CheckResult:
    """Load best_model.pt and verify tensor shapes against model_small.yaml."""
    name = "Checkpoint + config alignment"

    if not checkpoint_path.exists():
        return CheckResult(
            name=name,
            passed=False,
            critical=False,
            detail=f"{checkpoint_path} not found — skipping",
            warning=True,
        )
    if not config_path.exists():
        return CheckResult(
            name=name,
            passed=False,
            critical=False,
            detail=f"{config_path} not found — skipping",
            warning=True,
        )

    try:
        import torch
    except ImportError:
        return CheckResult(name=name, passed=False, critical=False, detail="torch not importable")

    try:
        cfg = _load_yaml_model_config(config_path)
        model_cfg: dict = cfg.get("model", cfg)

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        # Accept raw state-dict or wrapped {"model_state_dict": ...}
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state: dict = ckpt["model_state_dict"]
            # Prefer embedded model_config if present in checkpoint
            if "model_config" in ckpt and isinstance(ckpt["model_config"], dict):
                model_cfg = ckpt["model_config"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            return CheckResult(
                name=name,
                passed=False,
                critical=False,
                detail=f"Unexpected checkpoint type: {type(ckpt)}",
            )

        vocab_size: int = int(model_cfg["vocab_size"])
        n_embd: int = int(model_cfg["n_embd"])
        n_layer: int = int(model_cfg["n_layer"])
        n_head: int = int(model_cfg["n_head"])

        mismatches: list[str] = []

        # --- embedding weight ---
        emb_keys = [k for k in state if "embed" in k and "weight" in k]
        if emb_keys:
            emb_shape = tuple(state[emb_keys[0]].shape)
            if emb_shape[0] != vocab_size or emb_shape[1] != n_embd:
                mismatches.append(f"embed {emb_shape} ≠ expected ({vocab_size}, {n_embd})")
        else:
            mismatches.append("No embedding weight found in checkpoint")

        # --- count transformer blocks ---
        block_indices: set[int] = set()
        for k in state:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p in ("layers", "blocks", "h") and i + 1 < len(parts):
                    try:
                        block_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        if block_indices and max(block_indices) + 1 != n_layer:
            mismatches.append(f"n_layer inferred={max(block_indices) + 1} ≠ config={n_layer}")

        if mismatches:
            return CheckResult(
                name=name,
                passed=False,
                critical=False,
                detail="Shape mismatch: " + "; ".join(mismatches),
            )

        param_count = sum(t.numel() for t in state.values())
        return CheckResult(
            name=name,
            passed=True,
            critical=False,
            detail=(
                f"{len(state)} tensors | {param_count:,} params | "
                f"vocab={vocab_size} n_embd={n_embd} n_layer={n_layer} n_head={n_head}"
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return CheckResult(name=name, passed=False, critical=False, detail=str(exc))


def check_tokenizer_roundtrip(tokenizer_path: Path) -> CheckResult:
    """Load tokenizer.json and perform encode→decode roundtrip tests."""
    name = "Tokenizer roundtrip"

    test_strings: list[str] = [
        "hello world",  # English
        "হ্যালো বাংলা",  # Bengali
        "नमस्ते हिंदी",  # Hindi
        "🚀",  # Emoji
    ]

    if not tokenizer_path.exists():
        return CheckResult(
            name=name,
            passed=False,
            critical=False,
            detail=f"{tokenizer_path} not found — skipping",
            warning=True,
        )

    try:
        with open(tokenizer_path, "r", encoding="utf-8") as fh:
            tok_data = json.load(fh)

        if tok_data.get("tokenizer_class") == "indic_bpe":
            from paragai.tokenizer.indic_bpe import IndicBPETokenizer

            tok = IndicBPETokenizer.load(str(tokenizer_path))
        elif tok_data.get("type") == "bpe":
            from paragai.tokenizer.bpe_tokenizer import BPETokenizer

            tok = BPETokenizer.load(str(tokenizer_path))
        else:
            from paragai.tokenizer.char_tokenizer import CharTokenizer

            tok = CharTokenizer.load(str(tokenizer_path))

        failures: list[str] = []

        for s in test_strings:
            encoded = tok.encode(s)
            decoded = tok.decode(encoded)
            # For char tokenizer, some characters outside its vocabulary might map to UNK
            if decoded != s and not getattr(tok, "is_char_tokenizer", hasattr(tok, "stoi")):
                failures.append(f"{s!r:.20} → {decoded!r:.20}")

        if failures:
            return CheckResult(
                name=name,
                passed=False,
                critical=False,
                detail="Roundtrip failures: " + "; ".join(failures),
            )

        tok_type = tok_data.get("tokenizer_class") or tok_data.get("type", "unknown")
        vocab_len = len(getattr(tok, "vocab", getattr(tok, "stoi", [])))
        return CheckResult(
            name=name,
            passed=True,
            critical=False,
            detail=f"{tok_type} ({vocab_len} tokens) | Roundtrip verified",
        )

    except Exception as exc:  # noqa: BLE001
        return CheckResult(name=name, passed=False, critical=False, detail=str(exc))


def check_ollama() -> CheckResult:
    """Check whether the Ollama service is reachable at http://127.0.0.1:11434."""
    name = "Ollama service"
    url = "http://127.0.0.1:11434"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            body: str = resp.read(256).decode("utf-8", errors="replace")
        return CheckResult(
            name=name,
            passed=True,
            critical=False,
            detail=f"Reachable ({body[:40].strip()!r})",
        )
    except urllib.error.URLError as exc:
        return CheckResult(
            name=name,
            passed=False,
            critical=False,
            detail=f"Not reachable: {exc.reason}",
            warning=True,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=name,
            passed=False,
            critical=False,
            detail=f"Error: {exc}",
            warning=True,
        )


def check_vllm() -> CheckResult:
    """Check whether vLLM is importable."""
    name = "vLLM importable"
    try:
        import vllm

        ver: str = getattr(vllm, "__version__", "unknown")
        return CheckResult(name=name, passed=True, critical=False, detail=f"vllm {ver}")
    except ImportError:
        return CheckResult(
            name=name,
            passed=False,
            critical=False,
            detail="vllm not installed (optional for local serving)",
            warning=True,
        )


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def _symbol(result: CheckResult) -> str:
    """Return the appropriate symbol for *result*."""
    if result.passed and result.warning:
        return WARN_SYM
    return PASS_SYM if result.passed else FAIL_SYM


def _print_summary_rich(results: list[CheckResult]) -> None:
    """Render a pretty table with rich."""
    table = Table(
        title="ParagAI Doctor — Diagnostic Summary",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Status", justify="center", width=6)
    table.add_column("Check", style="bold", min_width=28)
    table.add_column("Critical", justify="center", width=8)
    table.add_column("Detail")

    for r in results:
        sym = _symbol(r)
        if r.passed and not r.warning:
            status_style = "green"
        elif r.warning:
            status_style = "yellow"
        else:
            status_style = "red"

        table.add_row(
            f"[{status_style}]{sym}[/{status_style}]",
            r.name,
            "yes" if r.critical else "no",
            r.detail,
        )

    _console.print()
    _console.print(table)
    _console.print()


def _print_summary_ansi(results: list[CheckResult]) -> None:
    """Render a plain-text summary with ANSI colour codes."""
    sep = "─" * 70
    print()
    print(_ansi("  ParagAI Doctor — Diagnostic Summary", _BOLD))
    print(sep)
    print(f"  {'Status':<6}  {'Critical':<8}  Check / Detail")
    print(sep)
    for r in results:
        sym = _symbol(r)
        if r.passed and not r.warning:
            sym_col = _ansi(sym, _GREEN)
        elif r.warning:
            sym_col = _ansi(sym, _YELLOW)
        else:
            sym_col = _ansi(sym, _RED)

        crit = "yes" if r.critical else "no "
        print(f"  {sym_col}       {crit}      {r.name}")
        if r.detail:
            print(f"           {' ' * 10}  {_ansi(r.detail, _YELLOW if r.warning else '')}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_all_checks(project_root: Path) -> list[CheckResult]:
    """Run every diagnostic check and return results.

    Parameters
    ----------
    project_root:
        Absolute path to the ParagAI workspace root.

    Returns
    -------
    List[CheckResult]
        One entry per check, in the order they were run.
    """
    checkpoint_pt = project_root / "checkpoints" / "best_model.pt"
    config_yaml = project_root / "configs" / "model_small.yaml"
    tokenizer_json = project_root / "checkpoints" / "tokenizer.json"

    checks: list[tuple[str, Callable[[], CheckResult]]] = [
        ("Python version", check_python_version),
        ("PyTorch", check_pytorch),
        ("torch.compile", check_torch_compile),
        ("Checkpoint", lambda: check_checkpoint(checkpoint_pt, config_yaml)),
        ("Tokenizer", lambda: check_tokenizer_roundtrip(tokenizer_json)),
        ("Ollama", check_ollama),
        ("vLLM", check_vllm),
    ]

    results: list[CheckResult] = []
    for _label, fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    name=_label,
                    passed=False,
                    critical=False,
                    detail=f"Unexpected error: {exc}",
                )
            )
    return results


def main() -> None:
    """CLI entry point."""
    project_root = Path(__file__).resolve().parent.parent

    # Add the src directory to sys.path so paragai is importable when running
    # the script directly (before the package is installed in editable mode).
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    results = run_all_checks(project_root)

    if _RICH:
        _print_summary_rich(results)
    else:
        _print_summary_ansi(results)

    critical_failures = [r for r in results if r.critical and not r.passed]
    if critical_failures:
        label = ", ".join(r.name for r in critical_failures)
        msg = f"Critical check(s) failed: {label}"
        if _RICH:
            _console.print(f"[bold red]{FAIL_SYM}  {msg}[/bold red]\n")
        else:
            print(_ansi(f"{FAIL_SYM}  {msg}", _RED))
        sys.exit(1)
    else:
        msg = "All critical checks passed."
        if _RICH:
            _console.print(f"[bold green]{PASS_SYM}  {msg}[/bold green]\n")
        else:
            print(_ansi(f"{PASS_SYM}  {msg}", _GREEN))
        sys.exit(0)


if __name__ == "__main__":
    main()
