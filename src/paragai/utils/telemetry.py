"""Live training telemetry and dashboard for ParagAI.

Tracks tokens/sec, VRAM, learning rate, gradient norm, train/val loss,
perplexity, and estimated time remaining. Renders a formatted terminal
dashboard without any external dependencies (stdlib + torch only).
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple


try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ─────────────────────────── helpers ────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS`` or ``??:??:??`` when unknown."""
    if seconds < 0 or not math.isfinite(seconds):
        return "??:??:??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _box_line(label: str, value: str, width: int = 48) -> str:
    """Return a single ``│ label : value │`` row padded to *width*."""
    content = f"  {label:<18} {value}"
    # Truncate or pad to fit inside the box borders
    inner = content[: width - 2].ljust(width - 2)
    return f"│{inner}│"


# ─────────────────────────── main class ─────────────────────────────────────

class TrainingTelemetry:
    """Collects and displays live metrics during a ParagAI training run.

    Usage::

        tel = TrainingTelemetry(total_steps=10_000)
        for step, batch in enumerate(loader):
            loss, grad_norm = train_step(batch)
            tel.update(step, loss=loss, grad_norm=grad_norm, lr=scheduler.get_last_lr()[0],
                       tokens_processed=batch_tokens)
            if step % 200 == 0:
                val_loss = evaluate()
                tel.log_eval(val_loss, math.exp(val_loss))
            print(tel.format_dashboard())

    Parameters
    ----------
    total_steps:
        Total number of training steps (used for ETA estimation).  Pass
        ``None`` if unknown.
    rolling_window:
        Number of most-recent steps used for the tokens/sec rolling average.
        Default: 50.
    """

    _WINDOW: int = 50
    _BOX_WIDTH: int = 50

    def __init__(
        self,
        total_steps: Optional[int] = None,
        rolling_window: int = _WINDOW,
    ) -> None:
        self.total_steps: Optional[int] = total_steps
        self._window: int = rolling_window

        # --- timing ---
        self._start_wall: float = time.monotonic()
        # deque of (wall_time, tokens_so_far) pairs for rolling window
        self._tok_window: Deque[Tuple[float, int]] = deque(maxlen=rolling_window + 1)
        self._total_tokens: int = 0

        # --- tracked metrics ---
        self.current_step: int = 0
        self.train_loss: float = float("nan")
        self.grad_norm: float = float("nan")
        self.lr: float = float("nan")
        self.val_loss: float = float("nan")
        self.val_ppl: float = float("nan")

        # Seed the window so the first call to tokens/sec doesn't divide by 0
        self._tok_window.append((self._start_wall, 0))

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float,
        tokens_processed: int,
    ) -> None:
        """Record metrics for the current training *step*.

        Parameters
        ----------
        step:
            Global training step index (0-based).
        loss:
            Scalar cross-entropy loss for this step.
        grad_norm:
            L2 norm of all gradients (before clipping) for this step.
        lr:
            Current learning rate from the scheduler.
        tokens_processed:
            Number of tokens in the batch processed at this step.
        """
        self.current_step = step
        self.train_loss = loss
        self.grad_norm = grad_norm
        self.lr = lr
        self._total_tokens += tokens_processed
        self._tok_window.append((time.monotonic(), self._total_tokens))

    def log_eval(self, val_loss: float, val_ppl: float) -> None:
        """Record the most-recent validation metrics.

        Parameters
        ----------
        val_loss:
            Cross-entropy loss on the validation set.
        val_ppl:
            Perplexity on the validation set (``exp(val_loss)``).
        """
        self.val_loss = val_loss
        self.val_ppl = val_ppl

    # ------------------------------------------------------------------ #
    # Computed properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def elapsed(self) -> float:
        """Wall-clock seconds since this object was created."""
        return time.monotonic() - self._start_wall

    @property
    def tokens_per_sec(self) -> float:
        """Rolling-average tokens per second over the last *window* steps."""
        if len(self._tok_window) < 2:
            return 0.0
        t0, tok0 = self._tok_window[0]
        t1, tok1 = self._tok_window[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return (tok1 - tok0) / dt

    @property
    def vram_mb(self) -> float:
        """Peak VRAM allocated on the default CUDA device, in MB.

        Returns 0.0 when CUDA is unavailable.
        """
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    @property
    def eta(self) -> float:
        """Estimated seconds remaining, or ``-1`` if unknown."""
        if self.total_steps is None or self.current_step <= 0:
            return -1.0
        tps = self.tokens_per_sec
        if tps <= 0:
            return -1.0
        # Use elapsed / step as seconds-per-step estimate
        sps = self.elapsed / max(self.current_step, 1)
        remaining_steps = self.total_steps - self.current_step
        return sps * remaining_steps

    # ------------------------------------------------------------------ #
    # Dashboard rendering                                                  #
    # ------------------------------------------------------------------ #

    def format_dashboard(self) -> str:
        """Return a multi-line ASCII dashboard string for terminal display.

        The box is 50 characters wide and shows all key training metrics.

        Returns
        -------
        str
            A self-contained string ready for ``print()``.  Includes a top
            border, metric rows, a separator between train and eval sections,
            and a bottom border.

        Example output::

            ╔══════════════════════════════════════════════════╗
            ║            ParagAI  Training  Dashboard          ║
            ╠══════════════════════════════════════════════════╣
            │  Step               12 345                       │
            │  Elapsed            00:03:21                     │
            │  ETA                01:12:04                     │
            │  Tokens/sec         42 318                       │
            │  VRAM (MB)          2 048.3                      │
            ╠══════════════════════════════════════════════════╣
            │  Learning Rate      3.0000e-04                   │
            │  Grad Norm          1.4142                       │
            │  Train Loss         2.3456                       │
            ╠══════════════════════════════════════════════════╣
            │  Val Loss           2.5678                       │
            │  Perplexity         13.03                        │
            ╚══════════════════════════════════════════════════╝
        """
        W = self._BOX_WIDTH
        inner_w = W - 2  # inside │ │

        def top(ch: str = "═") -> str:
            return "╔" + ch * inner_w + "╗"

        def mid(ch: str = "═") -> str:
            return "╠" + ch * inner_w + "╣"

        def bot(ch: str = "═") -> str:
            return "╚" + ch * inner_w + "╝"

        def row(label: str, value: str) -> str:
            content = f"  {label:<18} {value}"
            padded = content[: inner_w].ljust(inner_w)
            return f"│{padded}│"

        def title_row(text: str) -> str:
            padded = text.center(inner_w)
            return f"║{padded}║"

        # --- format values -----------------------------------------------
        step_str = f"{self.current_step:,}"
        elapsed_str = _fmt_time(self.elapsed)
        eta_str = _fmt_time(self.eta)
        tps_str = f"{self.tokens_per_sec:,.0f}"
        vram_str = f"{self.vram_mb:.1f}" if _TORCH_AVAILABLE and torch.cuda.is_available() else "N/A"
        lr_str = f"{self.lr:.4e}" if math.isfinite(self.lr) else "N/A"
        gnorm_str = f"{self.grad_norm:.4f}" if math.isfinite(self.grad_norm) else "N/A"
        tloss_str = f"{self.train_loss:.4f}" if math.isfinite(self.train_loss) else "N/A"
        vloss_str = f"{self.val_loss:.4f}" if math.isfinite(self.val_loss) else "—"
        ppl_str = f"{self.val_ppl:.2f}" if math.isfinite(self.val_ppl) else "—"

        lines = [
            top(),
            title_row("ParagAI  Training  Dashboard"),
            mid(),
            row("Step", step_str),
            row("Elapsed", elapsed_str),
            row("ETA", eta_str),
            row("Tokens/sec", tps_str),
            row("VRAM (MB)", vram_str),
            mid(),
            row("Learning Rate", lr_str),
            row("Grad Norm", gnorm_str),
            row("Train Loss", tloss_str),
            mid(),
            row("Val Loss", vloss_str),
            row("Perplexity", ppl_str),
            bot(),
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TrainingTelemetry("
            f"step={self.current_step}, "
            f"train_loss={self.train_loss:.4f}, "
            f"tok/s={self.tokens_per_sec:.0f})"
        )
