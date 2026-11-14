"""Trainer class for ParagAI — Phase 3: Hardware Acceleration & Distributed Training."""

import math
import os
import sys
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from paragai.config import TrainConfig
from paragai.model.transformer import ParagAIModel
from paragai.training.dataset import TextDataset
from paragai.training.lr_scheduler import CosineWarmupScheduler
from paragai.utils.logger import get_logger
from paragai.utils.telemetry import TrainingTelemetry

logger = get_logger("Trainer")

# ChatML generation-preview prompt
_PREVIEW_PROMPT = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"


class Trainer:
    """Manages model training, validation, optimisation, and checkpointing.

    Supports:
    - ``torch.compile`` (Linux/macOS + CUDA only; silently skipped on Windows)
    - bfloat16 mixed-precision autocast via ``torch.cuda.amp.autocast``
    - Gradient accumulation over *N* micro-steps
    - Multi-GPU Distributed Data Parallel (DDP) when launched with ``torchrun``
    - Live training telemetry dashboard via :class:`~paragai.utils.telemetry.TrainingTelemetry`
    - Generation preview every ``eval_interval`` steps (optional tokenizer)
    """

    def __init__(
        self,
        model: ParagAIModel,
        dataset: TextDataset,
        train_config: TrainConfig,
        device: torch.device | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = train_config
        self.tokenizer = tokenizer

        # ------------------------------------------------------------------ #
        # Device resolution                                                     #
        # ------------------------------------------------------------------ #
        if device is not None:
            self.device = device
        elif train_config.device == "cuda" or (
            train_config.device == "auto" and torch.cuda.is_available()
        ):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # ------------------------------------------------------------------ #
        # DDP initialisation (torchrun sets LOCAL_RANK)                        #
        # ------------------------------------------------------------------ #
        self._is_ddp = False
        self._ddp_rank = 0
        self._is_main = True

        if dist.is_available() and "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self._ddp_rank = dist.get_rank()
            self._is_main = self._ddp_rank == 0
            self._is_ddp = True
            self.device = torch.device(f"cuda:{local_rank}")
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[local_rank])
            logger.info(f"DDP initialised — rank {self._ddp_rank}/{dist.get_world_size()}")

        # ------------------------------------------------------------------ #
        # torch.compile                                                         #
        # ------------------------------------------------------------------ #
        if train_config.use_compile:
            if sys.platform == "win32":
                logger.warning("torch.compile disabled on Windows (eager mode)")
            elif not torch.cuda.is_available():
                logger.warning("torch.compile skipped — CUDA not available")
            else:
                raw = self.model.module if self._is_ddp else self.model
                compiled = torch.compile(raw, mode="default")
                if self._is_ddp:
                    self.model.module = compiled  # type: ignore[assignment]
                else:
                    self.model = compiled  # type: ignore[assignment]
                logger.info("torch.compile enabled (mode=default)")

        # ------------------------------------------------------------------ #
        # Mixed-precision context                                               #
        # ------------------------------------------------------------------ #
        self._amp_ctx: Any = nullcontext()
        self._scaler: torch.cuda.amp.GradScaler | None = None
        if train_config.use_amp and self.device.type == "cuda":
            self._amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
            # GradScaler only needed for float16, NOT bfloat16
            logger.info("AMP enabled — bfloat16 autocast (no GradScaler needed)")

        # ------------------------------------------------------------------ #
        # Optimiser & scheduler                                                 #
        # ------------------------------------------------------------------ #
        self.optimizer = self._configure_optimizer()
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_iters=self.config.warmup_iters,
            max_iters=self.config.max_iters,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
        )

        self.best_val_loss = float("inf")
        self.current_iter = 0

        # ------------------------------------------------------------------ #
        # Telemetry                                                             #
        # ------------------------------------------------------------------ #
        self._telemetry = TrainingTelemetry(total_steps=self.config.max_iters)

    # ---------------------------------------------------------------------- #
    # Private helpers                                                           #
    # ---------------------------------------------------------------------- #

    def _raw_model(self) -> ParagAIModel:
        """Return the unwrapped model (strips DDP and compile wrappers)."""
        m = self.model
        if self._is_ddp and isinstance(m, DDP):
            m = m.module  # type: ignore[assignment]
        # torch.compile wraps in OptimizedModule; unwrap via _orig_mod if present
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod  # type: ignore[return-value]
        return m  # type: ignore[return-value]

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW with decoupled weight decay for 2-D weight matrices only."""
        decay_params = []
        no_decay_params = []

        raw = self._raw_model()
        for name, param in raw.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    # ---------------------------------------------------------------------- #
    # Evaluation                                                                #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Estimate average loss and perplexity across train and val splits."""
        self.model.eval()
        out: dict[str, float] = {}

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.dataset.get_batch(
                    split,
                    batch_size=self.config.batch_size,
                    seq_len=self._raw_model().config.max_seq_len,
                    device=self.device,
                )
                with self._amp_ctx:
                    _, loss = self.model(x, targets=y)
                losses[k] = loss.item()

            mean_loss = losses.mean().item()
            out[split] = mean_loss
            out[f"{split}_ppl"] = math.exp(min(20.0, mean_loss))

        self.model.train()
        return out

    # ---------------------------------------------------------------------- #
    # Generation preview                                                        #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _generation_preview(self) -> None:
        """Generate up to 50 tokens from the ChatML prompt and print to stdout."""
        if self.tokenizer is None:
            return
        raw = self._raw_model()
        raw.eval()
        try:
            ids = self.tokenizer.encode(_PREVIEW_PROMPT, add_special_tokens=False)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            max_new = 50
            seq_len = raw.config.max_seq_len

            for _ in range(max_new):
                ctx = input_ids[:, -seq_len:]
                with self._amp_ctx:
                    logits, _ = raw(ctx)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=1)

            generated_ids = input_ids[0, len(ids) :].tolist()
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\n[Preview] {text!r}\n", flush=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Generation preview failed: {exc}")
        finally:
            raw.train()

    # ---------------------------------------------------------------------- #
    # Checkpointing                                                             #
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, file_path: str, is_best: bool = False) -> None:
        """Save training state checkpoint using the unwrapped raw model."""
        if not self._is_main:
            return  # Only rank-0 saves
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        raw = self._raw_model()
        checkpoint = {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iter": self.current_iter,
            "best_val_loss": self.best_val_loss,
            "model_config": raw.config.model_dump(),
            "train_config": self.config.model_dump(),
        }
        torch.save(checkpoint, file_path)
        logger.info(f"Saved checkpoint to {file_path}")

    def load_checkpoint(self, file_path: str) -> None:
        """Resume training from a saved checkpoint (backwards-compatible)."""
        checkpoint = torch.load(file_path, map_location=self.device)
        # Filter legacy causal-mask buffers that are no longer part of the state
        state_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items() if not k.endswith(".attn.mask")
        }
        self._raw_model().load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["current_iter"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed checkpoint from {file_path} at iteration {self.current_iter}")

    # ---------------------------------------------------------------------- #
    # Main training loop                                                        #
    # ---------------------------------------------------------------------- #

    def train(self) -> None:
        """Run the complete training schedule with gradient accumulation and telemetry."""
        self.model.train()
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        raw = self._raw_model()
        seq_len = raw.config.max_seq_len
        tokens_per_step = self.config.batch_size * seq_len
        grad_accum = max(1, self.config.grad_accum_steps)

        if self._is_main:
            logger.info(
                f"Starting training: {self.config.max_iters} steps | "
                f"Device: {self.device} | "
                f"Params: {raw.count_parameters():,} | "
                f"GradAccum: {grad_accum} | "
                f"AMP: {self.config.use_amp} | "
                f"Compile: {self.config.use_compile}"
            )

        for step in range(self.current_iter, self.config.max_iters):
            self.current_iter = step
            time.perf_counter()

            # -------------------------------------------------------------- #
            # Periodic evaluation                                               #
            # -------------------------------------------------------------- #
            if step > 0 and step % self.config.eval_interval == 0 and self._is_main:
                metrics = self.estimate_loss()
                val_loss = metrics["val"]
                val_ppl = metrics["val_ppl"]

                self._telemetry.log_eval(val_loss, val_ppl)

                # Print telemetry dashboard (clear previous line with \r)
                dashboard = self._telemetry.format_dashboard()
                print(f"\r{dashboard}", flush=True)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                    self.save_checkpoint(best_path, is_best=True)

                # Generation preview
                self._generation_preview()

            # -------------------------------------------------------------- #
            # Gradient accumulation micro-steps                                 #
            # -------------------------------------------------------------- #
            accumulated_loss = 0.0

            for micro_step in range(grad_accum):
                x, y = self.dataset.get_batch(
                    "train",
                    batch_size=self.config.batch_size,
                    seq_len=seq_len,
                    device=self.device,
                )

                # In DDP, only sync gradients on the last micro-step
                sync_ctx = (
                    self.model.no_sync()  # type: ignore[union-attr]
                    if self._is_ddp and micro_step < grad_accum - 1
                    else nullcontext()
                )

                with sync_ctx:
                    with self._amp_ctx:
                        _, loss = self.model(x, targets=y)
                    # Scale loss for accumulation
                    loss = loss / grad_accum
                    loss.backward()
                    accumulated_loss += loss.item()

            # -------------------------------------------------------------- #
            # Gradient clipping & optimiser step                               #
            # -------------------------------------------------------------- #
            if self.config.grad_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                ).item()
            else:
                grad_norm = 0.0

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            # -------------------------------------------------------------- #
            # Telemetry update                                                  #
            # -------------------------------------------------------------- #
            current_lr = (
                self.scheduler.get_last_lr()[0]
                if hasattr(self.scheduler, "get_last_lr")
                else self.config.learning_rate
            )
            self._telemetry.update(
                step=step,
                loss=accumulated_loss * grad_accum,  # restore un-scaled loss for display
                grad_norm=grad_norm,
                lr=current_lr,
                tokens_processed=tokens_per_step,
            )

            # Live progress feedback for terminal & notebooks
            if self._is_main and (step == 0 or (step + 1) % 25 == 0 or (step + 1) == self.config.max_iters):
                vram_gb = 0.0
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.memory_allocated() / (1024**3)
                tok_sec = self._telemetry.tokens_per_sec
                step_loss = accumulated_loss * grad_accum
                print(
                    f"🚀 [Step {step + 1:4d}/{self.config.max_iters}] "
                    f"Loss: {step_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {tok_sec:,.0f} tok/s | "
                    f"VRAM: {vram_gb:.1f} GB",
                    flush=True,
                )

        # ------------------------------------------------------------------ #
        # Final checkpoint                                                      #
        # ------------------------------------------------------------------ #
        final_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_path)
        if self._is_main:
            logger.info("Training complete!")

        if self._is_ddp and dist.is_initialized():
            dist.destroy_process_group()
