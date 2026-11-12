"""Trainer class for ParagAI."""

import os
import math
import time
from typing import Optional, Dict, Any
import torch
from paragai.config import ModelConfig, TrainConfig
from paragai.model.transformer import ParagAIModel
from paragai.training.dataset import TextDataset
from paragai.training.lr_scheduler import CosineWarmupScheduler
from paragai.utils.logger import get_logger

logger = get_logger("Trainer")


class Trainer:
    """Manages the model training, validation, optimization, and checkpointing."""

    def __init__(
        self,
        model: ParagAIModel,
        dataset: TextDataset,
        train_config: TrainConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.config = train_config

        if device is None:
            if train_config.device == "cuda" or (train_config.device == "auto" and torch.cuda.is_available()):
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)
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

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW with decoupled weight decay for 2D weight matrices only."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Do not decay 1D parameters (RMSNorm scale weights, biases)
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        return optimizer

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate average loss and perplexity across train and val splits."""
        self.model.eval()
        out = {}

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.dataset.get_batch(
                    split,
                    batch_size=self.config.batch_size,
                    seq_len=self.model.config.max_seq_len,
                    device=self.device,
                )
                _, loss = self.model(x, targets=y)
                losses[k] = loss.item()

            mean_loss = losses.mean().item()
            out[split] = mean_loss
            out[f"{split}_ppl"] = math.exp(min(20.0, mean_loss))

        self.model.train()
        return out

    def save_checkpoint(self, file_path: str, is_best: bool = False) -> None:
        """Save training state checkpoint."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_iter": self.current_iter,
            "best_val_loss": self.best_val_loss,
            "model_config": self.model.config.model_dump(),
            "train_config": self.config.model_dump(),
        }
        torch.save(checkpoint, file_path)
        logger.info(f"Saved checkpoint to {file_path}")

    def load_checkpoint(self, file_path: str) -> None:
        """Resume training from saved checkpoint."""
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["current_iter"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed checkpoint from {file_path} at iteration {self.current_iter}")

    def train(self) -> None:
        """Run complete training schedule."""
        self.model.train()
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        logger.info(
            f"Starting training: {self.config.max_iters} steps, "
            f"Device: {self.device}, Parameters: {self.model.count_parameters():,}"
        )

        start_time = time.time()
        for step in range(self.current_iter, self.config.max_iters):
            self.current_iter = step

            # Periodic validation evaluation
            if step > 0 and step % self.config.eval_interval == 0:
                metrics = self.estimate_loss()
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step}/{self.config.max_iters} | "
                    f"Train Loss: {metrics['train']:.4f} | "
                    f"Val Loss: {metrics['val']:.4f} (PPL: {metrics['val_ppl']:.2f}) | "
                    f"Time: {elapsed:.1f}s"
                )

                # Save best model
                if metrics["val"] < self.best_val_loss:
                    self.best_val_loss = metrics["val"]
                    best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                    self.save_checkpoint(best_path, is_best=True)

            # Sample training batch
            x, y = self.dataset.get_batch(
                "train",
                batch_size=self.config.batch_size,
                seq_len=self.model.config.max_seq_len,
                device=self.device,
            )

            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(x, targets=y)

            # Backward pass & gradient clipping
            loss.backward()
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer & scheduler step
            self.optimizer.step()
            self.scheduler.step()

        # Save final checkpoint
        final_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_path)
        logger.info("Training complete!")

