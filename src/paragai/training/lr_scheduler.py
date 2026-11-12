"""Learning rate scheduler with linear warmup and cosine decay."""

import math


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup."""

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        max_iters: int,
        max_lr: float,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self, step: int) -> float:
        # Linear warmup phase
        if step < self.warmup_iters:
            return self.max_lr * (step + 1) / max(1, self.warmup_iters)

        # Constant min_lr if step exceeds max_iters
        if step > self.max_iters:
            return self.min_lr

        # Cosine decay phase
        decay_ratio = (step - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self) -> float:
        """Update optimizer learning rates and advance step."""
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr

