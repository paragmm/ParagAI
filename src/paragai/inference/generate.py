"""Autoregressive text generation pipeline."""

from typing import Optional, Callable, Iterator
import torch
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.base import BaseTokenizer
from paragai.inference.sampler import sample_next_token


class Generator:
    """Manages text generation and interactive streaming from a trained ParagAI model."""

    def __init__(self, model: ParagAIModel, tokenizer: BaseTokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_on_eos: bool = True,
    ) -> str:
        """Generate text continuation from a prompt."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        generated = list(token_ids)
        for _ in range(max_new_tokens):
            # Crop context if length exceeds model max_seq_len
            x_cond = x if x.size(1) <= self.model.config.max_seq_len else x[:, -self.model.config.max_seq_len :]

            logits, _ = self.model(x_cond)
            # Focus on the logits for the last token position: shape (VocabSize,)
            next_logits = logits[0, -1, :].clone()

            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=torch.tensor(generated, device=self.device),
            )

            if stop_on_eos and next_token == self.tokenizer.EOS_ID:
                break

            generated.append(next_token)
            x = torch.cat((x, torch.tensor([[next_token]], device=self.device)), dim=1)

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Iterator[str]:
        """Stream generated characters/tokens one by one."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        generated = list(token_ids)
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.model.config.max_seq_len else x[:, -self.model.config.max_seq_len :]
            logits, _ = self.model(x_cond)
            next_logits = logits[0, -1, :].clone()

            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=torch.tensor(generated, device=self.device),
            )

            if next_token == self.tokenizer.EOS_ID:
                break

            generated.append(next_token)
            x = torch.cat((x, torch.tensor([[next_token]], device=self.device)), dim=1)
            yield self.tokenizer.decode([next_token])

