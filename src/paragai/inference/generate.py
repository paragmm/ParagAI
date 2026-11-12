"""Autoregressive text generation pipeline with Key-Value (KV) Caching."""

from typing import Optional, Iterator
import torch
from paragai.model.transformer import ParagAIModel
from paragai.tokenizer.base import BaseTokenizer
from paragai.inference.sampler import sample_next_token


class Generator:
    """Manages high-speed text generation using KV-Caching from a trained ParagAI model."""

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
        use_cache: bool = True,
    ) -> str:
        """Generate text continuation from a prompt with optional KV-caching."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        generated = list(token_ids)
        kv_caches = None

        # Prefill stage (process prompt)
        if use_cache:
            logits, _, kv_caches = self.model(x, use_cache=True)
            next_logits = logits[0, -1, :].clone()
        else:
            logits, _ = self.model(x)
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
            return self.tokenizer.decode(generated)

        generated.append(next_token)

        # Decoding stage (generate new tokens one by one)
        curr_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens - 1):
            if len(generated) >= self.model.config.max_seq_len:
                break

            if use_cache:
                # O(1) step: feed only the single new token into the network
                logits, _, kv_caches = self.model(curr_token, past_kv_caches=kv_caches, use_cache=True)
                next_logits = logits[0, -1, :].clone()
            else:
                x_full = torch.tensor([generated], dtype=torch.long, device=self.device)
                logits, _ = self.model(x_full)
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
            curr_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

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
        """Stream generated characters/tokens using KV-cache with minimal latency."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        generated = list(token_ids)

        # Prefill prompt
        logits, _, kv_caches = self.model(x, use_cache=True)
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
            return

        generated.append(next_token)
        yield self.tokenizer.decode([next_token])

        curr_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens - 1):
            if len(generated) >= self.model.config.max_seq_len:
                break

            logits, _, kv_caches = self.model(curr_token, past_kv_caches=kv_caches, use_cache=True)
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
            curr_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            yield self.tokenizer.decode([next_token])
