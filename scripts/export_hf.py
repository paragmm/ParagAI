"""Convert ParagAI PyTorch checkpoint into HuggingFace/LLaMA SafeTensors format.

Enhanced export with:
- generation_config.json
- tokenizer_config.json  (ChatML template)
- special_tokens_map.json
- --verify flag (SafeTensors sanity check)
- Tensor shape table printout
- vLLM serve instructions
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# ChatML Jinja2 chat template (same as llama-3 / Mistral instruct style)
# ---------------------------------------------------------------------------
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


# ---------------------------------------------------------------------------
# Tensor key mapping
# ---------------------------------------------------------------------------


def _map_key(key: str) -> str | None:
    """Map a ParagAI state-dict key to its HuggingFace/LLaMA equivalent.

    Returns ``None`` if the key should be skipped.
    """
    if key == "token_embedding.weight":
        return "model.embed_tokens.weight"
    if key == "final_norm.weight":
        return "model.norm.weight"
    if key == "lm_head.weight":
        return "lm_head.weight"

    if key.startswith("blocks."):
        parts = key.split(".")
        block_idx = parts[1]
        sub_key = ".".join(parts[2:])

        mapping: dict[str, str] = {
            "norm1.weight": f"model.layers.{block_idx}.input_layernorm.weight",
            "norm2.weight": f"model.layers.{block_idx}.post_attention_layernorm.weight",
            "attn.q_proj.weight": f"model.layers.{block_idx}.self_attn.q_proj.weight",
            "attn.k_proj.weight": f"model.layers.{block_idx}.self_attn.k_proj.weight",
            "attn.v_proj.weight": f"model.layers.{block_idx}.self_attn.v_proj.weight",
            "attn.out_proj.weight": f"model.layers.{block_idx}.self_attn.o_proj.weight",
            "ffn.w_gate.weight": f"model.layers.{block_idx}.mlp.gate_proj.weight",
            "ffn.w_up.weight": f"model.layers.{block_idx}.mlp.up_proj.weight",
            "ffn.w_down.weight": f"model.layers.{block_idx}.mlp.down_proj.weight",
        }
        return mapping.get(sub_key)  # None if not found → skip

    return None  # skip anything else


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert_checkpoint_to_hf(checkpoint_path: str, output_dir: str, verify: bool = False) -> None:
    """Export a ParagAI .pt checkpoint to HuggingFace SafeTensors + JSON configs.

    Args:
        checkpoint_path: Path to the ``*.pt`` ParagAI checkpoint.
        output_dir:      Directory in which all HF artefacts are written.
        verify:          When *True*, reload the exported SafeTensors file and
                         confirm every tensor loads without corruption.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[export_hf] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config: dict = checkpoint["model_config"]
    state_dict: dict[str, torch.Tensor] = checkpoint["model_state_dict"]

    # ---- tensor key remapping ------------------------------------------------
    hf_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        hf_key = _map_key(key)
        if hf_key is not None:
            hf_tensors[hf_key] = tensor.clone().contiguous()

    # ---- print tensor shape table --------------------------------------------
    _print_tensor_table(hf_tensors)

    # ---- SafeTensors ----------------------------------------------------------
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(hf_tensors, safetensors_path)
    print(f"\n[export_hf] Saved {len(hf_tensors)} tensors → {safetensors_path}")

    # ---- derive architecture numbers -----------------------------------------
    hidden_dim = 4 * model_config["n_embd"]
    hidden_dim = int(2 * hidden_dim / 3)
    intermediate_size = 64 * ((hidden_dim + 63) // 64)

    vocab_size: int = model_config["vocab_size"]
    n_embd: int = model_config["n_embd"]
    n_layer: int = model_config["n_layer"]
    n_head: int = model_config["n_head"]
    max_seq_len: int = model_config["max_seq_len"]
    rms_norm_eps: float = model_config.get("rms_norm_eps", 1e-5)
    rope_theta: float = model_config.get("rope_theta", 10000.0)

    # Guess special token IDs from tokenizer (best effort — may not exist yet)
    bos_token_id: int = model_config.get("bos_token_id", 1)
    eos_token_id: int = model_config.get("eos_token_id", 2)
    pad_token_id: int = model_config.get("pad_token_id", 0)

    # ---- config.json ---------------------------------------------------------
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": n_embd,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head,
        "max_position_embeddings": max_seq_len,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "tie_word_embeddings": True,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    }
    _write_json(hf_config, output_dir, "config.json")

    # ---- generation_config.json ----------------------------------------------
    generation_config = {
        "_from_model_config": False,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.15,
        "do_sample": True,
        "transformers_version": "4.40.0",
    }
    _write_json(generation_config, output_dir, "generation_config.json")

    # ---- tokenizer_config.json -----------------------------------------------
    tokenizer_config = {
        "model_type": "llama",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "padding_side": "right",
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "chat_template": CHATML_TEMPLATE,
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|tool_call|>",
        ],
    }
    _write_json(tokenizer_config, output_dir, "tokenizer_config.json")

    # ---- special_tokens_map.json ---------------------------------------------
    special_tokens_map = {
        "bos_token": {
            "content": "<bos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        },
        "eos_token": {
            "content": "<eos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        },
        "unk_token": {
            "content": "<unk>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        },
        "pad_token": {
            "content": "<pad>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        },
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|tool_call|>",
        ],
    }
    _write_json(special_tokens_map, output_dir, "special_tokens_map.json")

    # ---- optional verification -----------------------------------------------
    if verify:
        _verify_safetensors(safetensors_path, hf_tensors)

    # ---- vLLM instructions ---------------------------------------------------
    _print_vllm_instructions(output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(data: dict, directory: str, filename: str) -> None:
    """Serialise *data* as pretty-printed JSON to ``directory/filename``."""
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"[export_hf] Wrote {filename} → {path}")


def _print_tensor_table(tensors: dict[str, torch.Tensor]) -> None:
    """Print a formatted table of exported tensor names and shapes."""
    col_key = max((len(k) for k in tensors), default=10)
    col_shape = 30
    col_dtype = 12
    separator = (
        "+" + "-" * (col_key + 2) + "+" + "-" * (col_shape + 2) + "+" + "-" * (col_dtype + 2) + "+"
    )
    header = f"| {'Tensor Key':<{col_key}} | {'Shape':<{col_shape}} | {'DType':<{col_dtype}} |"
    print("\n" + separator)
    print(header)
    print(separator)
    for key, tensor in sorted(tensors.items()):
        shape_str = str(tuple(tensor.shape))
        dtype_str = str(tensor.dtype).replace("torch.", "")
        print(f"| {key:<{col_key}} | {shape_str:<{col_shape}} | {dtype_str:<{col_dtype}} |")
    print(separator)
    print(f"  Total tensors exported: {len(tensors)}\n")


def _verify_safetensors(path: str, original: dict[str, torch.Tensor]) -> None:
    """Reload the SafeTensors file and verify every tensor matches the original.

    Args:
        path:     Path to the ``model.safetensors`` file just written.
        original: The in-memory mapping used to write the file.

    Raises:
        SystemExit: If any tensor fails the sanity check.
    """
    print("[export_hf] Running --verify: reloading SafeTensors …")
    try:
        from safetensors import safe_open  # type: ignore[import]
    except ImportError:
        print("[export_hf] WARNING: 'safetensors' not installed; skipping verify.")
        return

    errors: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as sf:  # type: ignore[attr-defined]
        loaded_keys = set(sf.keys())
        expected_keys = set(original.keys())

        missing = expected_keys - loaded_keys
        extra = loaded_keys - expected_keys
        if missing:
            errors.append(f"Missing keys in file: {missing}")
        if extra:
            errors.append(f"Extra unexpected keys in file: {extra}")

        for key in expected_keys & loaded_keys:
            reloaded = sf.get_tensor(key)
            orig = original[key]
            if reloaded.shape != orig.shape:
                errors.append(f"Shape mismatch for '{key}': {reloaded.shape} != {orig.shape}")
            elif not torch.allclose(reloaded.float(), orig.float(), atol=1e-6):
                errors.append(f"Value mismatch for '{key}'")

    if errors:
        print("[export_hf] VERIFY FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"[export_hf] ✓ Verify passed — all {len(original)} tensors match.\n")


def _print_vllm_instructions(output_dir: str) -> None:
    """Print ready-to-run vLLM serve commands."""
    sep = "=" * 60
    print(sep)
    print("  vLLM Serve Instructions")
    print(sep)
    print("\n  1. Install vLLM (if not already):")
    print("       pip install vllm\n")
    print("  2. Start the server:")
    print(f"       vllm serve {output_dir} \\")
    print("         --port 8000 \\")
    print("         --host 0.0.0.0 \\")
    print("         --tensor-parallel-size 1 \\")
    print("         --max-model-len 256 \\")
    print("         --dtype bfloat16\n")
    print("  3. Query (example):")
    print("       curl http://localhost:8000/v1/completions \\")
    print('         -H "Content-Type: application/json" \\')
    print(f'         -d \'{{"model": "{output_dir}", "prompt": "Hello", "max_tokens": 64}}\'\n')
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Export ParagAI checkpoint to HuggingFace SafeTensors format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ParagAI .pt checkpoint file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/export_hf",
        help="Directory for HF artefacts (default: checkpoints/export_hf).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="After export, reload SafeTensors and verify tensor integrity.",
    )
    args = parser.parse_args()

    convert_checkpoint_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
