"""Convert ParagAI PyTorch checkpoint into HuggingFace/LLaMA SafeTensors format."""

import os
import json
import argparse
import torch
from safetensors.torch import save_file


def convert_checkpoint_to_hf(checkpoint_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    state_dict = checkpoint["model_state_dict"]

    hf_tensors = {}
    for key, tensor in state_dict.items():
        if key == "token_embedding.weight":
            hf_tensors["model.embed_tokens.weight"] = tensor.clone().contiguous()
        elif key == "final_norm.weight":
            hf_tensors["model.norm.weight"] = tensor.clone().contiguous()
        elif key == "lm_head.weight":
            hf_tensors["lm_head.weight"] = tensor.clone().contiguous()
        elif key.startswith("blocks."):
            parts = key.split(".")
            block_idx = parts[1]
            sub_key = ".".join(parts[2:])

            if sub_key == "norm1.weight":
                target = f"model.layers.{block_idx}.input_layernorm.weight"
            elif sub_key == "norm2.weight":
                target = f"model.layers.{block_idx}.post_attention_layernorm.weight"
            elif sub_key == "attn.q_proj.weight":
                target = f"model.layers.{block_idx}.self_attn.q_proj.weight"
            elif sub_key == "attn.k_proj.weight":
                target = f"model.layers.{block_idx}.self_attn.k_proj.weight"
            elif sub_key == "attn.v_proj.weight":
                target = f"model.layers.{block_idx}.self_attn.v_proj.weight"
            elif sub_key == "attn.out_proj.weight":
                target = f"model.layers.{block_idx}.self_attn.o_proj.weight"
            elif sub_key == "ffn.w_gate.weight":
                target = f"model.layers.{block_idx}.mlp.gate_proj.weight"
            elif sub_key == "ffn.w_up.weight":
                target = f"model.layers.{block_idx}.mlp.up_proj.weight"
            elif sub_key == "ffn.w_down.weight":
                target = f"model.layers.{block_idx}.mlp.down_proj.weight"
            else:
                continue
            hf_tensors[target] = tensor.clone().contiguous()

    # Save SafeTensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(hf_tensors, safetensors_path)

    # Calculate exact intermediate_size matching SwiGLUFeedForward
    hidden_dim = 4 * model_config["n_embd"]
    hidden_dim = int(2 * hidden_dim / 3)
    intermediate_size = 64 * ((hidden_dim + 63) // 64)

    # Save standard LLaMA-style config.json
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": model_config["vocab_size"],
        "hidden_size": model_config["n_embd"],
        "intermediate_size": intermediate_size,
        "num_hidden_layers": model_config["n_layer"],
        "num_attention_heads": model_config["n_head"],
        "num_key_value_heads": model_config["n_head"],
        "max_position_embeddings": model_config["max_seq_len"],
        "rms_norm_eps": model_config.get("rms_norm_eps", 1e-5),
        "rope_theta": model_config.get("rope_theta", 10000.0),
        "tie_word_embeddings": True,
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    print(f"Exported {len(hf_tensors)} tensors to {safetensors_path}")
    print(f"Exported config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Export ParagAI to HuggingFace SafeTensors")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ParagAI .pt checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/export_hf", help="Output directory")
    args = parser.parse_args()

    convert_checkpoint_to_hf(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
