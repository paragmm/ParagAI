"""Convert exported SafeTensors & Tokenizer to GGUF format for Ollama."""

import os
import json
import argparse
import numpy as np
import gguf
from safetensors import safe_open


def convert_to_gguf(
    input_dir: str,
    tokenizer_path: str,
    output_path: str,
) -> None:
    config_path = os.path.join(input_dir, "config.json")
    safetensors_path = os.path.join(input_dir, "model.safetensors")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load tokenizer vocabulary
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    if tok_data.get("type") == "bpe":
        vocab_dict = {int(k): bytes(v) for k, v in tok_data["vocab"].items()}
        tokens = [vocab_dict.get(i, b"<unk>") for i in range(len(vocab_dict))]
    else:
        itos = {int(k): v for k, v in tok_data["itos"].items()}
        tokens = [itos.get(i, "<unk>").encode("utf-8") for i in range(len(itos))]

    writer = gguf.GGUFWriter(output_path, arch="llama")

    # Determine accurate feed_forward_length (SwiGLU intermediate size)
    ffn_length = config.get("intermediate_size")
    if ffn_length is None:
        with safe_open(safetensors_path, framework="numpy", device="cpu") as f:
            for k in f.keys():
                if "gate_proj.weight" in k or "ffn_gate.weight" in k:
                    ffn_length = int(f.get_tensor(k).shape[0])
                    break
        if ffn_length is None:
            hidden_dim = 4 * config["hidden_size"]
            hidden_dim = int(2 * hidden_dim / 3)
            ffn_length = 64 * ((hidden_dim + 63) // 64)

    # 1. Model architecture hyperparameters
    writer.add_name("ParagAI")
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config.get("num_key_value_heads", config["num_attention_heads"]))
    writer.add_feed_forward_length(ffn_length)
    writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-5))
    writer.add_rope_freq_base(config.get("rope_theta", 10000.0))
    writer.add_rope_dimension_count(config["hidden_size"] // config["num_attention_heads"])

    # 2. Tokenizer metadata
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(3)
    writer.add_pad_token_id(0)

    # 3. Tensors mapped to GGUF standard llama naming convention
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, config["num_hidden_layers"])
    with safe_open(safetensors_path, framework="numpy", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Ensure float32 or float16 numpy array
            if tensor.dtype == np.float64:
                tensor = tensor.astype(np.float32)
            mapped_key = tensor_map.get_name(key, try_suffixes=(".weight", ".bias"))
            if mapped_key is None:
                mapped_key = key
            writer.add_tensor(mapped_key, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully generated GGUF model: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SafeTensors model to GGUF format for Ollama")
    parser.add_argument("--input_dir", type=str, default="checkpoints/export_hf", help="Input directory containing config.json and model.safetensors")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--output", type=str, default="checkpoints/paragai.gguf", help="Output .gguf file path")
    args = parser.parse_args()

    convert_to_gguf(args.input_dir, args.tokenizer, args.output)


if __name__ == "__main__":
    main()
