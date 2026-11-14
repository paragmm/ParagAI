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

    b2u = gguf.bytes_to_unicode()

    def to_gpt2_token_bytes(raw_token) -> bytes:
        if isinstance(raw_token, str):
            if raw_token in ("<pad>", "<unk>", "<bos>", "<eos>"):
                return raw_token.encode("utf-8")
            raw_bytes = raw_token.encode("utf-8")
        elif isinstance(raw_token, (bytes, bytearray)):
            raw_bytes = bytes(raw_token)
        else:
            raw_bytes = str(raw_token).encode("utf-8")
        return "".join(b2u[b] for b in raw_bytes).encode("utf-8")

    if tok_data.get("type") == "bpe":
        vocab_dict = {int(k): v for k, v in tok_data["vocab"].items()}
        tokens = [to_gpt2_token_bytes(vocab_dict.get(i, "<unk>")) for i in range(len(vocab_dict))]
    else:
        itos = {int(k): v for k, v in tok_data["itos"].items()}
        tokens = [to_gpt2_token_bytes(itos.get(i, "<unk>")) for i in range(len(itos))]

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
    tok_type = tok_data.get("type", "char")
    writer.add_tokenizer_model("gpt2")

    token_types = []
    for i, tok in enumerate(tokens):
        if i in (0, 2, 3):  # pad, bos, eos
            token_types.append(int(gguf.TokenType.CONTROL))
        elif i == 1:  # unk
            token_types.append(int(gguf.TokenType.UNKNOWN))
        else:
            token_types.append(int(gguf.TokenType.NORMAL))

    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_scores([0.0] * len(tokens))

    # Add merges if BPE, or dummy merge if character-level (llama.cpp requires tokenizer.ggml.merges for gpt2)
    # Supports two merge formats:
    #   BPETokenizer  : list of {"pair": [p0, p1], "new_id": n}
    #   IndicBPETokenizer: list of [p0, p1, new_id]
    merges_list = []
    tokenizer_class = tok_data.get("tokenizer_class", "")
    if tok_type == "bpe" and "merges" in tok_data:
        for entry in tok_data["merges"]:
            if isinstance(entry, dict):
                p0, p1 = entry["pair"]
            else:
                p0, p1 = entry[0], entry[1]
            t0 = tok_data["vocab"].get(str(p0), "")
            t1 = tok_data["vocab"].get(str(p1), "")
            merges_list.append(f"{t0} {t1}")
    if not merges_list:
        merges_list = ["<pad> <unk>"]

    writer.add_token_merges(merges_list)

    writer.add_bos_token_id(2)
    writer.add_eos_token_id(3)
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(1)

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
