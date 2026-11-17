"""Synthesizes high-quality, professional English SFT (Supervised Fine-Tuning) datasets.

Generates diverse samples across:
1. Chat & Persona (English greetings, identity, reasoning, coding, science)
2. Technical QA (Python, Transformer architectures, Machine Learning)
3. Autonomous Automation & Tool Calling (Structured JSON actions)

All examples follow the strict ParagAI Chat Template:
<|system|>{system_message}<|end|>
<|user|>{user_message}<|end|>
<|assistant|>{assistant_response}<|end|>
"""

import os

# ==============================================================================
# 1. CORE ENGLISH CHAT & PERSONA EXAMPLES
# ==============================================================================

ENGLISH_CHAT_DATA = [
    # Identity & Greetings
    ("Hello!", "Hello! I am ParagAI, your AI assistant. How can I help you today?"),
    ("Hi there", "Hello! It is a pleasure to meet you. What would you like to work on today?"),
    ("Good morning", "Good morning! I hope you have a productive day ahead. How may I assist you?"),
    ("Good evening", "Good evening! How can I assist you tonight?"),
    (
        "Who are you?",
        "I am ParagAI, a compact and professional language model designed and trained from scratch by Parag.",
    ),
    ("What is your name?", "My name is ParagAI. I am an intelligent conversational assistant."),
    (
        "Who created you?",
        "I was created and engineered by Parag as a custom neural network model built from first principles in Python and PyTorch.",
    ),
    ("Who made you?", "I was designed, architected, and trained from scratch by Parag."),
    (
        "What can you do?",
        "I can assist you with interactive chat, answer technical questions, explain code, and format structured automation tasks for execution.",
    ),
    (
        "Tell me about yourself.",
        "I am ParagAI, a modern Transformer-based AI assistant. My neural architecture features Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU feed-forward networks, and PyTorch 2.0 SDPA with Key-Value caching for fast inference.",
    ),
    (
        "How are you today?",
        "I am operating at full efficiency and ready to assist you! What can I help you explore?",
    ),
    (
        "Thank you very much!",
        "You are very welcome! If you have any more questions, feel free to ask anytime.",
    ),
    ("Thanks for your help.", "Glad I could assist you! Let me know if you need anything else."),
    ("Are you an AI?", "Yes, I am ParagAI, an autoregressive language model built from scratch."),
    ("What is your purpose?", "My purpose is to assist developers and users with coding, reasoning, and automated system commands."),
    
    # Technical QA (Python & Computer Science)
    (
        "What is a Transformer model?",
        "The Transformer is a deep learning architecture introduced in 2017 in the paper 'Attention Is All You Need'. It replaces sequential recurrence with multi-head self-attention, allowing models to process all tokens in parallel and capture complex long-range dependencies.",
    ),
    (
        "Explain self-attention simply.",
        "Self-attention allows each token in a sentence to dynamically compare itself against every other token, calculating attention weights that signify which other words provide crucial context for understanding its meaning.",
    ),
    (
        "What is Key-Value (KV) Caching?",
        "KV-caching is an inference optimization for autoregressive decoders. Because past keys and values never change during generation, we store them in memory and pass only the single newest token into the network at each decoding step, turning quadratic generation into constant O(1) step latency.",
    ),
    (
        "What is RMSNorm?",
        "RMSNorm (Root Mean Square Normalization) is a computationally efficient variant of LayerNorm. It normalizes activations by dividing by their root mean square without subtracting the mean, reducing floating-point overhead while preserving stable gradient flow.",
    ),
    (
        "Why is SwiGLU better than standard ReLU?",
        "SwiGLU combines a Gated Linear Unit with the smooth, non-monotonic SiLU activation function. The multiplicative gating mechanism allows the network to modulate information flow dynamically, leading to faster convergence and lower perplexity.",
    ),
    (
        "What are Rotary Position Embeddings (RoPE)?",
        "RoPE encodes token positions by rotating query and key representations in complex vector space rather than adding fixed position vectors. This preserves relative token distances naturally and scales effectively to longer sequence lengths.",
    ),
    (
        "How do I define a function in Python?",
        "You define a function using the 'def' keyword followed by the function name, parameters in parentheses, a colon, and an indented code block. Example:\ndef greet(name):\n    return f'Hello, {name}!'",
    ),
    (
        "What is the difference between a list and a tuple in Python?",
        "A list is mutable (you can modify, append, or delete elements), defined with square brackets []. A tuple is immutable (cannot be altered after creation), defined with parentheses (), and offers faster execution and memory safety.",
    ),
    (
        "Explain Python list comprehensions.",
        "List comprehensions offer a concise syntax for creating new lists by applying an expression to each item in an existing iterable. Example:\nsquares = [x**2 for x in range(10) if x % 2 == 0]",
    ),
    (
        "What is an API?",
        "An API (Application Programming Interface) is a standardized set of protocols and rules that allows different software applications to communicate and exchange data securely.",
    ),
    (
        "What is backpropagation?",
        "Backpropagation is the algorithm used to train neural networks. It computes the gradient of the loss function with respect to every weight parameter using the calculus chain rule, propagating errors backward from output to input layers.",
    ),
    (
        "What is gradient descent?",
        "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction of the steepest decrease of the loss function to minimize prediction errors.",
    ),
    (
        "What is cross-entropy loss?",
        "Cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. It penalizes incorrect predictions with logarithmically increasing severity.",
    ),
    (
        "Why do we use AdamW instead of standard Adam?",
        "AdamW decouples weight decay from gradient updates. In standard Adam with L2 regularization, weights with large historical gradients decay slower, whereas AdamW applies direct proportional decay to all weights uniformly.",
    ),
    (
        "What is BPE tokenization?",
        "Byte-Pair Encoding (BPE) is a subword tokenization algorithm that iteratively merges the most frequent pairs of adjacent bytes or characters in a text corpus, creating a compact and expressive vocabulary.",
    ),
    (
        "What is temperature in text generation?",
        "Temperature scales the logits prior to the softmax calculation. Lower temperatures (e.g. 0.2 to 0.4) make the model more deterministic and focused, while higher temperatures (e.g. 0.7 to 1.0) increase diversity and randomness.",
    ),
    (
        "What is top-p nucleus sampling?",
        "Top-p nucleus sampling restricts candidate tokens to the smallest subset whose cumulative probability exceeds the threshold p. This cuts off the improbable long tail of tokens while adapting dynamically to the model's confidence.",
    ),
]

# ==============================================================================
# 2. AUTOMATION & TOOL-CALLING EXAMPLES (Pure JSON output)
# ==============================================================================

AUTOMATION_TASKS = [
    (
        "List all files currently modified in this git repository.",
        '{"action": "shell", "command": "git status --short"}',
    ),
    (
        "Show current disk space and free storage.",
        '{"action": "shell", "command": "df -h"}',
    ),
    (
        "Display active python processes on the machine.",
        '{"action": "shell", "command": "tasklist | findstr python"}',
    ),
    (
        "Check Python environment version and interpreter path.",
        '{"action": "shell", "command": "python --version"}',
    ),
    (
        "Run unit test suite using pytest.",
        '{"action": "shell", "command": "pytest tests/ -v"}',
    ),
    (
        "List all models currently installed in Ollama.",
        '{"action": "shell", "command": "ollama list"}',
    ),
    (
        "Show current network configuration and IP address.",
        '{"action": "shell", "command": "ipconfig"}',
    ),
    (
        "Search for all markdown files in the plans directory.",
        '{"action": "shell", "command": "dir /s /b .plans\\\\*.md"}',
    ),
    (
        "Count the total lines of code in the src directory.",
        '{"action": "shell", "command": "Get-ChildItem -Recurse src | Get-Content | Measure-Object -Line"}',
    ),
    (
        "Export the current trained model to GGUF format.",
        '{"action": "shell", "command": "python scripts/export_gguf.py --input_dir checkpoints/export_hf --output checkpoints/paragai.gguf"}',
    ),
    (
        "Run the ParagAI diagnostic doctor script.",
        '{"action": "shell", "command": "python scripts/doctor.py"}',
    ),
    (
        "Format code across the repository with black.",
        '{"action": "shell", "command": "black src/ scripts/ tests/"}',
    ),
]


def format_chat_entry(system_prompt: str, user_prompt: str, assistant_response: str) -> str:
    """Format a single dialogue turn into standard ParagAI Chat Template."""
    return (
        f"<|system|>{system_prompt}<|end|>\n"
        f"<|user|>{user_prompt}<|end|>\n"
        f"<|assistant|>{assistant_response}<|end|>\n\n"
    )


def generate_sft_dataset(
    output_file: str = "data/processed/sft_corpus.txt",
    multiplier: int = 20,
) -> None:
    """Compile uniform, English-only SFT dataset file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    default_system = "You are ParagAI, a compact, professional AI assistant."
    tool_system = "You are an autonomous AI execution agent capable of emitting structured JSON tool commands."

    entries = []

    # 1. English Chat & Technical QA
    for _ in range(multiplier):
        for user_msg, asst_msg in ENGLISH_CHAT_DATA:
            entries.append(format_chat_entry(default_system, user_msg, asst_msg))

    # 2. Automation & Tool Calling
    for _ in range(multiplier):
        for user_msg, json_action in AUTOMATION_TASKS:
            entries.append(format_chat_entry(tool_system, user_msg, json_action))

    with open(output_file, "w", encoding="utf-8") as out:
        for entry in entries:
            out.write(entry)

    total_bytes = sum(len(e.encode("utf-8")) for e in entries)
    print("=== ParagAI Standardized SFT Dataset Generated ===")
    print(f"Output File:      {output_file}")
    print(f"Total Dialogues:  {len(entries):,}")
    print(f"Total Size:       {total_bytes:,} bytes")


if __name__ == "__main__":
    generate_sft_dataset()
