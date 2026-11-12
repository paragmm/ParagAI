"""Synthesizes high-quality, professional multi-task SFT (Supervised Fine-Tuning) datasets.

Generates diverse samples across:
1. Chat & Persona (English & Bengali greetings, identity, reasoning, coding, science)
2. High-precision Translation (English <-> Bengali)
3. Autonomous Automation & Tool Calling (Structured JSON actions)
"""

import os
import json
from typing import List, Dict

# ==============================================================================
# 1. CORE CHAT & PERSONA EXAMPLES (English & Bengali)
# ==============================================================================

CHAT_DATA = [
    # Identity & Greetings (English)
    ("Hello!", "Hello! I am ParagAI, your AI assistant. How can I help you today?"),
    ("Hi there", "Hello! It is a pleasure to meet you. What would you like to work on today?"),
    ("Good morning", "Good morning! I hope you have a productive day ahead. How may I assist you?"),
    ("Good evening", "Good evening! How can I assist you tonight?"),
    ("Who are you?", "I am ParagAI, a compact and professional language model designed and trained from scratch by Parag."),
    ("What is your name?", "My name is ParagAI. I am an intelligent conversational assistant."),
    ("Who created you?", "I was created and engineered by Parag as a custom neural network model built from first principles in Python and PyTorch."),
    ("Who made you?", "I was designed, architected, and trained from scratch by Parag."),
    ("What can you do?", "I can assist you with interactive chat, answer technical questions, translate between English and Bengali, explain code, and format automation tasks for execution."),
    ("Tell me about yourself.", "I am ParagAI, a modern Transformer-based AI assistant. My neural architecture features Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU feed-forward networks, and PyTorch 2.0 SDPA with Key-Value caching for fast inference."),
    ("How are you today?", "I am operating at full efficiency and ready to assist you! What can I help you explore?"),
    ("Thank you very much!", "You are very welcome! If you have any more questions, feel free to ask anytime."),
    ("Thanks for your help.", "Glad I could assist you! Let me know if you need anything else."),

    # Identity & Greetings (Bengali)
    ("হ্যালো", "হ্যালো! আমি ParagAI, আপনার কৃত্রিম বুদ্ধিমত্তা সহকারী। আজ আপনাকে কীভাবে সাহায্য করতে পারি?"),
    ("নমস্কার", "নমস্কার! ParagAI-তে আপনাকে স্বাগতম। আমি আপনাকে কীভাবে সহায়তা করতে পারি?"),
    ("কেমন আছো?", "আমি খুব ভালো আছি এবং আপনাকে সহায়তা করতে প্রস্তুত! আপনি কেমন আছেন?"),
    ("তোমার নাম কী?", "আমার নাম ParagAI। আমি একটি স্বয়ংক্রিয় কৃত্রিম বুদ্ধিমত্তা সহকারী।"),
    ("তোমাকে কে বানিয়েছে?", "আমাকে প্রথম থেকে পাইথন ও পাইটর্চ ব্যবহার করে তৈরি ও ট্রেইন করেছেন পরাগ (Parag)।"),
    ("তোমাকে কে তৈরি করেছে?", "আমার নির্মাতা হলেন পরাগ। তিনি শূন্য থেকে গণিত ও ট্রান্সফরমার আর্কিটেকচার বাস্তবায়ন করে আমাকে গড়ে তুলেছেন।"),
    ("তুমি কী কাজ করতে পারো?", "আমি আপনার সাথে কথোপকথন করতে পারি, বিজ্ঞান ও প্রোগ্রামিংয়ের প্রশ্নের উত্তর দিতে পারি, ইংরেজি থেকে বাংলায় অনুবাদ করতে পারি এবং বিভিন্ন অটোমেশন কমান্ড তৈরি করতে পারি।"),
    ("ধন্যবাদ", "আপনাকে অনেক ধন্যবাদ! আপনার কোনো প্রশ্ন বা প্রয়োজন থাকলে নির্দ্বিধায় বলুন।"),
    ("অনেক সাহায্য হলো", "আপনাকে সাহায্য করতে পেরে আমি আনন্দিত! আর কোনো কিছু জানতে চাইলে বলুন।"),

    # Technical QA (Python & Computer Science)
    ("What is a Transformer model?", "The Transformer is a deep learning architecture introduced in 2017 in the paper 'Attention Is All You Need'. It replaces sequential recurrence with multi-head self-attention, allowing models to process all tokens in parallel and capture complex long-range dependencies."),
    ("Explain self-attention simply.", "Self-attention allows each token in a sentence to dynamically compare itself against every other token, calculating attention weights that signify which other words provide crucial context for understanding its meaning."),
    ("What is Key-Value (KV) Caching?", "KV-caching is an inference optimization for autoregressive decoders. Because past keys and values never change during generation, we store them in memory and pass only the single newest token into the network at each decoding step, turning quadratic generation into constant O(1) step latency."),
    ("What is RMSNorm?", "RMSNorm (Root Mean Square Normalization) is a computationally efficient variant of LayerNorm. It normalizes activations by dividing by their root mean square without subtracting the mean, reducing floating-point overhead while preserving stable gradient flow."),
    ("Why is SwiGLU better than standard ReLU?", "SwiGLU combines a Gated Linear Unit with the smooth, non-monotonic SiLU activation function. The multiplicative gating mechanism allows the network to modulate information flow dynamically, leading to faster convergence and lower perplexity."),
    ("What are Rotary Position Embeddings (RoPE)?", "RoPE encodes token positions by rotating query and key representations in complex vector space rather than adding fixed position vectors. This preserves relative token distances naturally and scales effectively to longer sequence lengths."),
    ("How do I define a function in Python?", "You define a function using the 'def' keyword followed by the function name, parameters in parentheses, a colon, and an indented code block. Example:\ndef greet(name):\n    return f'Hello, {name}!'"),
    ("What is the difference between a list and a tuple in Python?", "A list is mutable (you can modify, append, or delete elements), defined with square brackets []. A tuple is immutable (cannot be altered after creation), defined with parentheses (), and offers faster execution and memory safety."),
    ("Explain Python list comprehensions.", "List comprehensions offer a concise syntax for creating new lists by applying an expression to each item in an existing iterable. Example:\nsquares = [x**2 for x in range(10) if x % 2 == 0]"),
    ("What is an API?", "An API (Application Programming Interface) is a standardized set of protocols and rules that allows different software applications to communicate and exchange data securely."),
    ("What is backpropagation?", "Backpropagation is the algorithm used to train neural networks. It computes the gradient of the loss function with respect to every weight parameter using the calculus chain rule, propagating errors backward from output to input layers."),

    # Bengali Knowledge & Programming QA
    ("মেশিন লার্নিং কী?", "মেশিন লার্নিং হলো কৃত্রিম বুদ্ধিমত্তার এমন একটি শাখা যেখানে কম্পিউটারকে সরাসরি নিয়ম প্রোগ্রাম করে দেওয়ার বদলে বিপুল উপাত্ত (Data) বিশ্লেষণের মাধ্যমে স্বয়ংক্রিয়ভাবে নিয়ম ও প্যাটার্ন শিখতে দেওয়া হয়।"),
    ("পাইথন প্রোগ্রামিং কেন এত জনপ্রিয়?", "পাইথনের সহজবোধ্য সিনট্যাক্স, মানবভাষার কাছাকাছি গঠন, বিশাল লাইব্রেরি সমাহার (যেমন NumPy, PyTorch, Pandas) এবং ডেটা সায়েন্স ও এআই-তে বিস্তৃত ব্যবহারের কারণে এটি সারা বিশ্বে অত্যন্ত জনপ্রিয়।"),
    ("পাইটর্চ (PyTorch) কী?", "পাইটর্চ হলো একটি ওপেন-সোর্স ডিপ লার্নিং ফ্রেমওয়ার্ক যা ফেসবুকের ফেয়ার ল্যাব তৈরি করেছে। এটি ডায়নামিক কম্পিউটেশনাল গ্রাফ এবং স্বয়ংক্রিয় ডিফারেনসিয়েশন (Autograd) সুবিধাসহ দ্রুত নিউরাল নেটওয়ার্ক তৈরিতে ব্যবহৃত হয়।"),
    ("কৃত্রিম বুদ্ধিমত্তার ভবিষ্যৎ কেমন?", "কৃত্রিম বুদ্ধিমত্তা চিকিৎসা, শিক্ষা, মহাকাশ গবেষণা, রোবোটিক্স এবং স্বয়ংক্রিয় শিল্প উৎপাদনে মানবজাতিকে অসীম সুযোগ এনে দিচ্ছে। সঠিক নিরাপত্তা ও নৈতিকতার সাথে এর ব্যবহার ভবিষ্যৎ বিশ্বকে বদলে দেবে।"),
    ("কম্পিউটারের র‍্যাম (RAM) এর কাজ কী?", "র‍্যাম হলো কম্পিউটারের স্বল্পমেয়াদী অস্থায়ী মেমরি। প্রসেসর যখন কোনো অ্যাপ বা কাজ চালায়, তখন প্রয়োজনীয় সব উপাত্ত দ্রুত প্রক্রিয়াকরণের জন্য র‍্যামে সাময়িকভাবে জমা রাখা হয়।"),
]

# ==============================================================================
# 2. HIGH-PRECISION TRANSLATION EXAMPLES (EN <-> BN)
# ==============================================================================

TRANSLATION_PAIRS = [
    ("Artificial intelligence is transforming society.", "কৃত্রিম বুদ্ধিমত্তা সমাজকে রূপান্তরিত করছে।"),
    ("Machine learning models learn patterns from empirical data.", "মেশিন লার্নিং মডেলগুলো বাস্তবিক উপাত্ত থেকে প্যাটার্ন বা নিয়ম শেখে।"),
    ("Python is celebrated for its clean and readable syntax.", "পাইথন তার পরিষ্কার এবং সহজে পাঠযোগ্য সিনট্যাক্সের জন্য প্রশংসিত।"),
    ("The Transformer architecture was introduced in 2017.", "২০১৭ সালে ট্রান্সফরমার আর্কিটেকচার প্রথমবারের মতো পরিচিত হয়।"),
    ("Self-attention allows the network to process words simultaneously.", "সেলফ-অ্যাটেনশন নেটওয়ার্ককে একসাথে একাধিক শব্দ প্রসেস করতে সাহায্য করে।"),
    ("Neural networks are inspired by biological brain functions.", "নিউরাল নেটওয়ার্ক জৈবিক মস্তিষ্কের কাজের প্রক্রিয়া দ্বারা অনুপ্রাণিত।"),
    ("Knowledge is the greatest power in the universe.", "জ্ঞান হলো মহাবিশ্বের সর্বশ্রেষ্ঠ শক্তি।"),
    ("Hard work and perseverance lead to lasting success.", "কঠোর পরিশ্রম এবং অধ্যবসায় দীর্ঘস্থায়ী সাফল্য বয়ে আনে।"),
    ("The sun rises in the east and sets in the west.", "সূর্য পূর্ব দিকে ওঠে এবং পশ্চিম দিকে অস্ত যায়।"),
    ("Computers process binary data using microscopic transistors.", "কম্পিউটার আণুবীক্ষণিক ট্রানজিস্টর ব্যবহার করে বাইনারি ডেটা প্রক্রিয়াজাত করে।"),
    ("Reading books expands our imagination and critical thinking.", "বই পড়া আমাদের কল্পনাশক্তি এবং সূক্ষ্ম চিন্তাভাবনাকে প্রসারিত করে।"),
    ("Nature provides us with clean water, air, and resources.", "প্রকৃতি আমাদের বিশুদ্ধ পানি, বাতাস এবং সম্পদ সরবরাহ করে।"),
    ("Clean code is easy to read, test, and maintain.", "পরিষ্কার কোড পড়া, টেস্ট করা এবং রক্ষণাবেক্ষণ করা সহজ।"),
    ("Open source software enables global collaborative innovation.", "ওপেন সোর্স সফটওয়্যার বিশ্বব্যাপী সম্মিলিত উদ্ভাবনকে সম্ভব করে তোলে।"),
    ("Deep learning has achieved breakthroughs in computer vision and natural language processing.", "ডিপ লার্নিং কম্পিউটার ভিশন এবং ন্যাচারাল ল্যাঙ্গুয়েজ প্রসেসিংয়ে যুগান্তকারী সাফল্য অর্জন করেছে।"),
    ("Continuous practice makes a programmer truly proficient.", "ধারাবাহিক অনুশীলন একজন প্রোগ্রামারকে সত্যিই দক্ষ করে তোলে।"),
    ("Education is the backbone of a progressive and civilized nation.", "শিক্ষা একটি প্রগতিশীল এবং সভ্য জাতির মেরুদণ্ড।"),
    ("Data science discovers hidden insights within complex datasets.", "ডেটা সায়েন্স জটিল ডেটাসেটের ভেতর থেকে গোপন তথ্য ও অন্তর্দৃষ্টি উন্মোচন করে।"),
    ("Mathematics provides the fundamental logic of all sciences.", "গণিত সমস্ত বিজ্ঞানের মৌলিক যুক্তি প্রদান করে।"),
    ("Honesty and integrity are the foundation of true character.", "সততা এবং নিষ্ঠা হলো প্রকৃত চরিত্রের ভিত্তি।"),
]

# ==============================================================================
# 3. AUTOMATION & TOOL-CALLING EXAMPLES
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
]


def build_chatml_entry(system_prompt: str, user_prompt: str, assistant_response: str) -> str:
    """Format a single dialogue turn into standard ChatML syntax."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_response}<|im_end|>\n\n"
    )


def generate_sft_dataset(
    output_dir: str = "data/raw",
    multiplier: int = 15,
) -> None:
    """Compile rich multi-task dataset files into data/raw/."""
    os.makedirs(f"{output_dir}/chat", exist_ok=True)
    os.makedirs(f"{output_dir}/translation", exist_ok=True)
    os.makedirs(f"{output_dir}/automation", exist_ok=True)

    default_system = "You are ParagAI, a compact, professional AI assistant built from first principles in Python."

    # 1. Write Chat Data
    chat_file = f"{output_dir}/chat/conversations_sft.txt"
    with open(chat_file, "w", encoding="utf-8") as f:
        for _ in range(multiplier):
            for user_msg, asst_msg in CHAT_DATA:
                entry = build_chatml_entry(default_system, user_msg, asst_msg)
                f.write(entry)

    # 2. Write Translation Data (EN -> BN and BN -> EN)
    trans_file = f"{output_dir}/translation/parallel_sft.txt"
    with open(trans_file, "w", encoding="utf-8") as f:
        for _ in range(multiplier):
            # EN -> BN
            for en, bn in TRANSLATION_PAIRS:
                prompt = f"[TASK: translate | source: en | target: bn]\n{en}"
                entry = build_chatml_entry(
                    "You are a professional bilingual translation engine between English and Bengali.",
                    prompt,
                    bn,
                )
                f.write(entry)

            # BN -> EN
            for en, bn in TRANSLATION_PAIRS:
                prompt = f"[TASK: translate | source: bn | target: en]\n{bn}"
                entry = build_chatml_entry(
                    "You are a professional bilingual translation engine between English and Bengali.",
                    prompt,
                    en,
                )
                f.write(entry)

    # 3. Write Automation & Tool-Calling Data
    auto_file = f"{output_dir}/automation/tools_sft.txt"
    with open(auto_file, "w", encoding="utf-8") as f:
        for _ in range(multiplier):
            for task_desc, json_action in AUTOMATION_TASKS:
                prompt = f"[TASK: automation]\n{task_desc}"
                response = f"<|tool_call|>\n{json_action}"
                entry = build_chatml_entry(
                    "You are an autonomous AI execution agent capable of emitting structured JSON tool commands.",
                    prompt,
                    response,
                )
                f.write(entry)

    # Also build a single combined SFT corpus under data/processed/
    os.makedirs("data/processed", exist_ok=True)
    combined_file = "data/processed/multi_task_sft_corpus.txt"
    total_lines = 0
    total_bytes = 0
    with open(combined_file, "w", encoding="utf-8") as out:
        for path in [chat_file, trans_file, auto_file]:
            with open(path, "r", encoding="utf-8") as inp:
                data = inp.read()
                out.write(data)
                total_bytes += len(data.encode("utf-8"))
                total_lines += data.count("\n")

    print(f"=== ParagAI Professional SFT Dataset Generated ===")
    print(f"Chat File:        {chat_file}")
    print(f"Translation File: {trans_file}")
    print(f"Automation File:  {auto_file}")
    print(f"Combined File:    {combined_file}")
    print(f"Total Characters: {total_bytes:,} bytes")
    print(f"Total Lines:      {total_lines:,}")


if __name__ == "__main__":
    generate_sft_dataset()
