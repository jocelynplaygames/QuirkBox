# generate.py

import os
import time
import json
import urllib.request
import torch

from config.app_config import (
    MODEL_FILE, TOKENIZER_FILE, DEVICE,
    MODEL_CONTEXT_LENGTH, MAX_NEW_TOKENS, TEMPERATURE, TOP_K
)
from model import QuirkBox, generate, text_to_token_ids, token_ids_to_text
from tokenizer import Llama3Tokenizer, ChatFormat, clean_text


# 1. 读取 Prompt 模板
with open("config/prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)


# 2. 封装生成主函数
def run_generation(prompt_style: str, user_prompt: str) -> str:
    if prompt_style not in PROMPTS:
        raise ValueError(f"Prompt style '{prompt_style}' not found.")
    
    system_prompt = PROMPTS[prompt_style]["prompt"]

    # 模型文件
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE}...")
        url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"
        urllib.request.urlretrieve(url, MODEL_FILE)

    # tokenizer
    if not os.path.exists(TOKENIZER_FILE):
        url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"
        urllib.request.urlretrieve(url, TOKENIZER_FILE)

    # 配置模型
    if "1B" in MODEL_FILE:
        from model import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
    elif "3B" in MODEL_FILE:
        from model import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
    else:
        raise ValueError("Incorrect model file name")

    LLAMA32_CONFIG["context_length"] = MODEL_CONTEXT_LENGTH

    model = QuirkBox(LLAMA32_CONFIG)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    tokenizer = Llama3Tokenizer(TOKENIZER_FILE)
    tokenizer = ChatFormat(tokenizer, default_system=system_prompt)

    # 文本生成
    torch.manual_seed(42)
    input_ids = text_to_token_ids(user_prompt, tokenizer).to(DEVICE)
    output_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=MODEL_CONTEXT_LENGTH,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        return_attn = True
    )

    output_ids, attn_maps = output_ids
    output_text = token_ids_to_text(output_ids, tokenizer)
    return clean_text(output_text), attn_maps


# 3. CLI 测试（仅用于调试，未来可以独立成 cli.py）
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import numpy as np
    import json

    prompt_style = "joke"
    user_prompt = "Tell me a joke about robots."

    print(f"Running with style: {prompt_style}")
    start = time.time()

    result, attn_maps = run_generation(prompt_style, user_prompt)
    print(f"Time: {time.time() - start:.2f} sec\n")
    print("Output:\n", result)

    attn_data = [[head.cpu().tolist() for head in layer] for layer in attn_maps]

    with open("static/attn_map.json","w") as f: # 保存为：static/attn.json，供前端读取
        json.dump(attn_data,f)
    # plt.imshow(attn, cmap = 'viridis')
    # plt.title("Layer 1 Head 1 Attention")
    # plt.colorbar()
    # plt.show()