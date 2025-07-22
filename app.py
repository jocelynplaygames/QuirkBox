


from flask import Flask, render_template, request, jsonify
import torch
from model import QuirkBox, generate, text_to_token_ids, token_ids_to_text
from tokenizer import Llama3Tokenizer, ChatFormat, clean_text
from prompt_templates import TEMPLATES
from config.app_config import MODEL_FILE, TOKENIZER_FILE, DEVICE
from generate import run_generation
import json

MODEL_FILE = "llama3.2-1B-instruct.pth"
TOKENIZER_FILE = "tokenizer.model"
MODEL_CONTEXT_LENGTH = 256
MAX_NEW_TOKENS = 10 #模型每次生成时 最多输出几个词（token）.意思是 —— 让模型「生成 10 个新的 token」。那整个 generate() 函数内部的逻辑就会重复执行 10 次。
TEMPERATURE = 0.0
TOP_K = 1

app = Flask(__name__)   # 运行的 Flask 服务
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 限制最大请求体为 1MB


# 下载权重 & tokenizer
import os, urllib.request

if not os.path.exists(MODEL_FILE):
    print("Downloading model...")
    urllib.request.urlretrieve(
        f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}",
        MODEL_FILE,
    )

if not os.path.exists(TOKENIZER_FILE):
    print("Downloading tokenizer...")
    urllib.request.urlretrieve(
        f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}",
        TOKENIZER_FILE,
    )

# 配置
from model import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
LLAMA32_CONFIG["context_length"] = MODEL_CONTEXT_LENGTH

# 初始化模型
model = QuirkBox(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
model.eval()
model.to("cpu")

# 初始化 tokenizer
tokenizer = ChatFormat(Llama3Tokenizer(TOKENIZER_FILE))


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 添加 favicon 路由，避免浏览器自动请求这个导致崩；204 表示“无内容”，浏览器就不会再请求了


@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json
    user_input = data.get("input", "")
    style = data.get("style", "casual")

    try:
        print("✅ Step 1: Received request")

        prompt = TEMPLATES.get(style, "Talk to me about:") + "\n" + user_input
        print("✅ Step 2: Built prompt:", prompt)

        with torch.no_grad():
            input_ids = text_to_token_ids(prompt, tokenizer).to("cpu")
            print("✅ Step 3: Encoded input. Shape:", input_ids.shape)

            output_ids, attn_maps = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                context_size=LLAMA32_CONFIG["context_length"],
                top_k=TOP_K,
                temperature=TEMPERATURE,
                return_attn=True
            )
            print("✅ Step 4: Generation done. Shape:", output_ids.shape)

            output_text = token_ids_to_text(output_ids, tokenizer)
            print("✅ Step 5: Decoded output text:", output_text)

            output_text = clean_text(output_text)
            print("✅ Step 6: Cleaned output")
        
        #保存attention JSON 到 static/attn,json
        attn_serialized = [
            [head.cpu().tolist() for head in layer] for layer in attn_maps
        ]
        with open("static/attn.json","w") as f:
            json.dump(attn_serialized, f)
        print("✅ Step 7: Saved attention map to static/attn.json")

        return jsonify({"output": output_text})
    # try:
    #     prompt = TEMPLATES.get(style, "Talk to me about:") + "\n" + user_input
    #     print("🧪 Prompt:", prompt)

    #     # mock 部分，用于排查是否是模型阻塞
    #     output_text = f"Echo: {prompt}"  # ✅ 临时替代真正的生成
    #     return jsonify({"output": output_text})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"output": f"[ERROR] {str(e)}"})




if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
#     #app.run(debug=True, use_reloader=False)